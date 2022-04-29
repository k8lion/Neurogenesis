export NeuroSearchSpace, NeuroSearchSpaceVGG11, NeuroSearchSpaceWRN28, NeuroSearchSpaceWRN28_new, NeuroVertex, NeuroVertexConv, NeuroVertexDense,
    unmaskneuron, unmaskoutputs, maskneuron, maskoutputs, layerparams, maskparams, auxparams, Wb, 
    getactiveinputindices, Wmasked, getactiveindices, getinactiveindices, countactiveneurons, getweights,
    gmrelu, togpu, countparams


using Flux 
using Flux: convfilter, gpu, cpu
using Zygote: @adjoint, ignore
using NNlib: scatter!
using LinearAlgebra: svd, norm
using Statistics: mean
using Random: GLOBAL_RNG, shuffle!
using Base.Iterators: partition
using CUDA

"""
Growable dense layer
"""
struct NeuroVertexDense
    layer::Dense 
    maskW::AbstractMatrix{Int} 
    maskb::AbstractVector{Int}
    activeinputneurons::Vector{Bool}
    activeneurons::Vector{Bool}
    isactive::Bool
    biasinit::Bool
    preact::Bool
end

function NeuroVertexDense(inputshape::Int, width::Int, inputactives::Int, actives::Int, σ = identity, rng = GLOBAL_RNG, biasinit::Bool = false, preact::Bool = false)
    dense = Dense(glorot_uniform(inputshape, width, inputactives, actives, rng), true, σ)
    densemaskW = zeros(Int, size(dense.weight))
    densemaskW[1:actives,1:inputactives] .= 1
    densemaskb = zeros(Int, size(dense.bias))
    densemaskb[1:actives] .= 1
    active = zeros(Bool, width)
    active[1:actives] .= 1
    inputactive = zeros(Bool, inputshape)
    inputactive[1:inputactives] .= 1

    NeuroVertexDense(dense, densemaskW, densemaskb, inputactive, active, any(active), biasinit, preact)
end

Flux.@functor NeuroVertexDense

Flux.trainable(nv::NeuroVertexDense) = Flux.trainable(nv.layer)
layerparams(nv::NeuroVertexDense) = Flux.params(nv.layer)
maskparams(nv::NeuroVertexDense) = Flux.params((nv.maskW, nv.maskb))

function (m::NeuroVertexDense)(x)
    W, b, sigma, MW, Mb = Wb(m)..., m.layer.σ, m.maskW, m.maskb 
    if ndims(x) > 2
        x = flatten(x)
    end
    if m.preact
        out = (W .* MW) * sigma.(x) .+ b .* Mb
    else
        out = sigma.((W .* MW) * x .+ b .* Mb)
    end
    return out
end

function (m::NeuroVertexDense)(x, sigma)
    W, b, MW, Mb = Wb(m)..., m.maskW, m.maskb 
    if ndims(x) > 2
        x = flatten(x)
    end
    if m.preact
        out = (W .* MW) * sigma.(x) .+ b .* Mb
    else
        out = sigma.((W .* MW) * x .+ b .* Mb)
    end
    return out
end

function (m::NeuroVertexDense)(aux::AbstractMatrix, x::AbstractMatrix, old_x::AbstractArray, prevm::NeuroVertexDense)
    W, b, σ, MW, Mb = Wb(m)..., m.layer.σ, m.maskW, m.maskb 
    if m.preact
        out = (W .* MW) * σ.(x) .+ b .* Mb .+ flatten(aux * flatten(σ.(old_x)))
    else
        out = σ.((W .* MW) * x .+ b .* Mb .+ flatten(σ.(aux * flatten(old_x))))
    end
    return out
end

function (m::NeuroVertexDense)(aux::AbstractMatrix, x::AbstractVector, old_x::AbstractVector, prevm::NeuroVertexDense)
    W, b, σ, MW, Mb = Wb(m)..., m.layer.σ, m.maskW, m.maskb
    if m.preact
        out = (W .* MW) * σ.(x) .+ b .* Mb .+ aux * σ.(old_x)
    else
        out = σ.((W .* MW) * x .+ b .* Mb .+ σ.(aux * old_x))
    end 
    return out
end

function Wb(nv::NeuroVertexDense) 
    return nv.layer.weight, nv.layer.bias
end

function Wmasked(nv::NeuroVertexDense, outlast=true, flattenouts=false, flattenins=false, trimin=true, trimout=true) 
    Wmask = nv.layer.weight.*nv.maskW |> cpu
    if trimin
        Wmask = Wmask[nv.activeneurons, :]
    end
    if trimout
        Wmask = Wmask[:, nv.activeinputneurons]
    end
    if !outlast
        Wmask = transpose(Wmask)
    end
    return Wmask
end

function assignweights(m::NeuroVertexDense, fanin, fanout, weights) 
    if isa(fanin, Int)
        fanin = fanin:fanin
    end 
    if isa(fanout, Int)
        fanout = fanout:fanout
    end
    weights = vec(weights)
    wzero = copy(weights)*0
    if isa(m.layer.weight, CuArray) & isa(weights, Array)
        weights = weights |> cu
        wzero = wzero |> cu
    end
    indices = [(i,j) for i in fanin for j in fanout]
    NNlib.scatter!(*, m.layer.weight, wzero, indices)
    NNlib.scatter!(+, m.layer.weight, weights, indices)
end

function addweights(m::NeuroVertexDense, fanin, fanout, weights::AbstractArray) 
    if isa(m.layer.weight, CuArray) & isa(weights, Array)
        weights = weights |> cu
    end
    m.layer.weight[fanin,fanout] .+= weights
end

function addweights(m::NeuroVertexDense, fanin, fanout, weight::Real) 
    if isa(fanin,Int) & isa(fanout,Int)
        fanin = fanin:fanin
    end
    m.layer.weight[fanin,fanout] .+= weight
end

function scaleweights(m::NeuroVertexDense, fanin, fanout, factor::Real)
    m.layer.weight[fanin,fanout] .*= factor
end

"""
Growable conv layer
"""
struct NeuroVertexConv
    layer::Conv 
    maskW::AbstractMatrix{Int} 
    maskb::AbstractVector{Int}
    activeinputneurons::Vector{Bool}
    activeneurons::Vector{Bool}
    isactive::Bool
    biasinit::Bool
    preact::Bool
    maxpool::Chain
    batchnorm::Chain
end

function NeuroVertexConv(filter::Tuple, inputshape::Int, width::Int, inputactives::Int, actives::Int, σ = identity, rng = GLOBAL_RNG, biasinit::Bool = false; maxpool = false, stride = 1, pad = 0, dilation = 1, groups = 1, preact = false, batchnorm=false)
    convmaskW = zeros(Int, width, inputshape)
    convmaskW[1:actives,1:inputactives] .= 1
    convmaskb = zeros(Int, width)
    convmaskb[1:actives] .= 1
    active = zeros(Bool, width)
    active[1:actives] .= 1
    inputactive = zeros(Bool, inputshape)
    inputactive[1:inputactives] .= 1
    W = zeros(Float32, filter..., inputshape, width)
    W[:,:,1:inputactives,1:actives] .= convfilter(filter, inputactives => actives)
    conv = Conv(W, true, σ, stride=stride, pad=pad, dilation=dilation, groups=groups)
    if maxpool
        mp = Chain(MaxPool((2,2)))
    else
        mp = Chain()
    end
    if batchnorm
        bn = Chain(BatchNorm(inputshape))
    else
        bn = Chain()
    end
    NeuroVertexConv(conv, convmaskW, convmaskb, inputactive, active, any(active), biasinit, preact, mp, bn)
end

Flux.@functor NeuroVertexConv

Flux.trainable(nv::NeuroVertexConv) = Flux.trainable(nv.layer)
layerparams(nv::NeuroVertexConv) = Flux.params(nv.layer)
maskparams(nv::NeuroVertexConv) = Flux.params((nv.maskW, nv.maskb))


function (m::NeuroVertexConv)(x)
    W = m.layer.weight
    σ, b, Mb = m.layer.σ, reshape(m.layer.bias, ntuple(_ -> 1, length(m.layer.stride))..., :, 1), reshape(m.maskb, ntuple(_ -> 1, length(m.layer.stride))..., :, 1)
    cdims = DenseConvDims(x, W; stride = m.layer.stride, padding = m.layer.pad, dilation = m.layer.dilation, groups = m.layer.groups)
    if !m.preact
        out = m.maxpool(Mb .* σ.(conv(x, W, cdims) .+ b))
    else
        out = m.maxpool(Mb .* conv(σ.(m.batchnorm(x)), W, cdims) .+ b)
    end
    return out
end

function (m::NeuroVertexConv)(x, sigma)
    W = m.layer.weight 
    b, Mb = reshape(m.layer.bias, ntuple(_ -> 1, length(m.layer.stride))..., :, 1), reshape(m.maskb, ntuple(_ -> 1, length(m.layer.stride))..., :, 1)
    cdims = DenseConvDims(x, W; stride = m.layer.stride, padding = m.layer.pad, dilation = m.layer.dilation, groups = m.layer.groups)
    if !m.preact
        out = m.maxpool(Mb .* sigma.(conv(x, W, cdims) .+ b))
    else
        out = m.maxpool(Mb .* conv(sigma.(m.batchnorm(x)), W, cdims) .+ b)
    end
    return out
end

function (m::NeuroVertexDense)(aux::AbstractArray, x::AbstractArray, old_x::AbstractArray, prevm::NeuroVertexConv)
    W, b, σ, MW, Mb = Wb(m)..., m.layer.σ, m.maskW, m.maskb 
    cdimsaux  = DenseConvDims(old_x, aux; stride = prevm.layer.stride, padding = prevm.layer.pad, dilation = prevm.layer.dilation, groups = prevm.layer.groups)
    if !m.preact
        return σ.((W .* MW) * flatten(x) .+ b .* Mb .+ flatten(prevm.maxpool(σ.(conv(old_x, aux, cdimsaux)))))
    else
        return (W .* MW) * flatten(σ.(x)) .+ b .* Mb .+ flatten(prevm.maxpool(conv(σ.(prevm.batchnorm(old_x)), aux, cdimsaux)))
    end
end

function (m::NeuroVertexConv)(aux::AbstractArray, x::AbstractArray, old_x::AbstractArray, prevm::NeuroVertexConv)
    W = m.layer.weight 
    σ, b, Mb = m.layer.σ, reshape(m.layer.bias, ntuple(_ -> 1, length(m.layer.stride))..., :, 1), reshape(m.maskb, ntuple(_ -> 1, length(m.layer.stride))..., :, 1)
    cdims = DenseConvDims(x, W; stride = m.layer.stride, padding = m.layer.pad, dilation = m.layer.dilation, groups = m.layer.groups)
    auxpad = (prevm.layer.pad[1]+m.layer.pad[1], prevm.layer.pad[2]+m.layer.pad[1])
    cdimsaux  = DenseConvDims(old_x, aux; stride = max(prevm.layer.stride, m.layer.stride), padding = auxpad, dilation = prevm.layer.dilation, groups = prevm.layer.groups)
    if !m.preact
        out = m.maxpool(σ.(Mb .* (conv(x, W, cdims) .+ b) .+ prevm.maxpool(σ.(conv(old_x, aux, cdimsaux)))))
    else
        out = m.maxpool(Mb .* (conv(σ.(m.batchnorm(x)), W, cdims) .+ b) .+ prevm.maxpool(conv(σ.(prevm.batchnorm(old_x)), aux, cdimsaux)))
    end
    return out
end

function Wb(nv::NeuroVertexConv) 
    return nv.layer.weight, nv.layer.bias
end

function Wmasked(m::NeuroVertexConv, outlast=true, flattenouts=false, flattenins=false, trimin=true, trimout=true) 
    W, MW = m.layer.weight |> cpu, transpose(m.maskW) |> cpu
    Wmask = mapslices(w -> w.*MW, W, dims=(3,4))
    if trimin
        Wmask = Wmask[:,:,:,getactiveindices(m)]
    end
    if trimout
        Wmask = Wmask[:,:,getactiveinputindices(m),:]
    end
    if flattenouts
        Wmask = flatten(Wmask)
        if outlast
            Wmask = transpose(Wmask)
        end
    elseif outlast
        Wmask = permutedims(Wmask,(1,2,4,3))
    end
    if flattenins
        Wmask = flatten(Wmask)
    end
    return Wmask
end

function assignweights(m::NeuroVertexConv, fanin, fanout, weights) 
    if isa(fanin, Int)
        fanin = fanin:fanin
    end 
    if isa(fanout, Int)
        fanout = fanout:fanout
    end
    weights = vec(weights)
    weightszero = copy(weights)*0
    if isa(m.layer.weight, CuArray) & isa(weights, Array)
        weights = weights |> cu
        weightszero = weightszero |> cu
    end
    kernel = size(m.layer.weight)[1:2]
    indices = [(i,j,k,l) for i in 1:kernel[1] for j in 1:kernel[2] for k in fanout for l in fanin]
    NNlib.scatter!(*, m.layer.weight, weightszero, indices)
    NNlib.scatter!(+, m.layer.weight, weights, indices)
end

function addweights(m::NeuroVertexConv, fanin, fanout, weights::AbstractArray) 
    weights = reshape(weights, size(m.layer.weight[:,:,fanout,fanin]))
    if isa(m.layer.weight, CuArray) & isa(weights, Array)
        weights = weights |> cu
    end
    m.layer.weight[:,:,fanout,fanin] .+= weights
end

function addweights(m::NeuroVertexConv, fanin, fanout, weight::Real)
    m.layer.weight[:,:,fanout:fanout,fanin] .+= weight
end

function scaleweights(m::NeuroVertexConv, fanin, fanout, factor::Real)
    m.layer.weight[:,:,fanout,fanin] .*= factor
end

mutable struct ActivationsStore
    currentacts::Dict{Int,AbstractArray}
end

"""
Growable group of residual blocks, such that all channels are tied and statistics are based on first layer of group
"""
struct NeuroVertexSequence
    layers::AbstractVector{NeuroVertexConv}
    skips::AbstractVector{Any}
    acts::ActivationsStore
    pool::Chain
    skipscale::Float32
end

Flux.@functor NeuroVertexSequence

Flux.trainable(nv::NeuroVertexSequence) = Flux.trainable(vcat(nv.layers, nv.skips))

NeuroVertex = Union{NeuroVertexDense, NeuroVertexConv, NeuroVertexSequence}

function getlayer(nv::NeuroVertex)
    if isa(nv, NeuroVertexSequence)
        return nv.layers[1].layer
    end
    return nv.layer 
end

function assignbias(m::NeuroVertex, fanin, bias::AbstractVector) 
    if !isa(m.layer.bias, AbstractVector)
        return
    end
    if isa(m.layer.bias, CuArray) & isa(bias, Array)
        bias = bias |> cu
    end
    m.layer.bias[fanin] .= bias
end

function assignbias(m::NeuroVertex, fanin, bias::Real) 
    if !isa(m.layer.bias, AbstractVector)
        return
    end
    if isa(fanin,Int)
        fanin = fanin:fanin
    end
    m.layer.bias[fanin] .= bias
end

function assignweightmask(m::NeuroVertex, fanin, fanout, values::AbstractArray) 
    if isa(m.layer.maskW, CuArray) & isa(values, Array)
        values = values |> cu
    end
    m.maskW[fanin,fanout] .= values
end

function assignweightmask(m::NeuroVertex, fanin, fanout, value::Int) 
    if isa(fanin,Int) & isa(fanout,Int)
        fanin = fanin:fanin
    end
    m.maskW[fanin,fanout] .= value
end

function assignbiasmask(m::NeuroVertex, fanin, values::AbstractVector) 
    if isa(m.layer.maskb, CuArray) & isa(values, Array)
        values = values |> cu
    end
    m.maskb[fanin] .= values
end

function assignbiasmask(m::NeuroVertex, fanin, value::Real) 
    if isa(fanin,Int)
        fanin = fanin:fanin
    end
    m.maskb[fanin] .= value
end

function togpu(nv::NeuroVertexDense)
    return NeuroVertexDense(Flux.gpu(nv.layer), nv.maskW |> cu, nv.maskb |> cu, nv.activeinputneurons, nv.activeneurons, nv.isactive, nv.biasinit, nv.preact)
end

function togpu(nv::NeuroVertexConv)
    return NeuroVertexConv(Flux.gpu(nv.layer), nv.maskW |> cu, nv.maskb |> cu, nv.activeinputneurons, nv.activeneurons, nv.isactive, nv.biasinit, nv.preact, gpu(nv.maxpool), gpu(nv.batchnorm))
end

function togpu(nvs::NeuroVertexSequence)
    return NeuroVertexSequence(togpu.(nvs.layers), [isa(skip, NeuroVertex) || isa(skip, Conv) ? togpu(skip) : skip for skip in nvs.skips], nvs.acts, gpu(nvs.pool), nvs.skipscale)
end

function getweights(m::NeuroVertexSequence)
    return m.layers[1].layer.weight
end

function getweights(m::NeuroVertex)
    return m.layer.weight
end

function isactive(nv::NeuroVertex)
    return nv.isactive
end

function getinactiveindices(nv::NeuroVertex, numindices::Int = -1)
    inactives = findall(==(0), nv.activeneurons)
    if numindices == -1
        return inactives
    end
    return inactives[1:minimum((numindices, length(inactives)))]
end

function countactiveneurons(nv::NeuroVertex)
    if isa(nv, NeuroVertexSequence)
        return countactiveneurons(nv.layers[1])
    end
    return sum(nv.activeneurons)
end

function getactiveindices(nv::NeuroVertex)
    return findall(==(1), nv.activeneurons)
end

function getactiveinputindices(nv::NeuroVertex)
    return findall(==(1), nv.activeinputneurons)
end

# Unmask neuron
function unmaskneuron(m::NeuroVertex, neuron::Union{Int,Vector{Int}}, init = glorot_uniform, rng=GLOBAL_RNG, rescale::Bool=false, device = cpu)
    if rescale & (init != zero_init)
        oldmeannorm = meannorm(m)
    end
    
    assignweightmask(m, neuron, findall(m.activeinputneurons), 1)
    assignbiasmask(m, neuron, 1)

    if isa(neuron, Int) 
        weights = init(m, neuron, sum(m.activeinputneurons), true, rng)
        if rescale & (init != zero_init)
            weights .*= oldmeannorm / vecnorm(weights)
        end
        m.activeneurons[neuron] = 1
    else
        weights = hcat([init(m, n, sum(m.activeinputneurons), true, rng) for n in neuron]...)
        if rescale & (init != zero_init)
            weights .*= oldmeannorm ./ vecnorm(weights, 1)
        end
        m.activeneurons[neuron] .= 1
    end

    assignweights(m, neuron, findall(m.activeinputneurons), weights)

    if m.biasinit
        assignbias(m, neuron, randn(rng, legnth(neuron)))
    else
        assignbias(m, neuron, 0)
    end
end

function unmaskneuron(m::NeuroVertex, weights::AbstractArray, neuron::Union{Int,Vector{Int}}, rescale::Bool=false, rng=GLOBAL_RNG, bias::Float32=0f0)
    if rescale
        if isa(neuron, Int)
            weights .*= meannorm(m)/vecnorm(vec(weights))
        else
            if ndims(weights) == 2
                weights .*= meannorm(m)./vecnorm(weights, 2)
            else
                weights .*= meannorm(m)./vecnorm(weights, (1,2,3))
            end
        end
    end

    assignweights(m, neuron, findall(m.activeinputneurons), weights)
    
    assignbias(m, neuron, bias)

    assignweightmask(m, neuron, findall(m.activeinputneurons), 1)
    
    assignbiasmask(m, neuron, 1)

    if isa(neuron, Int)
        m.activeneurons[neuron] = 1
    else
        m.activeneurons[neuron] .= 1
    end
end


function unmaskoutputs(m::NeuroVertex, neuron::Union{Int,Vector{Int}}, init = glorot_uniform, rng = GLOBAL_RNG, rescale::Bool=false)
    if rescale & (init != zero_init)
        oldmeannorm = meannorm(m, false)
    end

    assignweightmask(m, findall(m.activeneurons), neuron, 1)

    if isa(neuron, Int) 
        weights = init(m, neuron, sum(m.activeneurons), false, rng)
        if rescale & (init != zero_init)
            weights .*= oldmeannorm / vecnorm(weights)
        end
        m.activeinputneurons[neuron] = 1
    else
        weights = hcat([init(m, n, sum(m.activeneurons), false, rng) for n in neuron]...)
        if rescale & (init != zero_init)
            weights .*= oldmeannorm ./ vecnorm(weights, 2)
        end
        m.activeinputneurons[neuron] .= 1
    end

    assignweights(m, findall(m.activeneurons), neuron, weights)
end

function unmaskoutputs(m::NeuroVertex, weights::AbstractArray, neuron::Union{Int,Vector{Int}}, rescale::Bool=false)
    if rescale
        if isa(neuron, Int)
            weights .*= meannorm(m, false)/vecnorm(weights)
        elseif ndims(weights) == 2
            weights .*= meannorm(m, false)./vecnorm(weights, 2)
        else
            weights .*= meannorm(m, false)./vecnorm(weights, (1,2,3))
        end
    end

    assignweightmask(m, findall(m.activeneurons), neuron, 1)

    assignweights(m, findall(m.activeneurons), neuron, weights)

    if isa(neuron, Int)
        m.activeinputneurons[neuron] = 1
    else
        m.activeinputneurons[neuron] .= 1
    end
end

function unmasklayer(nv::NeuroVertex)
    nv.isactive = true
end

function maskneuron(m::NeuroVertex, neuron::Union{Int,Vector{Int}})
    assignweightmask(m, neuron, :, 0)
    assignbiasmask(m, neuron, 0)

    if isa(neuron, Int)
        m.activeneurons[neuron] = 0
    else
        m.activeneurons[neuron] .= 0
    end
end

function maskoutputs(m::NeuroVertex, neuron::Union{Int,Vector{Int}})
    assignweightmask(m, :, neuron, 0)

    if isa(neuron, Int)
        m.activeinputneurons[neuron] = 0
    else
        m.activeinputneurons[neuron] .= 0
    end
end

function getinactiveindices(nv::NeuroVertexSequence, numindices::Int = -1)
    return getinactiveindices(nv.layers[1], numindices)
end

function countactiveneurons(nv::NeuroVertexSequence)
    return countactiveneurons(nv.layers[1])
end

function getactiveindices(nv::NeuroVertexSequence)
    return getactiveindices(nv.layers[1])
end

function getactiveinputindices(nv::NeuroVertexSequence)
    return getactiveinputindices(nv.layers[1])
end

function unmaskneuron(m::NeuroVertexSequence, neuron::Union{Int,Vector{Int}}, init = glorot_uniform, rng=GLOBAL_RNG, rescale::Bool=false, device = cpu)
    unmaskneuron(m.layers[1], neuron, init, rng, rescale)
    [unmaskneuron(l, neuron, glorot_uniform, rng, rescale) for l in m.layers[2:end]]
    [unmaskoutputs(l, neuron, zero_init, rng, rescale) for l in m.layers[2:end]]
    if isa(m.skips[1], NeuroVertex)
        unmaskneuron(m.skips[1], neuron, init, rng, rescale)
    end
end

function unmaskneuron(m::NeuroVertexSequence, weights::AbstractArray, neuron::Union{Int,Vector{Int}}, rescale::Bool=false, rng=GLOBAL_RNG, bias::Float32=0f0)
    unmaskneuron(m.layers[1], weights, neuron, rescale, bias)
    [unmaskneuron(l, neuron, glorot_uniform, rng, rescale) for l in m.layers[2:end]]
    [unmaskoutputs(l, neuron, zero_init, rng, rescale) for l in m.layers[2:end]]
    if isa(m.skips[1], NeuroVertex)
        unmaskneuron(m.skips[1], neuron, glorot_uniform, rng, rescale)
    end
end

function unmaskoutputs(m::NeuroVertexSequence, neuron::Union{Int,Vector{Int}}, init = glorot_uniform, rng = GLOBAL_RNG, rescale::Bool=false)
    unmaskoutputs(m.layers[1], neuron, init, rng, rescale)
    if isa(m.skips[1], NeuroVertex)
        unmaskoutputs(m.skips[1], neuron, init, rng, rescale)
    end
end

function unmaskoutputs(m::NeuroVertexSequence, weights::AbstractArray, neuron::Union{Int,Vector{Int}}, rescale::Bool=false, rng=GLOBAL_RNG)
    unmaskoutputs(m.layers[1], weights, neuron, rescale)
    if isa(m.skips[1], NeuroVertex)
        unmaskoutputs(m.skips[1], neuron, glorot_uniform, rng, rescale)
    end
end

function maskneuron(m::NeuroVertexSequence, neuron::Union{Int,Vector{Int}})
    [maskneuron(l, neuron) for l in m.layers]
    [maskoutputs(l, neuron) for l in m.layers[2:end]]
    [maskneuron(l, neuron) for l in m.skips if isa(l, NeuroVertex)]
end

function maskoutputs(m::NeuroVertexSequence, neuron::Union{Int,Vector{Int}})
    maskoutputs(m.layers[1], neuron)
    if isa(m.skips[1], NeuroVertex)
        maskoutputs(m.skips[1], neuron)
    end
end

function assignweights(m::NeuroVertexSequence, fanin, fanout, weights) 
    assignweights(m.layers[1], fanin, fanout, weights) 
end

function addweights(m::NeuroVertexSequence, fanin, fanout, weights) 
    addweights(m.layers[1], fanin, fanout, weights) 
end

function scaleweights(m::NeuroVertexSequence, fanin, fanout, factor)
    scaleweights(m.layers[1], fanin, fanout, factor)
end

function Wb(nvs::NeuroVertexSequence) 
    return Wb(nvs.layers[1])
end

function Wmasked(nvs::NeuroVertexSequence, outlast=true, flattenouts=false, flattenins=false, trimin=true, trimout=true) 
    return Wmasked(nvs.layers[1], outlast, flattenouts, flattenins, trimin, trimout)
end

function (m::NeuroVertexSequence)(x::AbstractArray; depth::Int = -1, saveacts::Bool=false, skips::Bool=true)
    if length(m.layers) == 4
        for i in 1:length(m.layers)
            x = m.layers[i](x)
        end
    else
        for i in 1:floor(Int, length(m.layers)/2)
            nv1 = m.layers[i*2-1]
            nv2 = m.layers[i*2]
            skip = m.skips[i]
            x_h = nv1(x)
            x_o = nv2(x_h)
            if skips
                x_o += m.skipscale*skip(x)
            end
            x = x_o
            if saveacts
                m.acts.currentacts[2*i-1] = x_h
                m.acts.currentacts[2*i] = x
            else
                m.acts.currentacts[2*i-1] = []
                m.acts.currentacts[2*i] = []
            end
        end
    end
    return m.pool(x)
end

function (m::NeuroVertexSequence)(aux::AbstractArray, x::AbstractArray, old_x::AbstractArray, prevm::NeuroVertex; firstaux::AbstractArray = [])
    for i in 1:floor(Int, length(m.layers)/2)
        nv1 = m.layers[i*2-1]
        nv2 = m.layers[i*2]
        skip = m.skips[i]
        if i == 1 
            if !isa(prevm, NeuroVertexSequence)
                x_h = nv1(firstaux, x, old_x, prevm)
            else
                x_h = nv1(x)
            end
            if ndims(aux) > 1
                x = nv2(aux, x_h, x, nv1) + skip(x)
            else
                x = nv2(x_h) + skip(x)
            end
        else
            x_h = nv1(x)
            x = nv2(nv1(x))+skip(x)
        end
    end
    return m.pool(x)
end

"""
Growable model
"""
struct NeuroSearchSpace
    model::AbstractVector{Union{NeuroVertex,Chain}}
    acts::ActivationsStore
    auxs::AbstractVector{AbstractArray}
    conversion::Tuple{Int,Int}
    device
end

function NeuroSearchSpace(widths::AbstractVector{Int}, activeneurons::AbstractVector{Int}, σ = relu, rng = GLOBAL_RNG, biasinit::Bool = false)
    model = Vector{NeuroVertex}(undef, 0)
    acts = ActivationsStore(Dict{String,AbstractArray}())
    for i in 2:length(widths)
        i == length(widths) ? σ = identity : σ = σ #last layer is not activated
        push!(model, NeuroVertexDense(widths[i-1], widths[i], activeneurons[i-1], activeneurons[i], σ, rng, biasinit))
        acts.currentacts[i-1] = Matrix{Float32}(undef, 0, 0)
    end
    NeuroSearchSpace(model, acts, [], (-1,-1), cpu)
end

function NeuroSearchSpace(hiddenwidths::AbstractVector{Int}, hiddenactives::AbstractVector{Int}, input::Int, output::Int, biasinit::Bool = false, σ = relu, rng = GLOBAL_RNG)
    model = Vector{NeuroVertex}(undef, 0)
    auxs = Vector{AbstractVecOrMat}(undef, 0)
    acts = ActivationsStore(Dict{String,AbstractArray}())
    push!(model, NeuroVertexDense(input, hiddenwidths[1], input, hiddenactives[1], σ, rng))
    if length(hiddenwidths) > 1
        push!(auxs, zeros(Float32, hiddenwidths[2], input))
    else
        push!(auxs, zeros(Float32, output, input))
    end
    acts.currentacts[1] = Matrix{Float32}(undef, 0, 0)
    for i in 2:length(hiddenwidths)
        push!(model, NeuroVertexDense(hiddenwidths[i-1], hiddenwidths[i], hiddenactives[i-1], hiddenactives[i], σ, rng, biasinit))
        if i < length(hiddenwidths)
            push!(auxs, zeros(Float32, hiddenwidths[i+1], hiddenwidths[i-1]))
        end
        acts.currentacts[i-1] = Matrix{Float32}(undef, 0, 0)
    end
    push!(model, NeuroVertexDense(hiddenwidths[end], output, hiddenactives[end], output, identity, rng))
    if length(hiddenwidths) > 1
        push!(auxs, zeros(Float32, output, hiddenwidths[end-1]))
    end
    acts.currentacts[length(hiddenwidths)+1] = Matrix{Float32}(undef, 0, 0)
    NeuroSearchSpace(model, acts, auxs, (-1,-1), cpu)
end

function NeuroSearchSpace(hiddenwidths::AbstractVector{Int}, hiddenactives::AbstractVector{Int}, input::Tuple{Int,Int,Int}, output::Int, kernels::AbstractVector{Tuple{Int,Int}}, biasinit::Bool = false, σ = relu, rng = GLOBAL_RNG)
    mockdata = zeros(Float32, input...,1)
    conversion = (-1,-1)
    model = Vector{NeuroVertex}(undef, 0)
    auxs = Vector{AbstractArray}(undef, 0)
    acts = ActivationsStore(Dict{String,AbstractArray}())
    hiddenwidths = copy(hiddenwidths)
    hiddenactives = copy(hiddenactives)
    pushfirst!(hiddenwidths, input[end])
    pushfirst!(hiddenactives, input[end])
    push!(hiddenactives, output)
    push!(hiddenwidths, output)
    acts.currentacts[1] = Matrix{Float32}(undef, 0, 0)
    for i in 2:length(hiddenwidths)
        if i == length(hiddenwidths)
            sigma = identity
            acts.currentacts[i] = Matrix{Float32}(undef, 0, 0)
        else
            sigma = σ
        end
        if kernels[i-1] == (-1,-1)
            inputactives = hiddenactives[i-1]
            inputtotal = hiddenwidths[i-1]
            if i-2 > 0 && kernels[i-2] != (-1,-1)
                conversion = size(mockdata)[1:2]
                inputactives = hiddenactives[i-1]*prod(conversion)
                inputtotal = hiddenwidths[i-1]*prod(conversion)
            end
            nv = NeuroVertexDense(inputtotal, hiddenwidths[i], inputactives, hiddenactives[i], sigma, rng, biasinit)
            push!(model, nv)
            if i < length(hiddenwidths)
                push!(auxs, zeros(Float32, hiddenwidths[i+1], hiddenwidths[i-1]))
            end
        else 
            nv = NeuroVertexConv(kernels[i-1], hiddenwidths[i-1], hiddenwidths[i], hiddenactives[i-1], hiddenactives[i], sigma, rng, biasinit, maxpool=false)
            push!(model, nv)
            if kernels[i] == (-1,-1)
                push!(auxs, zeros(Float32, kernels[i-1][1], kernels[i-1][2], hiddenwidths[i-1], hiddenwidths[i+1]))
            else
                push!(auxs, zeros(Float32, kernels[i-1][1]+kernels[i][1]-1, kernels[i-1][2]+kernels[i][2]-1, hiddenwidths[i-1], hiddenwidths[i+1]))
            end
        end
        acts.currentacts[i-1] = Matrix{Float32}(undef, 0, 0)
        mockdata = model[end](mockdata)
    end
    NeuroSearchSpace(model, acts, auxs, conversion, cpu)
end

function NeuroSearchSpaceVGG11(activeratio::Float32, input::Tuple{Int,Int,Int}, output::Int, tries::Array{Int}, biasinit::Bool = false, batchnorm::Bool = false, σ = relu, rng = GLOBAL_RNG)
    mockdata = zeros(Float32, input...,1)
    conversion = (-1,-1)
    model = Vector{NeuroVertex}(undef, 0)
    auxs = Vector{AbstractArray}(undef, 0)
    acts = ActivationsStore(Dict{String,AbstractVecOrMat}())
    channels = [input[3], 64, 128, 256, 256, 512, 512, 512, 512]
    maxpool = [false, true, true, false, true, false, true, false, true]
    convactives = copy(channels)
    convactives[2:end] = floor.(Int, convactives[2:end].*activeratio)
    lastactive = convactives[end]
    acts.currentacts[1] = Matrix{Float32}(undef, 0, 0)
    channels[2:end] *= 2
    channels[2:end] += tries[1:end-2]
    for i in 2:length(channels)
        nv = NeuroVertexConv((3,3), channels[i-1], channels[i], convactives[i-1], convactives[i], σ, rng, biasinit, maxpool = maxpool[i], pad = (1,1))
        push!(model, nv)
        acts.currentacts[i-1] = Matrix{Float32}(undef, 0, 0)
        mockdata = model[end](mockdata)
        if i < length(channels)
            push!(auxs, zeros(Float32, 5, 5, channels[i-1], channels[i+1]))
        end
    end
    conversion = size(mockdata)[1:2]
    widths = [prod(conversion)*channels[end], 4096, 4096, output]
    denseactives = copy(widths)
    denseactives[1] = lastactive
    denseactives[2:end-1] = floor.(Int, denseactives[2:end-1].*activeratio)
    widths[2:end-1] *= 2
    widths[2:end-1] += tries[end-1:end]
    push!(auxs, zeros(Float32, 3, 3, channels[end-1], widths[2]))
    mockdata = flatten(mockdata)
    for i in 2:length(widths)
        if i == length(widths)
            sigma = identity
            acts.currentacts[i] = Matrix{Float32}(undef, 0, 0)
        else
            sigma = σ
            push!(auxs, zeros(Float32, widths[i+1], widths[i-1]))
        end
        push!(model, NeuroVertexDense(widths[i-1], widths[i], denseactives[i-1], denseactives[i], sigma, rng, biasinit))
        mockdata = model[end](mockdata)
    end
    NeuroSearchSpace(model, acts, auxs, conversion, cpu)
end

function NeuroVertexSequenceWRN28(maxinputchannels::Int, maxchannels::Int, initinputchannels::Int, initchannels::Int, stride::Int, skipscale::Float32, biasinit::Bool, batchnorm::Bool, σ, rng, avgpool::Bool)
    layers = Vector{NeuroVertex}(undef, 0)
    auxs = Vector{AbstractArray}(undef, 0)
    skips = Vector{Any}(undef, 0)
    acts = ActivationsStore(Dict{String,AbstractVecOrMat}())
    for i in 1:4
        if i == 1
            if stride > 1
                skip = NeuroVertexConv((1,1), maxinputchannels, maxchannels, initinputchannels, initchannels, identity, rng, biasinit, stride = stride, pad = (0,0), preact = true)
            else
                skip = Flux.identity
            end
            layer1 = NeuroVertexConv((3,3), maxinputchannels, maxchannels, initinputchannels, initchannels, σ, rng, biasinit, stride = stride, pad = (1,1), preact = true, batchnorm = batchnorm)
            push!(auxs, zeros(Float32, 5, 5, maxinputchannels, maxchannels))
        else
            skip = Flux.identity
            layer1 = NeuroVertexConv((3,3), maxchannels, maxchannels, initchannels, initchannels, σ, rng, biasinit, pad = (1,1), preact = true, batchnorm = batchnorm)
        end
        layer2 = NeuroVertexConv((3,3), maxchannels, maxchannels, initchannels, initchannels, σ, rng, biasinit, pad = (1,1), preact = true, batchnorm = batchnorm)
        append!(layers, [layer1, layer2])
        push!(skips, skip)
    end
    if avgpool
        pool = Chain(BatchNorm(maxchannels),
                     x -> σ.(x),
                     GlobalMeanPool(),
                     x -> flatten(x))
    else
        pool = Chain()
    end
    return NeuroVertexSequence(layers, skips, acts, pool, skipscale), auxs
end

function NeuroSearchSpaceWRN28(activeratio::Float32, widths::Vector, input::Tuple{Int,Int,Int}, output::Int, skipscale::Float32 = 1f0, biasinit::Bool = false, batchnorm::Bool = false, σ = relu, rng = GLOBAL_RNG)
    widths = [floor(Int, width) for width in widths]
    mockdata = zeros(Float32, input...,1)
    conversion = (-1,-1)
    model = Vector{NeuroVertex}(undef, 0)
    auxs = Vector{AbstractArray}(undef, 0)
    acts = ActivationsStore(Dict{String,AbstractVecOrMat}())
    push!(model, NeuroVertexConv((3,3), input[3], widths[1], input[3], floor(Int, 16*activeratio), identity, rng, biasinit, pad = (1,1)))
    acts.currentacts[1] = Matrix{Float32}(undef, 0, 0)
    mockdata = model[end](mockdata)
    push!(auxs, zeros(Float32, 5, 5, input[3], widths[2]))
    nvs1, auxs1 = NeuroVertexSequenceWRN28(widths[1], widths[2], floor(Int, 16*activeratio), floor(Int, 16*activeratio), 1, skipscale, biasinit, batchnorm, σ, rng, false)
    push!(model, nvs1)
    nvs2, auxs2 = NeuroVertexSequenceWRN28(widths[2], widths[3], floor(Int, 16*activeratio), floor(Int, 32*activeratio), 2, skipscale, biasinit, batchnorm, σ, rng, false)
    push!(model, nvs2)
    nvs3, auxs3 = NeuroVertexSequenceWRN28(widths[3], widths[4], floor(Int, 32*activeratio), floor(Int, 64*activeratio), 2, skipscale, biasinit, batchnorm, σ, rng, true)
    push!(model, nvs3)
    push!(model, NeuroVertexDense(widths[4], output, floor(Int, 64*activeratio), output, identity, rng, biasinit)) 
    auxs = vcat(auxs, auxs1, auxs2, auxs3)
    NeuroSearchSpace(model, acts, auxs, conversion, cpu)
end

Flux.trainable(m::NeuroSearchSpace) = Flux.trainable(m.model)
Flux.params(m::NeuroSearchSpace) = Flux.params(m.model)
auxparams(m::NeuroSearchSpace) = Flux.params(m.auxs)

Flux.@functor NeuroSearchSpace

function countparams(m::NeuroSearchSpace)
    count = 0
    for nv in m.model
        count += countparams(nv)
    end
    return count
end

function countparams(nv::NeuroVertexConv)
    count = prod(size(nv.layer.weight)[1:2])*sum(nv.maskW)+sum(nv.maskb)
    if length(nv.batchnorm.layers) > 0
        count += countparams(nv.batchnorm)
    end
    return count
end

function countparams(nv::NeuroVertexDense)
    return sum(nv.maskW)+sum(nv.maskb)
end

function countparams(m::NeuroVertexSequence)
    count = sum([countparams(l) for l in m.layers])
    for s in m.skips
        count += countparams(s)
    end
    return count
end

function togpu(m::NeuroSearchSpace)
    return NeuroSearchSpace([togpu(nv) for nv in m.model], m.acts, gpu(m.auxs), m.conversion, gpu)
end

function layerparams(m::NeuroSearchSpace)
    ps = Flux.Params()
    for nv in m.model
        for p in layerparams(nv)
            push!(ps, p)
        end
    end
    return ps
end

function maskparams(m::NeuroSearchSpace)
    ps = Flux.Params()
    for nv in m.model
        for p in maskparams(nv)
            push!(ps, p)
        end
    end
    return ps
end

function (m::NeuroSearchSpace)(x::AbstractArray; depth::Int = -1, saveacts::Bool=false, skips::Bool=true)
    for (index, nv) in enumerate(m.model)
        if !isa(nv, NeuroVertexSequence)
            x = nv(x)
        else
            x = nv(x; depth = depth, saveacts = saveacts, skips=skips)
        end
        if !isa(nv, Chain)
            if saveacts
                if !isa(nv, NeuroVertexSequence)
                    m.acts.currentacts[index] = x
                else
                    m.acts.currentacts[index] = nv.acts.currentacts[1]
                end
            else
                m.acts.currentacts[index] = []
            end
        end
        if index == depth
            return x
        end
    end
    return x
end

function (m::NeuroSearchSpace)(auxs::Vector{AbstractArray}, x::AbstractArray, saveacts::Bool=false)
    old_x = x
    previousnv = m.model[1]
    for (index, nv) in enumerate(m.model)
        if index == 1
            global old_x = x
            global previousnv = nv
            x = nv(x)
        else
            x_copy = x
            if isa(nv, NeuroVertexSequence)
                x = nv(auxs[index], x, old_x, previousnv, firstaux=auxs[1])
            elseif isa(previousnv, NeuroVertexSequence)
                x = nv(x)
            else
                x = nv(auxs[index-1], x, old_x, previousnv)
            end
            global old_x = x_copy
            global previousnv = nv
        end
        if saveacts
            m.acts.currentacts[index] = x
        else
            m.acts.currentacts[index] = []
        end
    end
    return x
end

function (m::NeuroSearchSpace)(i::Int, aux::AbstractArray, x::AbstractArray, saveacts::Bool=false)
    old_x = x
    previousnv = m.model[1]
    if any([isa(nv, NeuroVertexSequence) for nv in m.model])
        for (index, nv) in enumerate(m.model)
            if (index == i && i > 1) || (index == 2 && i == 1)
                if i == 1
                    x = nv([], x, old_x, previousnv, firstaux=aux)
                else
                    x = nv(aux, x, old_x, nv)
                end
            else
                global old_x = x
                x = nv(x)
                global previousnv = nv
            end
            if saveacts
                m.acts.currentacts[index] = x
            else
                m.acts.currentacts[index] = []
            end
        end
    else
        for (index, nv) in enumerate(m.model)
            if index == i+1
                x = nv(aux, x, old_x, previousnv)
            elseif index == i
                global old_x = x
                x = nv(x)
                global previousnv = nv
            else
                x = nv(x)
            end
            if saveacts
                m.acts.currentacts[index] = x
            else
                m.acts.currentacts[index] = []
            end
        end
    end
    return x
end

function unmaskneuron(m::NeuroSearchSpace, layer::Int, neuron::Union{Int,Vector{Int}}, init = glorot_uniform, rng = GLOBAL_RNG, rescale::Bool = false)
    unmaskneuron(m.model[layer], neuron, init, rng, rescale)
    if ndims(Wb(m.model[layer])[1]) != ndims(Wb(m.model[layer+1])[1])
        kernel = prod(m.conversion)
        if isa(neuron, Int)
            unmaskoutputs(m.model[layer+1], collect((neuron-1)*kernel+1:neuron*kernel), init, rng, rescale)
        else
            unmaskoutputs(m.model[layer+1], vcat([collect((n-1)*kernel+1:n*kernel) for n in neuron]...), init, rng, rescale)
        end
    else
        unmaskoutputs(m.model[layer+1], neuron, init, rng, rescale)
    end
end

function maskneuron(m::NeuroSearchSpace, layer::Int, neuron::Union{Int,Vector{Int}})
    maskneuron(m.model[layer], neuron)
    if ndims(Wb(m.model[layer])[1]) != ndims(Wb(m.model[layer+1])[1])
        kernel = prod(m.conversion)
        if isa(neuron, Int)
            maskoutputs(m.model[layer+1], collect((neuron-1)*kernel+1:neuron*kernel))
        else
            maskoutputs(m.model[layer+1], vcat([collect((n-1)*kernel+1:n*kernel) for n in neuron]...))
        end
    else
        maskoutputs(m.model[layer+1], neuron)
    end
end 

function rescale(nv::NeuroVertex, neuron::Int, currindices::AbstractArray, fanin=true)
    Wm = Wmasked(nv, fanin, fanin, !fanin, false, true)
    factor = mean(vecnorm(Wm[currindices, :],2)) / vecnorm(Wm[neuron, :])
    if fanin
        scaleweights(nv, neuron, getactiveinputindices(nv), factor)
    else
        scaleweights(nv, getactiveindices(nv), neuron, factor)
    end
end

function rescale(nv::NeuroVertex, neurons::Vector{Int}, currindices::AbstractArray, fanin=true)
    Wm = Wmasked(nv, fanin, fanin, !fanin, false, true)
    factors = mean(vecnorm(Wm[currindices, :],2)) ./ [vecnorm(Wm[neuron, :]) for neuron in neurons]
    for (neuron, factor) in zip(neurons, factors)
        if fanin
            scaleweights(nv, neuron, getactiveinputindices(nv), factor)
        else
            scaleweights(nv, getactiveindices(nv), neuron, factor)
        end
    end
end

gmrelu(x) = max(0, x)
@adjoint gmrelu(x) = gmrelu(x), y->(ifelse(y<0, zero(y), y),)

addrelu(x, y) = @. relu(x + y)

togpu(m) = m |> gpu

countparams(m::Union{Chain, Parallel}) = sum(countparams(l) for l in m.layers)
function countparams(m::Union{Conv, Dense})
    params = 0
    for paramfield in [m.weight, m.bias]
        if isa(paramfield, AbstractArray)
            params += prod(size(paramfield))
        end
    end
    return params
end
function countparams(m::BatchNorm)
    return 0
    params = 0
    for paramfield in [m.β, m.γ, m.μ, m.σ²]
        if isa(paramfield, AbstractArray)
            params += prod(size(paramfield))
        end
    end
    return params
end
countparams(m) = 0
