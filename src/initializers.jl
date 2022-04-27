export initneuron, optorthogact, addorthogact, noisycopyneuron, nestconvneuron, 
    nestneuron, gradmaxneuron, randomoutneuron, bnorthoginit, orthogweightinit,
    glorot_normal, glorot_uniform, orthogonal_init, zero_init, auxgradpatches

using Flux: gpu, cpu
using Statistics: quantile
using StatsBase: sample
using LinearAlgebra: svd, norm, pinv, Diagonal, tr
using CUDA
using Random: MersenneTwister
using ChainRules: AbstractZero
using ChainRulesCore: add!!
using ChainRules
import ChainRules: svd_rev

function ChainRules.svd_rev(USV::CUDA.CUSOLVER.CuSVD{T}, Ū::AbstractZero, s̄::CuArray{T,1}, V̄::AbstractZero) where T
    U = USV.U
    Vt = USV.Vt
    S̄ = s̄ isa AbstractZero ? s̄ : Diagonal(s̄)
    Ā = U * S̄ * Vt
    return Ā 
end

function glorot_normal(nv::NeuroVertexDense, neuron::Int, outs::Int, row::Bool = true, rng=GLOBAL_RNG)
    return randn(rng, Float32, outs) .* sqrt(2.0f0 / (sum(nv.activeneurons) + sum(nv.activeinputneurons)))
end

function glorot_normal(nv::NeuroVertexConv, neuron::Int, outs::Int, row::Bool = true, rng=GLOBAL_RNG)
    return randn(rng, Float32, size(nv.layer.weight)[1:2]..., outs) .* sqrt(2.0f0 / (prod(size(nv.layer.weight)[1:2])*(sum(nv.activeneurons) + sum(nv.activeinputneurons))))
end

function glorot_normal(fullins::Int, fullouts::Int, activeins::Int, activeouts::Int, rng=GLOBAL_RNG)
    return (randn(rng, Float32, fullouts, fullins)) .* sqrt(2.0f0 / (sum(activeins) + sum(activeouts)))
end

function glorot_uniform(nv::NeuroVertexDense, neuron::Int, outs::Int, row::Bool = true, rng=GLOBAL_RNG)
    return (rand(rng, Float32, outs) .- 0.5f0 ) .* sqrt(24.0f0 / (sum(nv.activeneurons) + sum(nv.activeinputneurons)))
end

function glorot_uniform(nv::NeuroVertexConv, neuron::Int, outs::Int, row::Bool = true, rng=GLOBAL_RNG)
    return (rand(rng, Float32, size(nv.layer.weight)[1:2]..., outs) .- 0.5f0 ) .* sqrt(24.0f0 / (prod(size(nv.layer.weight)[1:2])*(sum(nv.activeneurons) + sum(nv.activeinputneurons))))
end

function glorot_uniform(fullins::Int, fullouts::Int, activeins::Int, activeouts::Int, rng=GLOBAL_RNG)
    return (rand(rng, Float32, fullouts, fullins) .- 0.5f0 ) .* sqrt(24.0f0 / (sum(activeins) + sum(activeouts)))
end

function orthogonal_init(nv::NeuroVertex, neuron::Int, outs::Int, row::Bool = true, rng=GLOBAL_RNG)
    mat = Wmasked(nv, true, row, !row, false, true)
    weights = vec(glorot_uniform(nv, neuron, outs, row, rng))
    if row & (size(mat)[1] < size(mat)[2])
        mat = mat[getactiveindices(nv),:]
        orig_len = norm(weights,2)
        _, _, matV = svd(mat) #rows of matV are orthonormal basis of row space of mat
        mat = matV' ./ vecnorm(matV', 2)
        proj = dropdims(sum(mat .* sum(mat .* transpose(weights), dims=2), dims=1), dims=1)
        weights = (weights-proj)*orig_len/norm(weights-proj,2)
    elseif !row & (size(mat)[2] < size(mat)[1])
        #mat = mat[:, cat(getactiveinputindices(nv), size(mat)[2], dims=1)]
        mat = mat[:, getactiveinputindices(nv)]
        mat = transpose(mat)
        orig_len = norm(weights,2)
        _, _, matV = svd(mat) #rows of matV are orthonormal basis of row space of mat
        mat = matV' ./ vecnorm(matV', 2)
        proj = dropdims(sum(mat .* sum(mat .* transpose(weights), dims=2), dims=1), dims=1)
        weights = (weights-proj)*orig_len/norm(weights-proj,2)
    end
    return weights
end

function zero_init(nv::NeuroVertexDense, neuron::Int, outs::Int, row::Bool, rng=GLOBAL_RNG)
    return zeros(Float32, outs)
end

function zero_init(nv::NeuroVertexConv, neuron::Int, outs::Int, row::Bool, rng=GLOBAL_RNG)
    return zeros(Float32, size(nv.layer.weight)[1:2]..., outs)
end

function zero_init(fullins::Int, fullouts::Int, activeins::Int, activeouts::Int, rng=GLOBAL_RNG)
    return zeros(Float32, fullouts, fullins)
end


function initneuron(m::NeuroSearchSpace, layer::Int, tries::Int, x::AbstractArray, orthog::String="countsvd", rng = GLOBAL_RNG, toadd::Int = 1)
    if layer > 1
        x = m(x, depth=layer-1)
    end
    currindices = getactiveindices(m.model[layer])
    newindices = getinactiveindices(m.model[layer], tries)
    for neuron in newindices
        unmaskneuron(m.model[layer], neuron, glorot_uniform, rng, true)
        if layer+1 <= length(m.model)
            if ndims(Wb(m.model[layer])[1]) != ndims(Wb(m.model[layer+1])[1])
                kernel = prod(m.conversion)
                for n in (neuron-1)*kernel+1:neuron*kernel
                    unmaskoutputs(m.model[layer+1], n, zero_init, rng, false) 
                end
            else
                unmaskoutputs(m.model[layer+1], neuron, zero_init, rng, false) 
            end
        end
    end
    acts = m.model[layer](x)
    curracts = flatten(acts)[currindices, :]
    newacts = flatten(acts)[newindices, :]
    if orthog == "countsvd"
        scores = countsvdactscore(newacts, curracts, true)
    elseif orthog == "relusep"
        scores = relusepscore(newacts, curracts)
    elseif orthog == "orthogonalgap"
        scores = orthogonalgapscore(newacts, curracts)
    end
    tokeeps = newindices[sortperm(scores)[1:toadd]]
    @show minimum(scores), maximum(scores)
    for neuron in newindices
        if !(neuron in tokeeps)
            maskneuron(m, layer, neuron)
        end
    end
    return tokeeps
end


function optorthogact(m::NeuroSearchSpace, layer::Int, x::AbstractArray, tries::Int = 1, steps::Int=1000, orthog::String="countsvd", device = "cpu", rng = GLOBAL_RNG, toadd::Int = 1)
    if orthog == "countsvd"
        sumscore = countsvdactsum
        score = countsvdactscore
    elseif orthog == "orthogonalgap"
        sumscore = orthogonalgapsum
        score = orthogonalgapscore
    end
    
    if layer > 1
        x = m(x, depth=layer-1)
    end
    currindices = getactiveindices(m.model[layer])
    if !isa(m.model[layer], NeuroVertexDense)
        trimmed_x = x[:,:,getactiveinputindices(m.model[layer]),:] 
        if isa(m.model[layer], NeuroVertexConv)
            curracts = flatten(m.model[layer](x)[:,:,currindices, :])
        else
            curracts = flatten(m.model[layer].layers[1](x)[:,:,currindices, :])
        end
    else
        trimmed_x = flatten(x)[getactiveinputindices(m.model[layer]),:] 
        curracts = m.model[layer](x)[currindices, :]
    end
    currweights = Wmasked(m.model[layer], true, true, false, false, true)
    currmeannorm = meannorm(m.model[layer])
    neuronwidth = size(currweights)[2]
    opt = ADAM()
    if !isa(m.model[layer], NeuroVertexDense)
        faninweights = glorot_uniform(tries, neuronwidth, neuronwidth, size(curracts)[2]+1, rng) |> f32 |> device
        if isa(m.model[layer], NeuroVertexSequence)
            mlayer = m.model[layer].layers[1]
        else
            mlayer = m.model[layer]
        end
        σ, mp = mlayer.layer.σ, mlayer.maxpool
        W = reshape(faninweights, size(mlayer.layer.weight)[1:2]..., :, tries)
        W .*= currmeannorm./vecnorm(W, (1,2,3)) 
        cdims = DenseConvDims(trimmed_x, W; stride = mlayer.layer.stride, padding = mlayer.layer.pad, dilation = mlayer.layer.dilation, groups = mlayer.layer.groups)
        if steps > 0
            pW = params(W)
            for _ in 1:steps
                gW = gradient(pW) do
                    out = mp(σ.(conv(trimmed_x, W, cdims)))
                    return sumscore(out, curracts)
                end
                Flux.Optimise.update!(opt, pW, gW)
                W .*= currmeannorm./vecnorm(W, (1,2,3)) 
            end
        end
        newact = mp(σ.(conv(trimmed_x, W, cdims)))
        faninweights = transpose(flatten(W))
        """
        if device == gpu
            CUDA.unsafe_free!(W)
            if steps > 0
                CUDA.unsafe_free!(gW)
            end
        end
        """
        scores = score(newact, curracts, true)
    else
        faninweights = glorot_uniform(neuronwidth, tries, neuronwidth, size(curracts)[2]+1, rng) |> f32 |> device
        σ = m.model[layer].layer.σ
        faninweights .*= currmeannorm./vecnorm(faninweights, 2) 
        if steps > 0
            pW = params(faninweights)
            for _ in 1:steps
                gW = gradient(pW) do
                    out = σ.(faninweights*trimmed_x)
                    scoresum = sumscore(out, curracts)
                    return scoresum
                end
                Flux.Optimise.update!(opt, pW, gW)
                faninweights .*= currmeannorm./vecnorm(faninweights, 2) 
            end
        end
        newact = σ.(faninweights*trimmed_x)
        """
        if (device == gpu) && (steps > 0)
            CUDA.unsafe_free!(gW)
        end
        """
        scores = score(newact, curracts, true)
    end
    tokeeps = sortperm(scores,rev=true)[1:toadd]
    newindices = getinactiveindices(m.model[layer], toadd)
    unmaskneuron(m.model[layer], faninweights[tokeeps,:], newindices, true, rng)
    if ndims(Wb(m.model[layer])[1]) != ndims(Wb(m.model[layer+1])[1])
        kernel = prod(m.conversion)
        unmaskoutputs(m.model[layer+1], vcat([collect((neuron-1)*kernel+1:neuron*kernel) for neuron in newindices]...), zero_init, rng, false)
    else
        unmaskoutputs(m.model[layer+1], newindices, zero_init, rng, false)
    end
    return newindices
end


function addorthogact(m::NeuroSearchSpace, layer::Int, x::AbstractArray, tries::Int = 1, orthog::String="countsvd", rng = GLOBAL_RNG, toadd::Int = 1)
    if isa(m.model[layer], NeuroVertexConv)
        return initneuron(m, layer, tries, x, orthog, rng)
    end
    origx = copy(x)
    old_active = getactiveindices(m.model[layer])
    if layer > 1
        x = m(x, depth=layer-1)
        x_active = flatten(cpu(x)[repeat([:], ndims(x) - 2)...,getactiveindices(m.model[layer-1]),:])
    else
        x_active = flatten(cpu(x))
    end
    acts = flatten(cpu(m.model[layer](x,x->x))[repeat([:], ndims(m.model[layer](x)) - 2)...,getactiveindices(m.model[layer]),:])
    _, _, actsVfull = svd(acts, full=true)
    orthospace = actsVfull[:,size(acts)[1]:end]
    if length(getinactiveindices(m.model[layer])) < tries
        tries = length(getinactiveindices(m.model[layer]))
    end
    weights = pinv(x_active')*orthospace*randn(rng, size(orthospace)[2], tries)
    newindices = getinactiveindices(m.model[layer], tries)
    scores = zeros(length(newindices))
    for (i,new_index) in enumerate(newindices) 
        unmaskneuron(m.model[layer], weights[:,i], new_index, true, rng) 
        if layer+1 <= length(m.model)
            if ndims(Wb(m.model[layer])[1]) != ndims(Wb(m.model[layer+1])[1])
                kernel = prod(m.conversion)
                for n in (neuron-1)*kernel+1:neuron*kernel
                    unmaskoutputs(m.model[layer+1], n, zero_init, rng, false) 
                end
            else
                unmaskoutputs(m.model[layer+1], new_index, zero_init, rng, false) 
            end
        end
    end
    acts = m(origx, depth=layer)
    curracts = acts[old_active, :]
    newacts = acts[newindices, :]
    if orthog == "countsvd"
        scores = countsvdactscore(newacts, curracts, true)
    elseif orthog == "orthogonalgap"
        scores = orthogonalgapscore(newacts, curracts)
    end
    toremove = newindices[sortperm(scores)[toadd+1:end]]
    @show minimum(filter(!isnan,scores)), maximum(filter(!isnan,scores))
    for neuron in toremove
        maskneuron(m,layer,neuron)
    end
    return newindices[sortperm(scores)[1:toadd]]
end

function noisycopyneuron(m::NeuroSearchSpace, i::Int, lossf, x, y, tries::Int = 1, ϵ::Float32 = 1f-4, copyratio::Float32 = 5f-1, device = cpu, rng = GLOBAL_RNG, toadd::Int = 1)
    actives = getactiveindices(m.model[i])
    inputactives = getactiveinputindices(m.model[i])
    if isa(m.model[i], NeuroVertexDense)
        oldW = copy(m.model[i].layer.weight[:, inputactives] |> cpu)
    else
        oldW = copy(Wmasked(m.model[i], true, true, false, false, true))
    end
    if !isa(m.model[i+1], NeuroVertexSequence)
        nextlayer = m.model[i+1]
    else
        nextlayer = m.model[i+1].layers[1]
    end
    newindices = getinactiveindices(m.model[i], tries)
    nextactives = getactiveindices(nextlayer)
    noises = rand(rng, size(oldW[newindices,:])...).*2f0 .-1f0
    tocopys = sort(sample(rng,actives,min(floor(Int,length(newindices)*copyratio), length(actives)),replace=false))
    for (j, newindex) in enumerate(newindices)
        if j <= length(tocopys)
            noises[j,:]*=ϵ*vecnorm(oldW[tocopys[j],:])
            faninweights = oldW[tocopys[j],:].+noises[j,:]
            unmaskneuron(m.model[i], faninweights, newindex, false, rng)
        else
            faninweights = noises[j,:]
            unmaskneuron(m.model[i], faninweights, newindex, true, rng)
        end
        if i < length(m.model)
            if j <= length(tocopys)
                scaleweights(nextlayer, nextactives, tocopys[j], 5f-1)
                fanoutweights = Wmasked(nextlayer, true, false, true, true, false)[:, tocopys[j]]
            else
                numweights = length(nextactives)
                if isa(nextlayer, NeuroVertexConv)
                    numweights *= prod(size(Wb(nextlayer)[1])[1:2])
                end
                fanoutweights = randn(Float32, numweights)*ϵ*meannorm(nextlayer, false)
            end
            if ndims(Wb(m.model[i])[1]) != ndims(Wb(nextlayer)[1])
                kernel = prod(m.conversion)
                for n in (newindex-1)*kernel+1:newindex*kernel
                    unmaskoutputs(nextlayer, fanoutweights, n) #[1+(ind-1)*kernel:ind*kernel]
                end
            else
                unmaskoutputs(nextlayer, fanoutweights, newindex)
            end
        end
    end
    addweights(m.model[i], tocopys, inputactives, -noises[1:length(tocopys),:])
    ps = params(Wb(m.model[i])[1])
    if device == gpu
        GC.gc()
        CUDA.reclaim()
    end
    grad, = gradient(() -> lossf(m, x, y), ps)
    if ndims(grad) == 4
        gradnorms = vecnorm(cpu(grad)[:,:,inputactives,newindices], (1,2,3))
    else
        gradnorms = vecnorm(cpu(grad)[newindices,inputactives], 2)
    end
    sortedgn = sortperm(vec(gradnorms), rev=true)
    tokeep = sortedgn[1:toadd]
    toremove = sortedgn[toadd+1:end]
    if length(tocopys) > 0
        assignweights(m.model[i], tocopys, inputactives, copy(transpose(oldW[tocopys,:])|> device))
    end
    addweights(m.model[i], tocopys[tokeep[tokeep .<= length(tocopys)]], inputactives, -noises[tokeep[tokeep .<= length(tocopys)],:])
    scaleweights(m.model[i], newindices[toremove], inputactives, 0f0)
    maskneuron(m.model[i], newindices[toremove])
    if i < length(m.model)
        if length(tocopys) > 0
            scaleweights(nextlayer, nextactives, tocopys[toremove[toremove .<= length(tocopys)]], 2f0)   
        end 
        if ndims(Wb(m.model[i])[1]) != ndims(Wb(nextlayer)[1])
            kernel = prod(m.conversion)
            maskoutputs(nextlayer, vcat([collect((neuron-1)*kernel+1:neuron*kernel) for neuron in newindices[toremove]]...))
        else
            maskoutputs(nextlayer, newindices[toremove])
        end
    end
    if length(tocopys) > 0
        return newindices[tokeep], tocopys[tokeep[tokeep .<= length(tocopys)]]
    end
    return newindices[tokeep], []
end

function nestconvneuron(m::NeuroSearchSpace, i::Int, lossf, x, y, tries::Int = 1, device = cpu, rng = GLOBAL_RNG, toadd::Int = 1)
    oldW = copy(Wmasked(m.model[i], true, true, false, false, true))
    newindices = getinactiveindices(m.model[i],tries)
    faninnoises = transpose(glorot_uniform(size(oldW[newindices,:])..., size(oldW[newindices,:])..., rng))
    nextactives = getactiveindices(m.model[i+1])
    numweights = length(nextactives)
    if isa(m.model[i+1], NeuroVertexConv)
        numweights *= prod(size(Wb(m.model[i+1])[1])[1:2])
    end
    fanoutnoises = transpose(glorot_uniform(length(newindices), numweights, length(newindices), numweights, rng))
    losses = ones(Float32, length(newindices)).*Inf
    
    for (j, newindex) in enumerate(newindices)
        faninweights = vec(faninnoises[j,:])
        fanoutweights = vec(fanoutnoises[j,:])
        unmaskneuron(m.model[i], faninweights, newindex, true, rng)
        if ndims(Wb(m.model[i])[1]) != ndims(Wb(m.model[i+1])[1])
            kernel = prod(m.conversion)
            for n in (newindex-1)*kernel+1:newindex*kernel
                unmaskoutputs(m.model[i+1], fanoutweights, n, true) #[1+(ind-1)*kernel:ind*kernel]
            end
        else
            unmaskoutputs(m.model[i+1], fanoutweights, newindex, true)
        end
        losses[j] = lossf(m, x, y)
        maskneuron(m.model[i], newindex)
        if ndims(Wb(m.model[i])[1]) != ndims(Wb(m.model[i+1])[1])
            kernel = prod(m.conversion)
            for n in (newindex-1)*kernel+1:newindex*kernel
                maskoutputs(m.model[i+1], n)
            end
        else
            maskoutputs(m.model[i+1], newindex)
        end
    end
    tokeep = sortperm(losses)[1:toadd]
    unmaskneuron(m.model[i], faninnoises[tokeep,:], newindices[tokeep], true, rng)
    if ndims(Wb(m.model[i])[1]) != ndims(Wb(m.model[i+1])[1])
        kernel = prod(m.conversion)
        unmaskoutputs(m.model[i+1], fanoutnoises[tokeep,:], vcat([collect((neuron-1)*kernel+1:neuron*kernel) for neuron in newindices[tokeep]]...), true) 
    else
        unmaskoutputs(m.model[i+1], fanoutnoises[tokeep,:], newindices[tokeep], true)
    end
    return newindices[tokeep]
end

function auxgradpatches(grad::AbstractArray, device, outkernel=(3,3), stride=(1,1))
    #grad: ksize X ksize X m0 X m2 ; ksize=k1+k2-1; outkernel=(k2,k2)
    filter = zeros(Float32, outkernel..., prod(outkernel), size(grad,4))
    for i in 1:outkernel[1]
        for j in 1:outkernel[2]
            filter[i,j,(j-1)*outkernel[1]+(i-1)+1,:] .= 1
        end
    end
    filter = reshape(filter, outkernel..., 1, prod(outkernel)*size(grad,4))
    filterlayer = Conv(identity, filter, Flux.Zeros(), stride, (0,0), (1,1), size(grad,4)) |> device
    output = filterlayer(permutedims(grad, (1,2,4,3)))
    #output: k1 X k1 X (m2*k2*k2) X m0
    expanded = permutedims(reshape(output, size(output)[1:2]..., outkernel...,:,size(output,4))[:,:,end:-1:1,end:-1:1,:,:],(1,2,5,4,3,6)) #expanded
    output = reshape(expanded, size(output)) #collapsed
    output = permutedims(output, (2,1,4,3)) #rearrange before flattening
    return copy(flatten(output)')
end


function gradmaxneuron(m::NeuroSearchSpace, i::Int, x::AbstractArray, y::AbstractArray, lossf, device = cpu, rng = GLOBAL_RNG, toadd::Int = 1)
    grad, = gradient(auxw -> lossf(m,i,auxw,x,y), m.auxs[i]) 
    fanin = getactiveinputindices(m.model[i])
    fanout = getactiveindices(m.model[i+1])
    if !isa(m.model[i], NeuroVertexDense)
        if isa(m.model[i+1], NeuroVertexDense)
            gradfilt = auxgradpatches(grad[:,:,fanin,fanout], device, (1,1))
        else
            gradfilt = auxgradpatches(grad[:,:,fanin,fanout], device)
        end
    else
        gradfilt = grad[fanout, fanin]
    end
    U, _, _ = svd(gradfilt)
    allweights = copy(U[:,1:toadd]') #TODO: make sure this works for conv case
    newindices = getinactiveindices(m.model[i], toadd)
    unmaskneuron(m.model[i], newindices, zero_init, rng)
    unmaskoutputs(m.model[i+1], allweights |> device, newindices, true)
    return newindices
end


function gradmaxneuron(m::NeuroSearchSpace, i::Int, Ugradfilt::AbstractMatrix, device = cpu, rng = GLOBAL_RNG, toadd::Int = 1)
    allweights = copy(transpose(Ugradfilt[:,1:toadd]))
    newindices = getinactiveindices(m.model[i], toadd)
    unmaskneuron(m.model[i], newindices, zero_init, rng)
    unmaskoutputs(m.model[i+1], allweights |> device, newindices, true)
    return newindices
end


function nestneuron(m::NeuroSearchSpace, i::Int, x, y, lossf, beta::Float32 = 1f-1, device = cpu, rng = GLOBAL_RNG, toadd::Int = 1)
    aux = device(zeros(Float32, size(m.model[i+1].layer.weight)[1], size(m.model[i].layer.weight)[2])) 
    grad, = gradient(auxw -> lossf(m,i,auxw,x,y), aux) 
    fanin = getactiveinputindices(m.model[i])
    fanout = getactiveindices(m.model[i+1])
    grad = cpu(grad)[fanout, fanin]
    threshold = quantile(vec(abs.(grad)), beta)
    wout = zeros(Float32, toadd, length(fanout))
    win = zeros(Float32, toadd, length(fanin))
    newindices = getinactiveindices(m.model[i],toadd)
    for n in 1:toadd
        for i in 1:length(fanin)
            for o in 1:length(fanout)
                if abs(grad[o,i]) > threshold
                    dw = abs(grad[o,i])^(1/2)*rand(rng, (-1,1))
                    win[n,i] += dw
                    wout[n,o] += dw^2/(abs(dw))
                end
            end
        end
    end
    unmaskneuron(m.model[i], win, newindices, true, rng)
    unmaskoutputs(m.model[i+1], wout, newindices, true)
    return newindices
end


function nestneuron(m::NeuroSearchSpace, i::Int, grad::AbstractMatrix, beta::Float32 = 1f-1, rng = GLOBAL_RNG, toadd::Int = 1)
    threshold = quantile(vec(abs.(grad)), beta)
    fanin = getactiveinputindices(m.model[i])
    fanout = getactiveindices(m.model[i+1])
    wout = zeros(Float32, toadd, length(fanout))
    win = zeros(Float32, toadd, length(fanin))
    newindices = getinactiveindices(m.model[i],toadd)
    for n in 1:toadd
        for i in 1:length(fanin)
            for o in 1:length(fanout)
                if abs(grad[o,i]) > threshold
                    dw = abs(grad[o,i])^(1/2)*rand(rng, (-1,1))
                    win[n,i] += dw
                    wout[n,o] += dw^2/(abs(dw))
                end
            end
        end
    end
    unmaskneuron(m.model[i], win, newindices, true, rng)
    unmaskoutputs(m.model[i+1], wout, newindices, true, rng)
    return newindices
end

function randomoutneuron(m::NeuroSearchSpace, i::Int, rng = GLOBAL_RNG, toadd::Int = 1)
    newindices = getinactiveindices(m.model[i],toadd)
    unmaskneuron(m.model[i], newindices, zero_init, rng)
    if ndims(Wb(m.model[i])[1]) != ndims(Wb(m.model[i+1])[1])
        kernel = prod(m.conversion)
        unmaskoutputs(m.model[i+1], vcat([collect((neuron-1)*kernel+1:neuron*kernel) for neuron in newindices]...), glorot_uniform, rng, true)
    else
        unmaskoutputs(m.model[i+1], newindices, glorot_uniform, rng, true)
    end
    return newindices
end
