export relusep, svdweights, countsvd, relusepscore, orthogonalgapscore, orthogonalgap,
    countsvdactscore, countsvdactsum, orthogonalgapsum, meannorm, vecnorm, 
    gradnorm, orthogscore
using LinearAlgebra: norm, I, logabsdet, svd, tril
using LowRankApprox: psvdvals
using CUDA

vecnorm(v::AbstractVector,dims=-1) = sqrt.(sum(abs2,v))
vecnorm(v::AbstractArray{<:Real,3},dims=-1) = sqrt.(sum(abs2,vec(v)))
vecnorm(A::AbstractMatrix,dims=1) = sqrt.(sum(abs2,A,dims=dims))
vecnorm(A::AbstractArray{<:Real,4},dims::Int=1) = sqrt.(sum(abs2,flatten(A),dims=dims))
vecnorm(A::AbstractArray{<:Real,4},dims::Union{AbstractVector,Tuple{Int,Int,Int}}) = sqrt.(sum(abs2,A,dims=dims))

function meannorm(W::AbstractMatrix, fanin=true)
    if fanin
        return mean(vecnorm(W, 2))
    else
        return mean(vecnorm(W, 1))
    end
end

function meannorm(nv::NeuroVertex, fanin=true)
    #@show size(Wmasked(nv, fanin, fanin, !fanin)), size(vecnorm(Wmasked(nv, fanin, fanin, !fanin), 2))
    return mean(vecnorm(Wmasked(nv, fanin, fanin, !fanin), 2))
end

function gradnorm(m::NeuroSearchSpace, i::Int, gradin, gradout)
    if isa(m.model[i], NeuroVertexDense)
        fanin = vecnorm(gradin[getactiveindices(m.model[i]), getactiveinputindices(m.model[i])],2)
    else
        fanin = vecnorm(gradin[:, :, getactiveinputindices(m.model[i]), getactiveindices(m.model[i])],(1,2,3))
    end
    if isa(m.model[i+1], NeuroVertexDense)
        fanout = vecnorm(gradout[getactiveindices(m.model[i+1]), getactiveinputindices(m.model[i+1])],1)
    else
        fanout = vecnorm(gradout[:, :, getactiveinputindices(m.model[i+1]),getactiveindices(m.model[i+1])],(1,2,4))
    end
    return vec(fanin) + vec(fanout)
end


binarize(x) = ifelse(x<=0, zero(x), one(x))

function relusep(m::NeuroSearchSpace, x)
    K = 0.0
    for nv in m.model
        if (isactive(nv)) & (nv.layer.σ == relu)
            x = nv(x)
            x_bin = binarize.(x[nv.activeneurons,:])'
            K0 = x_bin * x_bin'
            K1 = (1 .- x_bin) * (1 .- x_bin')
            K += sum(K0) + sum(K1)
        end
    end
    return K
end

function relusep(acts::AbstractMatrix)
    x_bin = binarize.(acts)'
    K0 = x_bin * x_bin'
    K1 = (1 .- x_bin) * (1 .- x_bin')
    K = K0 + K1 
    return sum(tril(K ./ K[1,1], -1))/(size(K)[1]^2) #logabsdet(K)[1]
end

function svdweights(nv::NeuroVertex)
    mat = Wmasked(nv, true, true)
    return psvdvals(mat)
end

function svdweights(m::NeuroSearchSpace)
    return (svdweights(layer) for layer in m.model)
end

function countsvd(mat::AbstractMatrix, partial::Bool = false)
    if sum(abs2, mat) == 0
        return 0
    end
    if isa(mat, Matrix)
        svdvals = psvdvals(mat./sqrt(size(mat)[2]))
    else
        _,svdvals,_ = svd(mat./sqrt(size(mat)[2]))
    end
    countsvd = count(>(0.01), svdvals)
    if partial 
        countsvd += sum(abs, svdvals) 
    end
    return countsvd
end

function orthogonalgap(H::AbstractMatrix)
    if isa(H, Matrix)
        return norm(transpose(H)*H/norm(H)^2-I(size(H)[2])/size(H)[2])
    end
    return norm(transpose(H)*H/norm(H)^2-cu(I(size(H)[2]))/size(H)[2])
end

#purposefully across neurons instead of samples...?
function relusepscore(newacts::AbstractArray, curracts::AbstractArray)
    K = zeros(Float32, size(curracts)[2], size(curracts)[2])
    c_bin = binarize.(curracts)'
    K0 = c_bin * c_bin'
    K1 = (1 .- c_bin) * (1 .- c_bin)'
    K += K0 + K1
    bestK = K + I
    worstK = K + ones(size(K))
    best_lad = logabsdet(bestK)[1]
    worst_lad = logabsdet(worstK)[1]
    if (best_lad > -Inf) & (!isnan(best_lad))
        if (worst_lad == -Inf)
            newwl = true
            worst_lad = Inf
        else
            newwl = false
        end
        scores = ones(Float32, size(newacts)[1])
        for newneuron in 1:size(newacts)[1]
            K = zeros(Float32, size(curracts)[2], size(curracts)[2])
            nc_bin = binarize.(cat(curracts, newacts[newneuron:newneuron,:], dims=1))'
            K0 = nc_bin * nc_bin'
            K1 = (1 .- nc_bin) * (1 .- nc_bin)'
            K += K0 + K1
            ldK = logabsdet(K)[1]
            if newwl
                scores[newneuron] = best_lad-ldK
                if (ldK > -Inf) & (ldK < worst_lad)
                    worst_lad = ldK
                end
            else
                scores[newneuron] = (best_lad-ldK)/(best_lad-worst_lad)
            end
        end
        if minimum(scores) < Inf
            if newwl
                scores ./= (best_lad-worst_lad)
            end
            return scores
        end
    end

    K = zeros(Float32, size(curracts)[1], size(newacts)[1])
    na_bin = binarize.(newacts)'
    ca_bin = binarize.(curracts)
    K0 = ca_bin * na_bin
    K1 = (1 .- ca_bin) * (1 .- na_bin)
    K += K0 + K1
    return sum(abs2, K', dims = 2)./sum(abs2, (ones(size(ca_bin)) * ones(size(na_bin)))', dims = 2)
end

function countsvdactscore(newacts::AbstractMatrix, curracts::AbstractMatrix, partial::Bool=false)
    scores = ones(Float32, size(newacts)[1])
    for newneuron in 1:size(newacts)[1]
        mat = cat(newacts[newneuron:newneuron, :], curracts, dims=1)
        if sum(abs2, mat) > 0
            scores[newneuron] = -countsvd(mat, partial)
        end
    end
    return scores
end

function countsvdactsum(newacts::AbstractMatrix, curracts::AbstractMatrix)
    score = 0f0
    for newneuron in 1:size(newacts)[1]
        mat = cat(newacts[newneuron:newneuron, :], curracts, dims=1)
        _,svdvals,_ = svd(mat./sqrt(size(mat)[2]))
        score -= sum(svdvals)
    end
    return score
end

function countsvdactscore(newacts::AbstractArray, curracts::AbstractArray, partial::Bool=false)
    scores = ones(Float32, size(newacts)[3])
    curracts = flatten(curracts)
    for newneuron in 1:size(newacts)[3]
        mat = cat(flatten(newacts[:,:,newneuron,:]), curracts, dims=1)
        if sum(abs2, mat) > 0
            scores[newneuron] = -countsvd(mat, partial)
        end
    end
    return scores
end

function countsvdactsum(newacts::AbstractArray, curracts::AbstractArray)
    score = 0f0
    curracts = flatten(curracts)
    for newneuron in 1:size(newacts)[3]
        mat = cat(flatten(newacts[:,:,newneuron, :]), curracts, dims=1)
        _,svdvals,_ = svd(mat./sqrt(size(mat)[2]))
        score -= sum(svdvals)
    end
    return score
end

function orthogonalgapscore(newacts::AbstractMatrix, curracts::AbstractMatrix, partial::Bool=false)
    scores = ones(Float32, size(newacts)[1])
    for newneuron in 1:size(newacts)[1]
        H = cat(newacts[newneuron:newneuron, :], curracts, dims=1)
        if sum(abs2, H) > 0
            if isa(H, Matrix)
                scores[newneuron] = norm(transpose(H)*H/norm(H)^2-I(size(H)[2])/size(H)[2])
            else
                scores[newneuron] = norm(transpose(H)*H/norm(H)^2-cu(I(size(H)[2]))/size(H)[2])
            end
        end
    end
    return scores
end


function orthogonalgapsum(newacts::AbstractMatrix, curracts::AbstractMatrix)
    score = 0f0
    for newneuron in 1:size(newacts)[1]
        H = cat(newacts[newneuron:newneuron, :], curracts, dims=1)
        if sum(abs2, H) > 0
            if isa(H, Matrix)
                score += norm(transpose(H)*H/norm(H)^2-I(size(H)[2])/size(H)[2])
            else
                score += norm(transpose(H)*H/norm(H)^2-cu(I(size(H)[2]))/size(H)[2])
            end
        end
    end
    return score
end

function orthogonalgapscore(newacts::AbstractArray, curracts::AbstractArray, partial::Bool=false)
    scores = ones(Float32, size(newacts)[3])
    curracts = flatten(curracts)
    for newneuron in 1:size(newacts)[3]
        H = cat(flatten(newacts[:,:,newneuron,:]), curracts, dims=1)
        if sum(abs2, H) > 0
            if isa(H, Matrix)
                scores[newneuron] = norm(transpose(H)*H/norm(H)^2-I(size(H)[2])/size(H)[2])
            else
                scores[newneuron] = norm(transpose(H)*H/norm(H)^2-cu(I(size(H)[2]))/size(H)[2])
            end
        end
    end
    return scores
end

function orthogonalgapsum(newacts::AbstractArray, curracts::AbstractArray)
    score = 0f0
    curracts = flatten(curracts)
    for newneuron in 1:size(newacts)[3]
        H = cat(flatten(newacts[:,:,newneuron,:]), curracts, dims=1)
        if sum(abs2, H) > 0
            if isa(H, Matrix)
                score += norm(transpose(H)*H/norm(H)^2-I(size(H)[2])/size(H)[2])
            else
                score += norm(transpose(H)*H/norm(H)^2-cu(I(size(H)[2]))/size(H)[2])
            end
        end
    end
    return score
end

function orthogscore(acts::AbstractArray, orthog::String, actives::AbstractVector)
    if ndims(acts) == 2
        activeacts = acts[actives, :]
        if orthog == "countsvd"
            scorea = countsvd(activeacts)
        elseif orthog == "orthogonalgap"
            scorea = ceil((1 - orthogonalgap(activeacts)) * length(actives))
        elseif orthog == "relusep"
            scorea = ceil((1 - relusep(activeacts)) * length(actives))
        end
    else
        activeacts = acts[:,:,actives,:]
        if orthog == "countsvd"
            @show size(activeacts)
            activeacts = flatten(permutedims(activeacts, [1,2,4,3]))
            scorea = countsvd(activeacts)
        elseif orthog == "orthogonalgap"
            @show size(activeacts)
            activeacts = flatten(activeacts)
            scorea = ceil((1 - orthogonalgap(activeacts)) * length(actives))
        elseif orthog == "relusep"
            activeacts = flatten(activeacts)
            scorea = ceil((1 - relusep(activeacts)) * length(actives))
        end
    end
    return scorea
end