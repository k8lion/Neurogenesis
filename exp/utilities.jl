using Neurogenesis
using Flux
using Plots
using Zygote
using Colors
using Random
using Random: GLOBAL_RNG, shuffle, shuffle!
using Base.Iterators: partition
using Statistics
using LinearAlgebra
using MLDatasets
using Dates
using ArgParse
gr()

SVDACTS = "svdacts"
SVDWEIGHTS = "svdweights"
GSVDC = "gsvdc"
LINEAR = "linear"
FASTLINEAR = "fastlinear"
BATCHED = "batched"
STATIC = "static"
BIGSTATIC = "bigstatic"
SMALLSTATIC = "smallstatic"
RANDOMTRIG = "randomtrig"
RANDOMIN = "randomin"
ORTHOGACT = "orthogact"
SOLVEORTHOGACT = "solveorthogact"
OPTORTHOGACT = "optorthogact"
ORTHOGWEIGHTS = "orthogweights"
RANDOMOUT = "randomout"
GRADMAX = "gradmax"
FIREFLY = "firefly"
NEST = "nest"
MNIST = "mnist"
SIM = "sim"
CIFAR10 = "cifar10"
CIFAR100 = "cifar100"
CARTPOLE = "cartpole"
ACROBOT = "acrobot"
COUNTSVD = "countsvd"
RELUSEP = "relusep"
ORTHOGONALGAP = "orthogonalgap"
inits = (RANDOMIN, ORTHOGACT, SOLVEORTHOGACT, OPTORTHOGACT, ORTHOGWEIGHTS, FIREFLY, RANDOMOUT, GRADMAX, NEST)
triggers = (RANDOMTRIG, SVDACTS, SVDWEIGHTS, GSVDC, LINEAR, FASTLINEAR, STATIC, BATCHED, BIGSTATIC, SMALLSTATIC)
datasets = (MNIST, SIM, CIFAR10, CIFAR100, CARTPOLE, ACROBOT)
orthogs = (COUNTSVD, RELUSEP, ORTHOGONALGAP)

function getsimdata(batchsize, features::Int=100, effdim::Int=50, outdim::Int=100, noiseportion::Float32=1f-1, samples::Int=5000, regression::Bool=false)
    rng = MersenneTwister(1337)
    datasetX = randn(rng, Float32, features, samples)
    if effdim < features
        datasetX[1+effdim:end,:] = randn(rng, Float32, features-effdim, effdim)*datasetX[1:effdim,:]
    end
    split = floor(Int, 0.8*samples)
    datasetresponse = (sum(datasetX[1:outdim,:], dims=1)*(1-noiseportion)) + randn(rng, Float32, 1, samples)*noiseportion
    if regression
        ytrain = datasetresponse[:, 1:split]
        ytest = datasetresponse[:, split+1:end]
    else
        datasetY = datasetresponse .> sum(datasetresponse)/length(datasetresponse)
        ytrain = datasetY[1, 1:split]
        ytest = datasetY[1, split+1:end]
        ytrain, ytest = Flux.onehotbatch(ytrain, 0:1), Flux.onehotbatch(ytest, 0:1)
    end
    xtrain = datasetX[:, 1:split]
    xtest = datasetX[:, split+1:end]
    trainDL = Flux.Data.DataLoader((xtrain,Matrix(ytrain)), batchsize=batchsize)
    testDL = Flux.Data.DataLoader((xtest,Matrix(ytest)), batchsize=batchsize)
    return trainDL, testDL
end


function getcifar10data(batchsize=64, dir=nothing, flatten=true, gpu=false, valid=false, rng=GLOBAL_RNG)
    if !valid
        xtrain, ytrain = MLDatasets.CIFAR10.traindata(Float32, dir = dir)
        xtest, ytest = MLDatasets.CIFAR10.testdata(Float32, dir = dir)
    else
        xtrain, ytrain = MLDatasets.CIFAR10.traindata(Float32, dir = dir)
        indices = shuffle!(MersenneTwister(1), collect(1:size(ytrain)[end]))
        split = floor(Int,length(indices)*0.9)
        xtest, ytest = xtrain[:,:,:,indices[split+1:end]], ytrain[indices[split+1:end]]
        xtrain, ytrain = xtrain[:,:,:,indices[1:split]], ytrain[indices[1:split]]
    end

    if flatten
        xtrain = Flux.flatten(xtrain)
        xtest = Flux.flatten(xtest)
    else
        xtrain = reshape(xtrain, (32,32,3,:))
        xtest = reshape(xtest, (32,32,3,:))
    end
    if gpu
        totaltrain, totaltest = length(ytrain), length(ytest)
        order = shuffle(rng, 1:totaltrain)
        xtrain = xtrain[repeat([:], ndims(xtrain) - 1)...,order]
        ytrain = ytrain[order]
        imgstrain = [xtrain[repeat([:], ndims(xtrain) - 1)...,i] for i in 1:totaltrain]
        labelstrain = Matrix(Flux.onehotbatch([ytrain[i] for i in 1:totaltrain],0:9))
        trainbatches = [(cat(imgstrain[i]..., dims = ndims(xtrain)), labelstrain[:,i]) for i in partition(1:totaltrain, batchsize)]
        imgstest = [xtest[repeat([:], ndims(xtest) - 1)...,i] for i in 1:totaltest]
        labelstest = Matrix(Flux.onehotbatch([ytest[i] for i in 1:totaltest],0:9))
        testbatches = [(cat(imgstest[i]..., dims = ndims(xtest)), labelstest[:,i]) for i in partition(1:totaltest, batchsize)]
        return CuIterator(trainbatches), CuIterator(testbatches)
    else
        ytrain, ytest = Flux.onehotbatch(ytrain, 0:9), Flux.onehotbatch(ytest, 0:9)
        train_loader = Flux.Data.DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true)
        test_loader = Flux.Data.DataLoader((xtest, ytest), batchsize=batchsize)
        return train_loader, test_loader
    end
end


function getcifar100data(batchsize=64, dir=nothing, flatten=true, gpu=false, valid=false, rng=GLOBAL_RNG)
    if !valid
        xtrain, ytrain = MLDatasets.CIFAR100.traindata(Float32, dir = dir)
        xtest, ytest = MLDatasets.CIFAR100.testdata(Float32, dir = dir)
    else
        xtrain, ytrain = MLDatasets.CIFAR100.traindata(Float32, dir = dir)
        indices = shuffle!(MersenneTwister(1), collect(1:size(ytrain)[end]))
        split = floor(Int,length(indices)*0.9)
        xtest, ytest = xtrain[:,:,:,indices[split+1:end]], ytrain[indices[split+1:end]]
        xtrain, ytrain = xtrain[:,:,:,indices[1:split]], ytrain[indices[1:split]]
    end

    if flatten
        xtrain = Flux.flatten(xtrain)
        xtest = Flux.flatten(xtest)
    else
        xtrain = reshape(xtrain, (32,32,3,:))
        xtest = reshape(xtest, (32,32,3,:))
    end
    if gpu
        totaltrain, totaltest = length(ytrain), length(ytest)
        order = shuffle(rng, 1:totaltrain)
        xtrain = xtrain[repeat([:], ndims(xtrain) - 1)...,order]
        ytrain = ytrain[order]
        imgstrain = [xtrain[repeat([:], ndims(xtrain) - 1)...,i] for i in 1:totaltrain]
        labelstrain = Matrix(Flux.onehotbatch([ytrain[i] for i in 1:totaltrain],0:9))
        trainbatches = [(cat(imgstrain[i]..., dims = ndims(xtrain)), labelstrain[:,i]) for i in partition(1:totaltrain, batchsize)]
        imgstest = [xtest[repeat([:], ndims(xtest) - 1)...,i] for i in 1:totaltest]
        labelstest = Matrix(Flux.onehotbatch([ytest[i] for i in 1:totaltest],0:9))
        testbatches = [(cat(imgstest[i]..., dims = ndims(xtest)), labelstest[:,i]) for i in partition(1:totaltest, batchsize)]
        return CuIterator(trainbatches), CuIterator(testbatches)
    else
        ytrain, ytest = Flux.onehotbatch(ytrain, 0:9), Flux.onehotbatch(ytest, 0:9)
        train_loader = Flux.Data.DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true)
        test_loader = Flux.Data.DataLoader((xtest, ytest), batchsize=batchsize)
        return train_loader, test_loader
    end
end

function getmnistdata(batchsize, dir=nothing, flatten=true, gpu=false, valid=false, rng=GLOBAL_RNG)
    if !valid
        xtrain, ytrain = MLDatasets.MNIST.traindata(Float32, dir = dir)
        xtest, ytest = MLDatasets.MNIST.testdata(Float32, dir = dir)
    else
        xtrain, ytrain = MLDatasets.MNIST.traindata(Float32, dir = dir)
        indices = shuffle!(MersenneTwister(1), collect(1:size(ytrain)[end]))
        split = floor(Int,length(indices)*0.9)
        xtest, ytest = xtrain[:,:,indices[split+1:end]], ytrain[indices[split+1:end]]
        xtrain, ytrain = xtrain[:,:,indices[1:split]], ytrain[indices[1:split]]
    end

    if flatten
        xtrain = Flux.flatten(xtrain)
        xtest = Flux.flatten(xtest)
    else
        xtrain = reshape(xtrain, (28,28,1,:))
        xtest = reshape(xtest, (28,28,1,:))
    end

    if gpu
        totaltrain, totaltest = size(ytrain)[end],  size(ytest)[end]
        order = shuffle(rng, 1:totaltrain)
        xtrain = xtrain[repeat([:], ndims(xtrain) - 1)...,order]
        ytrain = ytrain[order]
        imgstrain = [xtrain[repeat([:], ndims(xtrain) - 1)...,i] for i in 1:totaltrain]
        labelstrain = Matrix(Flux.onehotbatch([ytrain[i] for i in 1:totaltrain],0:9))
        trainbatches = [(cat(imgstrain[i]..., dims = ndims(xtrain)), labelstrain[:,i]) for i in partition(1:totaltrain, batchsize)]
        imgstest = [xtest[repeat([:], ndims(xtest) - 1)...,i] for i in 1:totaltest]
        labelstest = Matrix(Flux.onehotbatch([ytest[i] for i in 1:totaltest],0:9))
        testbatches = [(cat(imgstest[i]..., dims = ndims(xtest)), labelstest[:,i]) for i in partition(1:totaltest, batchsize)]
        return CuIterator(trainbatches), CuIterator(testbatches)
    else
        ytrain, ytest = Flux.onehotbatch(ytrain, 0:9), Flux.onehotbatch(ytest, 0:9)

        train_loader = Flux.Data.DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true)
        test_loader = Flux.Data.DataLoader((xtest, ytest), batchsize=batchsize)
        return train_loader, test_loader
    end

end

function getsplitmnistdata(batchsize, dir="", keepold=false, splitbylabel=false, rng=GLOBAL_RNG)
    if length(dir) > 0
        xtrain, ytrain = MLDatasets.MNIST.traindata(Float32, dir = dir)
        xtest, ytest = MLDatasets.MNIST.testdata(Float32, dir = dir)
    else
        xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
        xtest, ytest = MLDatasets.MNIST.testdata(Float32)
    end
	
    # Reshape Data in order to flatten each image into a linear array
    xtrain = Flux.flatten(xtrain) |> f32
    xtest = Flux.flatten(xtest) |> f32

    xtrains = Array{Any,1}(undef, 5)
    ytrains = Array{Any,1}(undef, 5)
    xtests = Array{Any,1}(undef, 5)
    ytests = Array{Any,1}(undef, 5)
    if !splitbylabel
        sortertrain = shuffle(rng, ytrain)
        sortertest = shuffle(rng, ytest) #or zeros so all test data is used always?
    else
        sortertrain = ytrain
        sortertest = ytest
    end
    for i in 0:4
        if keepold
            xtrains[i+1], ytrains[i+1] = xtrain[:, sortertrain.÷2  .<=i], ytrain[sortertrain.÷2 .<=i]
        else
            xtrains[i+1], ytrains[i+1] = xtrain[:, sortertrain.÷2 .==i], ytrain[sortertrain.÷2 .==i]
        end
        xtests[i+1], ytests[i+1] = xtest[:, sortertest.÷2 .==i], ytest[sortertest.÷2 .==i]
    end

    # One-hot-encode the labels
    #ytrains, ytests = Flux.onehotbatch.(ytrains, 0:9), Flux.onehotbatch.(ytests, 0:9)

    # Create DataLoaders (mini-batch iterators)
    train_loaders = Tuple(Flux.Data.DataLoader((x, Flux.onehotbatch(y, 0:9)), batchsize=batchsize, shuffle=true) for (x, y) in zip(xtrains, ytrains))
    test_loaders = Tuple(Flux.Data.DataLoader((x, Flux.onehotbatch(y, 0:9)), batchsize=batchsize) for (x, y) in zip(xtests, ytests))

    return train_loaders, test_loaders
end

Base.length(cuiter::CuIterator) = length(cuiter.batches)

function omniglot_dataloaders(batchsize::Int, class_portion::Float64 = 1.0)
    function get_omniglot(splitdir::String)
        count = 0
        for folder in (folder for folder in readdir(splitdir) if !startswith(folder, "."))
            for character in (folder for folder in readdir(join((splitdir,folder),"/")) if !startswith(folder, "."))
                for sample in readdir(join((splitdir,folder,character),"/"))
                    count += 1
                end
            end
        end
        x = Array{Bool, 3}(undef,105,105,count)
        y = Array{Int, 1}(undef,count)
        l = Array{String, 1}(undef,count)
        count = 1
        for folder in (folder for folder in readdir(splitdir) if !startswith(folder, "."))
            for character in (folder for folder in readdir(join((splitdir,folder),"/")) if !startswith(folder, "."))
                for sample in readdir(join((splitdir,folder,character),"/"))
                    label = split(sample,"_")[1]
                    image = load(join((splitdir,folder,character,sample),"/"))
                    image = image .!= image[1,1] 
                    if sum(image) > sum(1 .- image)
                        @show join(splitdir,folder,character,sample)
                    end
                    x[:,:,count] = image
                    y[count] = parse(Int, label)
                    l[count] = folder
                    count += 1
                end
            end
        end
        return x, y, l
    end

    basedir = "../omniglot"
    xtrain, ytrain, ltrain = get_omniglot(string(basedir, "/images_background"))
    xtest, ytest, ltest = get_omniglot(string(basedir, "/images_evaluation"))
    x = cat(xtrain, xtest, dims=3)
    y = cat(ytrain, ytest, dims=1)
    l = cat(ltrain, ltest, dims=1)
    trainlangs = sort(collect(Set{String}(ltrain)))
    testlangs = sort(collect(Set{String}(ltest)))
    alllangs = sort(cat(trainlangs, testlangs, dims=1))
    alllabels = collect(Set(y))
    portion = shuffle(alllabels)[1:floor(Int, class_portion*length(alllabels))]

    trainindices = Array{Int, 1}(undef,0)
    testindices = Array{Int, 1}(undef,0)
    for character in 1:(length(y)÷20)
        if character in portion
            range = (character-1) * 20 + 1:character * 20
            shuffled = Flux.shuffle(range)
            test = shuffled[1:4]
            train = shuffled[5:length(shuffled)]
            push!(trainindices, train...)
            push!(testindices, test...)
        end
    end

    downsample = Flux.MeanPool((3,3))
    xtrain = Flux.flatten(downsample(Flux.unsqueeze(float(x[:,:,trainindices]), 3)))
    xtest = Flux.flatten(downsample(Flux.unsqueeze(float(x[:,:,testindices]), 3)))
    ytrain = y[trainindices]
    ytest = y[testindices]
    #@test Set(ytrain) == Set(ytest)
    semilabels = collect(Set(ytrain))
    ltrain = l[trainindices]
    ltest = l[testindices]

    ytrain, ytest = Flux.onehotbatch(ytrain, semilabels), Flux.onehotbatch(ytest, semilabels) #, Flux.onehotbatch(y, alllabels)
                        
    train_loader = Flux.Data.DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true)
    test_loader = Flux.Data.DataLoader((xtest, ytest), batchsize=batchsize)
    return train_loader, test_loader, alllangs, semilabels
end

macro myelapsed(device, expr)
    if device == "gpu" && CUDA.functional()
        esc(quote
            CUDA.@elapsed $expr
        end)
    else
        esc(quote
            @elapsed $expr
        end)
    end
end

mutable struct Buffer
    data::AbstractArray
    size::Int
end

function update!(ab::Buffer, newdata::AbstractArray)
    if size(ab.data)[end] == 0
        ab.data = copy(newdata)
        return
    end
    to_keep = ab.size-size(newdata)[end]
    to_keep = min(to_keep, size(ab.data)[end])
    ab.data = cat(newdata, ab.data[repeat([:], ndims(newdata) - 1)..., 1:to_keep], dims = ndims(newdata))
end

binarize(x) = ifelse(x<=0, zero(x), one(x))

getsamples(DL::Flux.Data.DataLoader, numsamples::Int, rng = GLOBAL_RNG) = cat(shuffle!(rng, collect(x for (x,y) in DL))[1:ceil(Int, numsamples/DL.batchsize)]..., dims=2)[:,1:numsamples]

getsample(mat, num, rng) = mat[repeat([:], ndims(mat) - 1)..., shuffle(rng, 1:size(mat)[end])[1:num]]

function getsample(mat, y, num, rng)
    selection = shuffle(rng, 1:size(mat)[end])[1:num]
    return mat[repeat([:], ndims(mat) - 1)..., selection], y[:, selection]
end

mutable struct CosineAnnealing
    tmax::Int64
    t::Int64
end
  
CosineAnnealing(tmax::Int64 = 1) = CosineAnnealing(tmax, 0)
  
function Flux.Optimise.apply!(o::CosineAnnealing, x, Δ)
    tmax = o.tmax
    t = o.t
    Δ .*= (1 + cos(t/tmax*pi))/2
    return Δ
end

function parse_commandline()
    s = ArgParseSettings()

    dir = ""

    @add_arg_table! s begin
        "--name"
        help = "experiment name"
        arg_type = String
        default = string(now())
        "--expdir"
        help = "experiment directory"
        arg_type = String
        default = "test"
        "--init"
        help = "type of initialization"
        arg_type = String
        default = "randomin"
        "--trigger"
        help = "type of trigger"
        arg_type = String
        default = "randomtrig"
        "--orthog"
        help = "type of orthogonality measure"
        arg_type = String
        default = "countsvd"
        "--gpu"
        help = "use GPU"
        action = :store_true
        "--lr"
        help = "learning rate, 3f-4 for MLP and 3f-2 for VGG"
        arg_type = Float32
        default = -1f0
        "--SGD"
        help = "use SGD instead of ADAM"
        action = :store_true
        "--decay"
        help = "use weight decay"
        action = :store_true
        "--cosine"
        help = "use cosine annealing"
        action = :store_true
        "--hidden"
        help = "number of hidden layers"
        arg_type = Int
        default = 1
        "--fix"
        help = "fix first n layers"
        arg_type = Int
        default = 0
        "--endfix"
        help = "fix last n layers"
        arg_type = Int
        default = 0
        "--bias"
        help = "allow biases to be init to nonzero"
        action = :store_true
        "--conv"
        help = "use convolutional architecture"
        action = :store_true
        "--vgg"
        help = "use vgg architecture"
        action = :store_true
        "--staticwidth"
        help = "width of static layers (and max for static schedules), -1 is sdaptive to config"
        arg_type = Int
        default = -1
        "--datadir"
        help = "where to find/store MNIST"
        arg_type = String
        default = dir
        "--dataset"
        help = "which dataset to use: mnist or sim"
        arg_type = String
        default = "mnist"
        "--seed"
        help = "random seed"
        arg_type = Int
        default = 32
        "--batchsize"
        help = "batchsize"
        arg_type = Int
        default = 512
        "--maxwidth"
        help = "maximum hidden layer width, -1 is # of features"
        arg_type = Int
        default = -1
        "--initwidth"
        help = "initial hidden layer width"
        arg_type = Int
        default = 64
        "--initmultiplier"
        help = "init network to this fraction of static width. -1 to default to initwidth"
        arg_type = Float32
        default = -1f0
        "--epochs"
        help = "number of epochs"
        arg_type = Int
        default = 20
        "--buffermult"
        help = "buffer multiplier * maxlayer size"
        arg_type = Float32
        default = 2f0
        "--svdthreshold"
        help = "relative threshold of SVD between 0 and 1. <0 yields default value of .9 or .99"
        arg_type = Float32
        default = -1.0f0
        "--svdinit"
        help = "use initial orthogonality as raw threshold"
        action = :store_true
        "--nosvdinit"
        help = "use 1 as raw threshold"
        action = :store_true
        "--hptune"
        help = "tune hyperparameters, use validation instead of test set"
        action = :store_true
        "--eps"
        help = "epsilon for firefly"
        arg_type = Float32
        default = 1f-4
        "--densetries"
        help = "number of tries whenever many dense neurons are instantiated to select from"
        arg_type = Int
        default = 1000
        "--convtries"
        help = "number of tries whenever many dense neurons are instantiated to select from"
        arg_type = Int
        default = 100
        "--effdim"
        help = "number of dimensions to use for simulated data"
        arg_type = Int
        default = 50
        "--endpreset"
        help = "relative end of preset schedule"
        arg_type = Float32
        default = 0.75f0
        "--steps"
        help = "number of steps in RL experiments"
        arg_type = Int
        default = 10000
        "--episodes"
        help = "number of episodes in RL experiments"
        arg_type = Int
        default = 100
    end

    args = parse_args(s)
    if args["trigger"] == "datasep"
        args["trigger"] = "relusep"
    end
    if !(args["init"] in inits)
        @warn "Init not recognized."
        args["init"] = RANDOMIN
    end
    if !(args["trigger"] in triggers)
        @warn "Trigger not recognized."
        args["trigger"] = BATCHED
    end
    if !(args["dataset"] in datasets)
        @warn "Dataset not recognized."
        args["dataset"] = MNIST
    end
    if !(args["orthog"] in orthogs)
        @warn "Orthogonality measure not recognized."
        args["orthog"] = COUNTSVD
    end
    if args["conv"]
        args["hidden"] = 4
        args["initwidth"] = 32
    end
    if (args["svdthreshold"] < 0)
        if (args["trigger"] == SVDACTS)
            args["svdthreshold"] = 0.97
        else
            args["svdthreshold"] = 0.99
        end
    end
    if args["dataset"] == SIM
        args["initwidth"] = 4
        args["staticwidth"] = 64
        args["maxwidth"] = 512
        args["batchsize"] = 128
        if args["epochs"] == -1
            args["epochs"] = (20+4*args["effdim"])*args["hidden"]
        end
    end

    if args["trigger"] == STATIC
        args["fix"] = args["hidden"]
    elseif args["trigger"] == BIGSTATIC
        args["fix"] = args["hidden"]
        args["staticwidth"] = args["maxwidth"]
        args["initmultiplier"] = 2f0    
    elseif args["trigger"] == SMALLSTATIC
        args["fix"] = args["hidden"]
        args["staticwidth"] = args["initwidth"]
        args["initmultiplier"] = 0.25f0
    end

    if (args["staticwidth"] > args["maxwidth"]) & (args["maxwidth"] > 0) & (args["staticwidth"] > 0)
        args["staticwidth"] = args["maxwidth"]
    end

    args["ADAM"] = !args["SGD"]

    if args["vgg"]
        args["conv"] = true
        args["hidden"] = 10
        args["endpreset"] = 5f-1
        if args["dataset"] == MNIST
            args["dataset"] = CIFAR10
        end
        if args["buffermult"] == 2
            args["buffermult"] = 0.25f0
        end
        if args["initmultiplier"] == -1f0
            if args["trigger"] == BIGSTATIC
                args["initmultiplier"] = 2f0
            elseif args["trigger"] == STATIC
                args["initmultiplier"] = 1f0
            else
                args["initmultiplier"] = 0.25f0
            end
        end
        args["svdthreshold"] = 0.99
        args["cosine"] = true
    end
    if args["orthog"] == ORTHOGONALGAP
        args["svdinit"] = true
    end
    if args["lr"] < 0
        args["lr"] = 3f-4
    end
    if !args["svdinit"] && !args["nosvdinit"]
        args["svdinit"] = true
    end

    return args
end
