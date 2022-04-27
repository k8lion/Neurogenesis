using Neurogenesis
using Flux
using BSON
using Zygote
using Random
using Statistics
using LinearAlgebra
using CUDA
using CPUTime
include("utilities.jl")

function ng(args)
    if args["gpu"] && CUDA.functional()
        device = gpu
    else
        device = cpu
    end
    timeCPU = @CPUelapsed begin
        time = @myelapsed device begin 
            trigger = args["trigger"]
            init = args["init"]
            saveacts = trigger == SVDACTS
            orthog = args["orthog"]
            batchsize = args["batchsize"]
            lastfix = args["hidden"] - args["endfix"] + 1
            Random.seed!(args["seed"])
            otherrng = MersenneTwister(abs(rand(Int)))
            if args["vgg"]
                name = "vgg"
            elseif args["wrn"]
                name = "wrn"
            elseif args["conv"]
                name = "conv"
            else
                name = string("fc", args["hidden"])
            end
            name = string(args["seed"], "_", name)
            if length(args["datadir"]) == 0
                args["datadir"] = nothing
            end
            datadir = args["datadir"]
            folder = joinpath(string("test/plots/ng", args["dataset"], args["vgg"] ? "vgg" : "", args["wrn"] ? "wrn" : ""), args["expdir"], string(trigger, "_", init), args["name"])
            if isfile(joinpath(folder, "logs.bson")) & (args["expdir"] != "test")
                return
            end
            mkpath(folder)
            mkpath(joinpath(folder, "code"))
            if !isdir(joinpath(folder, "code/src"))
                cp("src", joinpath(folder, "code/src"))
                mkpath(joinpath(folder, "code/exp"))
                cp("exp/runneurogenesis.jl", joinpath(folder, "code/exp/runneurogenesis.jl"))
                cp("exp/utilities.jl", joinpath(folder, "code/exp/utilities.jl"))
            end
            open(joinpath(folder, "args.txt"), "w") do io
                for (k, v) in args
                    write(io, k, ": ", string(v), "\n")
                end
            end
            epochs = args["epochs"]
            init_hidden = args["initwidth"]
            buffer_multiplier = args["buffermult"]

            if args["dataset"] == MNIST
                trainDL, testDL = getmnistdata(batchsize, datadir, !args["conv"], args["gpu"], args["hptune"], otherrng) |> device
                features = 784
                classes = 10
            elseif args["dataset"] == CIFAR10
                trainDL, testDL = getcifar10data(batchsize, datadir, false, args["gpu"], args["hptune"], otherrng) |> device
                features = 32 * 32 * 3
                classes = 10
            elseif args["dataset"] == CIFAR100
                trainDL, testDL = getcifar100data(batchsize, datadir, false, args["gpu"], args["hptune"], otherrng) |> device
                features = 32 * 32 * 3
                classes = 100
            elseif args["dataset"] == SIM
                features = 64
                trainDL, testDL = getsimdata(batchsize, features, args["effdim"], features, 1f-1) |> device
                classes = 2
            end
            if args["maxwidth"] < 0
                args["maxwidth"] = features 
            end
            if args["trigger"] == BIGSTATIC
                args["staticwidth"] = args["maxwidth"]
            elseif args["staticwidth"] < 0
                args["staticwidth"] = features
            end
            max_hidden = args["maxwidth"]

            if !args["SGD"]
                if args["decay"]
                    opt = ADAMW(args["lr"], (0.9, 0.999), 1.0)
                else
                    opt = ADAM(args["lr"])
                end
            else
                if args["decay"]
                    opt = Flux.Optimiser(WeightDecay(0.1), Descent(args["lr"]))
                else
                    opt = Descent(args["lr"])
                end
            end
            if args["cosine"]
                opt = Flux.Optimiser(opt, CosineAnnealing(epochs))
            end

            function accuracy(m, x, y, validy = 1:classes)
                mean(Flux.onecold(m(x)[validy, :], validy) .== Flux.onecold(y[validy, :], validy))
            end

            function accuracy(m, dl::Flux.DataLoader, validy = 1:classes)
                accs = zeros(Float32, length(dl))
                sizes = zeros(Float32, length(dl))
                for (i, (x, y)) in enumerate(dl)
                    accs[i] = accuracy(m, x, y, validy)
                    sizes[i] = size(x)[2]
                end
                return sum(accs .* sizes ./ sum(sizes))
            end

            function accuracy(m, dl::CuIterator, validy = 1:classes)
                accs = zeros(Float32, length(dl))
                sizes = zeros(Float32, length(dl))
                i = 1
                for (x, y) in dl
                    accs[i] = accuracy(m, x, y, validy)
                    sizes[i] = size(x)[2]
                    i += 1
                end
                return sum(accs .* sizes ./ sum(sizes))
            end

            auxloss(m, i, aux, x, y) = Flux.logitcrossentropy(m(i, aux, x), y)

            input = features
            if args["conv"]
                input = size(first(trainDL)[1])[1:3]
            end

            if !args["vgg"] && !args["wrn"]
                init_hiddens = ones(Int, args["hidden"]) .* init_hidden
                init_hiddens_ = init_hiddens
                max_hiddens = ones(Int, args["hidden"]) .* max_hidden
                if args["conv"]
                    max_hiddens[1:2] .= 32
                    init_hiddens[1:2] .= 2
                    init_hiddens_[1:2] .= 2
                    defaultstatic = [6, 16, 120, 84]
                    init_hiddens_[1:args["fix"]] .= defaultstatic[1:args["fix"]]
                    init_hiddens[1:args["fix"]] .= defaultstatic[1:args["fix"]]
                    max_hiddens[1:args["fix"]] .= defaultstatic[1:args["fix"]]
                    init_hiddens_[lastfix:end] .= defaultstatic[lastfix:end]
                    init_hiddens[lastfix:end] .= defaultstatic[lastfix:end]
                    max_hiddens[lastfix:end] .= defaultstatic[lastfix:end]
                    tries = [args["convtries"], args["convtries"], args["densetries"], args["densetries"]]
                else
                    defaultstatic = ones(Int, length(init_hiddens)) .* args["staticwidth"]
                    init_hiddens_[1:args["fix"]] .= args["staticwidth"]
                    init_hiddens[1:args["fix"]] .= args["staticwidth"]
                    max_hiddens[1:args["fix"]] .= args["staticwidth"]
                    init_hiddens_[lastfix:end] .= args["staticwidth"]
                    init_hiddens[lastfix:end] .= args["staticwidth"]
                    max_hiddens[lastfix:end] .= args["staticwidth"]
                    tries = ones(Int, length(init_hiddens)) .* args["densetries"]
                end
            elseif args["vgg"]
                defaultstatic = [64, 128, 256, 256, 512, 512, 512, 512, 4096, 4096] 
                if args["trigger"] == STATIC
                    init_hiddens_ = defaultstatic
                    init_hiddens = init_hiddens_
                    tries = zeros(Int, length(init_hiddens_))
                    max_hiddens = init_hiddens_
                elseif args["trigger"] == BIGSTATIC
                    init_hiddens_ = defaultstatic .* 2
                    init_hiddens = init_hiddens_
                    tries = zeros(Int, length(init_hiddens_))
                    max_hiddens = init_hiddens_
                elseif args["trigger"] == SMALLSTATIC
                    init_hiddens_ = floor.(Int, defaultstatic ./ 4)
                    init_hiddens = init_hiddens_
                    tries = zeros(Int, length(init_hiddens_))
                    max_hiddens = init_hiddens_
                else
                    init_hiddens_ = floor.(Int, defaultstatic ./ 4)
                    init_hiddens = init_hiddens_
                    tries = ones(Int, length(init_hiddens_)) .* args["convtries"]
                    tries[end-1:end] .= args["densetries"]
                    max_hiddens = defaultstatic * 2
                end
                max_hidden = maximum([512*2+args["convtries"], 4096*2+args["densetries"]])
            elseif args["wrn"]
                if args["intergrowth"]
                    defaultstatic = vcat([16 for _ in 1:4],[32 for _ in 1:4],[64 for _ in 1:4]) 
                else
                    defaultstatic = [16, 16, 32, 64]
                end
                if args["trigger"] == STATIC
                    init_hiddens_ = defaultstatic .* args["initmultiplier"]
                    init_hiddens = init_hiddens_
                    tries = zeros(Int, length(init_hiddens_))
                    max_hiddens = init_hiddens_
                elseif args["trigger"] == BIGSTATIC
                    init_hiddens_ = defaultstatic .* args["maxwidth"]
                    init_hiddens = init_hiddens_
                    tries = zeros(Int, length(init_hiddens_))
                    max_hiddens = init_hiddens_
                elseif args["trigger"] == SMALLSTATIC
                    init_hiddens_ = floor.(Int, defaultstatic ./ 4)
                    init_hiddens = init_hiddens_
                    tries = zeros(Int, length(init_hiddens_))
                    max_hiddens = init_hiddens_
                else
                    init_hiddens_ = floor.(Int, defaultstatic ./ 4)
                    init_hiddens = init_hiddens_
                    tries = ones(Int, length(init_hiddens_)) .* args["convtries"]
                    max_hiddens = defaultstatic .* args["maxwidth"]
                end
                max_hidden = 64 .* args["maxwidth"] + args["convtries"]
            end
            @show init_hiddens_
            @show max_hiddens
            @show max_hidden

            if args["vgg"]
                m = NeuroSearchSpaceVGG11(args["initmultiplier"], input, classes, tries, false, false, gmrelu)
                [@show countactiveneurons(m.model[i]) for i in 1:length(max_hiddens)]
            elseif args["wrn"]
                m = NeuroSearchSpaceWRN28(args["initmultiplier"], max_hiddens.+tries, input, classes, args["skipscale"], false, true, gmrelu)

            elseif args["conv"]
                m = NeuroSearchSpace(max_hiddens .+ tries, init_hiddens_, input, classes, [(5, 5), (5, 5), (-1, -1), (-1, -1), (-1, -1)], args["bias"], gmrelu) |> f32
            else
                m = NeuroSearchSpace(max_hiddens .+ tries, init_hiddens_, input, classes, args["bias"], gmrelu) |> f32
            end
            if device == gpu
                m = togpu(m)
            end

            xs = Buffer(zeros(Float32, input..., 0)|> device, floor(Int, max_hidden * buffer_multiplier))

            xaxis = Array{Float32,1}(undef, 0)
            updates = [Array{Float32,1}(undef, 0) for _ in 1:args["hidden"]]

            allloss = Array{Float32,1}(undef, 0)
            allgradnorm = Array{Float32,1}(undef, 0)
            allscores = [Array{Float32,1}(undef, 0) for _ in 1:args["hidden"]]
            allthresholds = [Array{Float32,1}(undef, 0) for _ in 1:args["hidden"]]
            allcounts = Array{Int,1}(undef, 0)
            allactives = [Array{Int,1}(undef, 0) for _ in 1:args["hidden"]]
            allinittimes = [Array{Float32,1}(undef, 0) for _ in 1:args["hidden"]]
            alltrigtimes = Array{Float32,1}(undef, 0)
            allinittimesCPU = [Array{Float32,1}(undef, 0) for _ in 1:args["hidden"]]
            alltrigtimesCPU = Array{Float32,1}(undef, 0)

            testaccs = zeros(Float32, 1)

            svd_thresholds = [args["svdthreshold"] for _ in 1:args["hidden"]]
            if trigger == SVDACTS
                if device == cpu
                    for (x,_) in trainDL
                        update!(xs, x)
                        if size(xs.data)[end] == xs.size
                            break
                        end
                    end
                    m(xs.data, saveacts=saveacts, skips=false)
                else
                    for (x,_) in trainDL
                        m(x, saveacts=saveacts, skips=false)
                        x = nothing
                        break
                    end
                end
                if args["svdinit"]
                    initial_orthos = [orthogscore(m.acts.currentacts[i], orthog, getactiveindices(m.model[i]))/countactiveneurons(m.model[i])*args["svdthreshold"] for i in 1:args["hidden"]]
                end
            elseif trigger == SVDWEIGHTS && args["svdinit"]
                initial_orthos = [countsvd(Wmasked(m.model[i], true, true, false, true, true))/countactiveneurons(m.model[i])*args["svdthreshold"] for i in 1:args["hidden"]]
            end
            batch = 1
            if trigger in statics
                push!(allcounts, countparams(m))
                @show countparams(m)
            end
            for epoch in 1:epochs
                if device == gpu
                    shuffle!(trainDL.batches)
                end
                @show epoch
                for (x, y) in trainDL
                    if device == cpu
                        update!(xs, x)
                        xsdata = xs.data
                    else
                        xsdata = x
                    end
                    if length(xaxis) == 0
                        xaxis = [1 / length(trainDL)]
                    else
                        push!(xaxis, xaxis[end] + 1 / length(trainDL))
                    end
                    ps = params(m)

                    train_loss, back = Zygote.pullback(() -> Flux.logitcrossentropy(m(x, saveacts=saveacts), y), ps)

                    gs = back(one(train_loss))
                    Flux.Optimise.update!(opt, ps, gs)

                    push!(allloss, train_loss)
                    [push!(allactives[i], countactiveneurons(m.model[i])) for i in 1:length(max_hiddens)]
                    if !(trigger in statics)

                        push!(allcounts, countparams(m))
                        trigtimeCPU = @CPUelapsed begin
                            trigtime = @myelapsed device begin 
                                if trigger == GSVDC
                                    Ws = params([getweights(m.model[i]) for i in 1:length(m.model)])
                                    push!(allgradnorm, norm(gs[W] for W in Ws))
                                    gradnorms = [gradnorm(m, i, gs[Ws[i]], gs[Ws[i+1]]) for i in 1:length(m.model)-1]
                                elseif trigger == SVDACTS
                                    if device == cpu
                                        m(xs.data, saveacts=saveacts)
                                    else
                                        m(x, saveacts=saveacts, skips=false)
                                    end
                                end
                                if device == gpu
                                    ps = nothing
                                    gs = nothing
                                    Ws = nothing
                                    GC.gc()
                                    CUDA.reclaim()
                                end
                                if trigger == GSVDC
                                    grad, = gradient(auxs -> Flux.logitcrossentropy(m(auxs, x), y), m.auxs) #|> cpu
                                end
                            end
                        end
                        for i in 1:length(max_hiddens)
                            trigtimeCPU += @CPUelapsed begin
                                trigtime += @myelapsed device begin 
                                    trig = false
                                    toadd = 1
                                    if device == gpu
                                        GC.gc()
                                        CUDA.reclaim()
                                    end
                                    if trigger == SVDACTS
                                        actives = getactiveindices(m.model[i])
                                        scorea = orthogscore(m.acts.currentacts[i], orthog, actives)
                                        push!(allscores[i], scorea)
                                        if args["svdinit"]
                                            svd_thresholds[i] = floor(Int, initial_orthos[i] * length(actives))
                                            if (args["vgg"] || args["wrn"]) && scorea/length(actives)*args["svdthreshold"] > initial_orthos[i]
                                                initial_orthos[i] = scorea/length(actives)*args["svdthreshold"]
                                            end
                                        else
                                            svd_thresholds[i] = floor(Int, args["svdthreshold"] * length(actives))
                                        end
                                        push!(allthresholds[i], svd_thresholds[i]) 
                                        trig = (scorea > svd_thresholds[i])
                                        toadd = floor(Int, scorea - svd_thresholds[i])
                                    elseif trigger == SVDWEIGHTS
                                        scorew = countsvd(Wmasked(m.model[i], true, true, false, true, true))
                                        push!(allscores[i], scorew)
                                        if args["svdinit"]
                                            svd_thresholds[i] = floor(Int, initial_orthos[i] * length(getactiveindices(m.model[i])))
                                            if (args["vgg"] || args["wrn"]) && scorew/length(getactiveindices(m.model[i]))*args["svdthreshold"] > initial_orthos[i]
                                                initial_orthos[i] = scorew/length(getactiveindices(m.model[i]))*args["svdthreshold"]
                                            end
                                        else
                                            svd_thresholds[i] = floor(Int, args["svdthreshold"] * length(getactiveindices(m.model[i])))
                                        end
                                        push!(allthresholds[i], svd_thresholds[i])
                                        trig = (scorew > svd_thresholds[i])
                                        toadd = floor(Int, scorew - svd_thresholds[i])
                                    elseif trigger == GSVDC
                                        fanin = getactiveinputindices(m.model[i])
                                        fanout = getactiveindices(m.model[i+1])
                                        if ndims(grad[i]) == 2
                                            gradi = grad[i][fanout, fanin]
                                        else
                                            gradi = auxgradpatches(grad[i][:,:,fanin,fanout], device)
                                        end
                                        U, S, _ = svd(gradi)
                                        scorec = cpu(S)[1]
                                        push!(allscores[i], scorec)
                                        scoreg = sum(gradnorms[i])/2
                                        push!(allthresholds[i], scoreg)
                                        trig = (scorec >= scoreg)
                                        toadd = min(sum(S .>= scoreg), size(U)[2])
                                        if device == gpu
                                            S = nothing
                                            #grad[i] = similar(grad[i], 0, 0)
                                            if init !== NEST
                                                gradi = nothing
                                            elseif init !== GRADMAX
                                                U = nothing
                                            end
                                        end
                                    elseif trigger == RANDOMTRIG
                                        trig = (rand(otherrng) > (xaxis[end] / epochs)) & (countactiveneurons(m.model[i]) < (xaxis[end] / epochs * max_hiddens[i]))
                                        toadd = 1
                                    elseif trigger == LINEAR
                                        trig = (allactives[i][batch] < (defaultstatic[i] - init_hiddens_[i]) * xaxis[end] / (epochs * args["endpreset"]) + init_hiddens_[i]) & (allactives[i][batch] < defaultstatic[i])
                                        toadd = 1
                                    elseif trigger == FASTLINEAR
                                        trig = (allactives[i][batch] < defaultstatic[i])
                                        toadd = 1
                                    elseif trigger == BATCHED
                                        stage = floor(Int, floor(Int, ( 8*xaxis[end] / (epochs * args["endpreset"])))*(defaultstatic[i] - init_hiddens_[i])/8) + init_hiddens_[i]
                                        toadd = max(0, floor(Int, (defaultstatic[i] - init_hiddens_[i])/8))
                                        trig = (allactives[i][batch] < stage) && (allactives[i][batch] + toadd <= defaultstatic[i]) && toadd > 0
                                        if init == GRADMAX
                                            toadd = min(toadd, length(getactiveinputindices(m.model[i])), length(getactiveindices(m.model[i+1])), stage-allactives[i][batch])
                                        end
                                    end
                                    toadd = min(toadd, max_hiddens[i]-countactiveneurons(m.model[i]))
                                end
                            end

                            if (trig) & (countactiveneurons(m.model[i]) < max_hiddens[i]) & (i > args["fix"]) & (i < lastfix) & (toadd > 0)
                                inittimeCPU = @CPUelapsed begin
                                    inittime = @myelapsed device begin 
                                        copied = []
                                        if device == gpu
                                            GC.gc()
                                            CUDA.reclaim()
                                        end
                                        if init == RANDOMIN #random in, zero out
                                            newindices = getinactiveindices(m.model[i], toadd)
                                            unmaskneuron(m, i, newindices, Neurogenesis.glorot_uniform, otherrng, true)
                                            if i + 1 <= length(m.model)
                                                unmaskoutputs(m.model[i+1], newindices, Neurogenesis.zero_init, otherrng, false)
                                            end
                                        elseif init == RANDOMOUT #zero in, random out
                                            newindices = randomoutneuron(m, i, otherrng, toadd)
                                        elseif init == GRADMAX   #zero in, directly max grad-norm out
                                            if trig == GSVDC
                                                newindices = gradmaxneuron(m, i, U, device, otherrng, toadd)
                                            else
                                                newindices = gradmaxneuron(m, i, x, y, auxloss, device, otherrng, toadd)
                                            end
                                        elseif init == NEST #gradient based in and out
                                            if isa(m.model[i], Neurogenesis.NeuroVertexDense)
                                                if trig == GSVDC
                                                    newindices = nestneuron(m, i, cpu(gradi), 4.0f-1, otherrng, toadd)
                                                else
                                                    newindices = nestneuron(m, i, x, y, auxloss, 4.0f-1, device, otherrng, toadd)
                                                end
                                            else
                                                newindices = nestconvneuron(m, i, (m, x, y) -> Flux.logitcrossentropy(m(x), y), x, y, tries[i]+toadd, device, otherrng, toadd)
                                            end
                                        elseif init == FIREFLY #copy+noise in, half copy out
                                            newindices, copied = noisycopyneuron(m, i, (m, x, y) -> Flux.logitcrossentropy(m(x), y), x, y, tries[i]+toadd, args["eps"], 5.0f-1, device, otherrng, toadd)
                                        elseif init == ORTHOGACT #guess+check max orthog-act in, zero out
                                            if !isa(m.model[i], Neurogenesis.NeuroVertexDense)  #for speed
                                                newindices = initneuron(m, i, tries[i]+toadd, xsdata, orthog, otherrng, toadd)
                                            else
                                                newindices = optorthogact(m, i, xsdata, tries[i]+toadd, 0, orthog, device, otherrng, toadd)
                                            end
                                        elseif init == ORTHOGWEIGHTS #directly max orthog-weights in, zero out
                                            newindices = getinactiveindices(m.model[i], toadd)
                                            unmaskneuron(m.model[i], newindices, Neurogenesis.orthogonal_init, otherrng, true)
                                            if i + 1 <= length(m.model)
                                                unmaskoutputs(m.model[i+1], newindices, Neurogenesis.zero_init, otherrng, false)
                                            end
                                        elseif (init == SOLVEORTHOGACT) #directly max orthog pre-act in, zero out
                                            newindices = addorthogact(m, i, xsdata, tries[i]+toadd, orthog, otherrng, toadd) # length(getinactiveindices(m.model[1])
                                        elseif init == OPTORTHOGACT #use gradient descent to find weights that optimize orthogonality of act
                                            newindices = optorthogact(m, i, xsdata, floor(Int, tries[i]/10)+toadd, floor(Int, tries[i]/10), orthog, device, otherrng, toadd)
                                        end
                                    end
                                end
                                @show init, newindices
                                push!(updates[i], xaxis[end])
                                push!(allinittimes[i], inittime)
                                push!(allinittimesCPU[i], inittimeCPU)
                            end
                        end
                        push!(alltrigtimes, trigtime)
                        push!(alltrigtimesCPU, trigtimeCPU)
                    end
                    batch += 1
                end
                BSON.@save joinpath(folder, "temp_logs.bson") args allloss allgradnorm allactives allcounts allscores allthresholds testaccs xaxis epoch allinittimesCPU allinittimes alltrigtimesCPU alltrigtimes
            end
        end
    end
    @show time, timeCPU
    testaccs[end] = accuracy(m, testDL)
    BSON.@save joinpath(folder, "logs.bson") args allloss allgradnorm allactives allcounts allscores allthresholds testaccs xaxis time timeCPU allinittimesCPU allinittimes alltrigtimesCPU alltrigtimes
    @show allloss[end]
    @show testaccs[end]
end

args = parse_commandline()
showargs = sort(collect(args), by=x->x[1])
@show showargs
ng(args)
