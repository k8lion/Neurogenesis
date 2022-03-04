using BSON
using Plots
using Plots.PlotMeasures
using Statistics: mean, std, cov
using StatsPlots
using ColorSchemes
using DataFrames
using Tar
using CodecZlib
gr()
default(linewidth=2, legend=:bottomright, size=(600,400), yguidefonthalign=:right, left_margin=20px)

nanmean(x) = mean(filter(!isnan,x))
nanmean(x,y) = mapslices(nanmean,x,dims=y)
nanstd(x) = std(filter(!isnan,x))
nanstd(x,y) = mapslices(nanstd,x,dims=y)
movingaverage(g, n) = [i < n ? mean(g[begin:i]) : mean(g[i-n+1:i]) for i in 1:length(g)]
childdirs(path) = [child for child in joinpath.(path, readdir(path)) if isdir(child)]

linestyles = [:solid, :dash, :dot, :dashdot, :dashdotdot, :solid, :dash, :dot, :dashdot, :dashdotdot]
markershapes = [:circle, :diamond, :star5, :hexagon, :rect, :triangle, :xcross, :cross]
fillstyles = [nothing, :/, :\, :+, :x]

palette10 = palette(:tab10)[vcat(collect(1:7),[9,10,8])]

statics = Dict("Small Static"=>1, "Medium Static"=>2, "Big Static"=>3)

initorder = ["orthogact", "optorthogact", "solveorthogact", "randomin", "orthogweights", "gradmax", "firefly", "nest", "randomout", "static"]
initnames = Dict("orthogact"=>"NORTH-Select", 
                 "optorthogact"=>"NORTH-Opt", 
                 "solveorthogact"=>"NORTH-Pre", 
                 "orthogweights"=>"NORTH-Weight", 
                 "randomin"=>"NORTH-Rand", 
                 "gradmax"=>"GradMax", 
                 "firefly"=>"Firefly", 
                 "nest"=>"NeST", 
                 "randomout"=>"RandomOut",
                 "static"=>"Static")
initorder = [initnames[init] for init in initorder]
inits = Dict(init=>i for (i, init) in enumerate([init for init in initorder]))

trigorder = ["svdacts", "svdweights", "gsvdc", "linear", "batched", "smallstatic", "static", "bigstatic"]
trignames = Dict("svdacts"=>"Dynamic", 
                 "svdweights"=>"Dynamic", 
                 "gsvdc"=>"Dynamic",
                 "linear"=>"Linear", 
                 "batched"=>"Batched",
                 "smallstatic"=>"Small Static",
                 "static"=>"Medium Static", 
                 "bigstatic"=>"Big Static")
trigorder = [trignames[trig] for trig in trigorder]
trigs = Dict(trig=>i for (i, trig) in enumerate([trig for trig in trigorder]))

exppaths = ["outputs/ngmnist/mnistdynosched",
            "outputs/ngmnist/mniststaticsched",
            "outputs/ngmnist/mnistlr",
            "outputs/ngmnist/mnistcsthresh",
            "outputs/ngmnist/mnistogthresh",
            "outputs/ngmnist/mnistbatchsize",
            "outputs/ngcifar10/vggtime",
            "outputs/ngcifar10/vggdynosched",
            "outputs/ngsim/simdyno",
            "outputs/ngsim/simdyno2",
            ]
alldata = Dict()
cputime = 0 
gputime = 0

for exp in exppaths
    if !isdir(exp)
        expdir = dirname(exp)
        if isfile(string(expdir, ".tar.gz"))
            tarball = string(expdir, ".tar.gz")
            tar_gz = open(tarball)
            tar = GzipDecompressorStream(tar_gz)
            dir = Tar.extract(tar, expdir)
            close(tar)
        else
            println(string("error ", expdir))
        end
    end
    @show exp
    name = split(exp, "/")[end]
    alldata[name] = Dict()
    addcounts = false
    for triginit in childdirs(exp)
        trig, init = split(split(triginit, "/")[end], "_")
        trig = trignames[split(trig, "/")[end]]
        init = initnames[split(init, "/")[end]]
        if occursin("tatic", trig)
            init="Static"
        end
        for trial in sort(childdirs(triginit), rev=true)
            trialname = split(trial, "/")[end]
            trialnum, config = split(trialname, "_")
            if !haskey(alldata, name)
                alldata[name] = Dict()
            end
            if (isfile(joinpath(trial,"logs.bson")))
                logs = "logs.bson"
                BSON.@load joinpath(trial,logs) args testaccs xaxis allactives allloss
                allcounts_ = nothing
                try
                    BSON.@load joinpath(trial,logs) allcounts
                    if length(allcounts) > 0
                        addcounts = true
                        allcounts_ = allcounts
                    end
                catch end
                option = ""
                if occursin("lr", name)
                    if trig == "batched"
                        continue
                    end
                    option = args["lr"]
                elseif occursin("orthog", name)
                    option = args["orthog"]
                elseif occursin("thresh", name)
                    option = args["svdthreshold"]
                elseif occursin("batchsize", name)
                    option = args["batchsize"]
                elseif occursin("simdyno", name)
                    option = args["effdim"]
                    if config == "ngsim2"
                        name="simdyno2"
                        if !haskey(alldata, name)
                            alldata[name] = Dict()
                        end
                    else
                        name="simdyno"
                    end
                end
                seed = args["seed"]
                if length(option) > 0
                    if init == "optorthogact"
                        continue
                    end
                    if !haskey(alldata[name], trig)
                        alldata[name][trig] = Dict()
                    end
                    if !haskey(alldata[name][trig], init)
                        alldata[name][trig][init] = Dict()
                    end
                    if !haskey(alldata[name][trig][init], option)
                        alldata[name][trig][init][option] = Dict()
                    end
                    if !haskey(alldata[name][trig][init][option], seed)
                        alldata[name][trig][init][option][seed] = Dict()
                    else
                        continue
                    end
                    alldata[name][trig][init][option][seed]["args"] = args
                    alldata[name][trig][init][option][seed]["allaccs"]= testaccs
                    alldata[name][trig][init][option][seed]["xaxis"] = xaxis
                    if args["gpu"]
                        try
                            BSON.@load joinpath(trial,logs) time timeCPU
                            alldata[name][trig][init][option][seed]["time"] = time
                            BSON.@load joinpath(trial,logs) allinittimes alltrigtime
                            alldata[name][trig][init][option][seed]["allinittimes"] = allinittimes
                            alldata[name][trig][init][option][seed]["alltrigtimes"] = alltrigtimes
                        catch 
                            alldata[name][trig][init][option][seed]["time"] = time_
                        end
                    else
                        try
                            BSON.@load joinpath(trial,logs) timeCPU 
                            alldata[name][trig][init][option][seed]["time"] = timeCPU
                            BSON.@load joinpath(trial,logs) allinittimesCPU alltrigtimesCPU
                            alldata[name][trig][init][option][seed]["allinittimes"] = allinittimesCPU
                            alldata[name][trig][init][option][seed]["alltrigtimes"] = alltrigtimesCPU
                        catch 
                            alldata[name][trig][init][option][seed]["time"] = time_
                        end
                    end
                    alldata[name][trig][init][option][seed]["allactives"] = allactives
                    alldata[name][trig][init][option][seed]["allloss"] = allloss
                else
                    if !haskey(alldata[name], trig)
                        alldata[name][trig] = Dict()
                    end
                    if !haskey(alldata[name][trig], init)
                        alldata[name][trig][init] = Dict()
                    end
                    if !haskey(alldata[name][trig][init], seed)
                        alldata[name][trig][init][seed] = Dict()
                    end
                    alldata[name][trig][init][seed]["args"] = args
                    alldata[name][trig][init][seed]["allaccs"]= testaccs
                    alldata[name][trig][init][seed]["xaxis"] = xaxis
                    alldata[name][trig][init][seed]["trialnum"] = trialnum
                    if addcounts
                        alldata[name][trig][init][seed]["allcounts"] = allcounts_
                    end
                    if args["gpu"]
                        try
                            BSON.@load joinpath(trial,logs) allinittimes alltrigtimes
                            alldata[name][trig][init][seed]["allinittimes"] = allinittimes
                            alldata[name][trig][init][seed]["alltrigtimes"] = alltrigtimes
                            BSON.@load joinpath(trial,logs) time timeCPU
                            alldata[name][trig][init][seed]["timeGPU"] = time
                            alldata[name][trig][init][seed]["time"] = timeCPU
                            time_ = time
                        catch 
                            alldata[name][trig][init][seed]["time"] = time_
                        end
                    else
                        try
                            BSON.@load joinpath(trial,logs) timeCPU 
                            alldata[name][trig][init][seed]["time"] = timeCPU
                            time_ = timeCPU
                            BSON.@load joinpath(trial,logs) allinittimesCPU alltrigtimesCPU 
                            alldata[name][trig][init][seed]["allinittimes"] = allinittimesCPU
                            alldata[name][trig][init][seed]["alltrigtimes"] = alltrigtimesCPU
                        catch 
                            alldata[name][trig][init][seed]["time"] = time_
                        end
                    end
                    alldata[name][trig][init][seed]["allactives"] = allactives
                    alldata[name][trig][init][seed]["allloss"] = allloss
                end
                if length(option) > 0
                    tcpu = alldata[name][trig][init][option][seed]["time"]
                    tgpu = args["gpu"] ? alldata[name][trig][init][option][seed]["timeGPU"] : 0
                else
                    tcpu = alldata[name][trig][init][seed]["time"]
                    tgpu = args["gpu"] ? alldata[name][trig][init][seed]["timeGPU"] : 0
                end
                global cputime += tcpu
                global gputime += tgpu
            end
        end
    end
    if occursin("vggtime", name)
        for (j, trig) in enumerate(sort(collect(keys(alldata[name])), by=trig->trigs[trig]))
            for (i, init) in enumerate(sort(collect(keys(alldata[name][trig])), by=init->inits[init]))
                tts = alldata[name][trig][init][1]["alltrigtimes"]
                if length(tts) > 0
                    println(init == "Static" ? trig : init, " & \$", round(mean(tts); digits=2), " \\pm ", round(std(tts); digits=2), "\$ \\\\")
                end
            end
        end
    end
end
folder = "plots/"
for name in ["vggdynosched"]
    plot(xlabel="Number of Parameters", size=(500,350), left_margin=7px, bottom_margin=7px, xticks=(log10.([3e6,1e7,3e7]), ["3e6","1e7","3e7"]))
    for (j, trig) in enumerate(sort(collect(keys(alldata[name])), by=trig->trigs[trig]))
        for (i, init) in enumerate(sort(collect(keys(alldata[name][trig])), by=init->inits[init]))
            if init == "NeST" || init == "GradMax" || init == "Firefly"
                continue
            end
            actacc = zeros(Float32, 0, 2)
            for (k, seed) in enumerate(keys(alldata[name][trig][init]))
                if haskey(alldata[name][trig][init][seed], "allaccs") && haskey(alldata[name][trig][init][seed], "allactives") 
                    testaccs = alldata[name][trig][init][seed]["allaccs"]
                    testacc = [testaccs[testaccs .> 0][end]]
                    actives = log10(alldata[name][trig][init][seed]["allcounts"][end])
                    actacc = vcat(actacc, [actives testacc])
                end
            end
            if occursin("tatic", trig)
                label = trig
            elseif occursin("andom", init)
                label = string("Dynamic ", init)
            else
                label = init
            end
            if size(actacc)[1] > 1 && all(actacc .> 0)
                covaa = cov(actacc)
                if covaa[1,1] < 0.00001
                    scatter!([mean(actacc[:,1])], [mean(actacc[:,2])], yerror=[std(actacc[:,2])], label=label, markershape=markershapes[j], markersize=6, color=palette10[inits[init]], markerstrokecolor=palette10[inits[init]], alpha=0.7)
                elseif covaa[2,2] < 0.000000001
                    scatter!([mean(actacc[:,1])], [mean(actacc[:,2])], xerror=[std(actacc[:,1])], label=label, markershape=markershapes[j], markersize=6, color=palette10[inits[init]], markerstrokecolor=palette10[inits[init]], alpha=0.7)
                else
                    covellipse!(vec(mean(actacc, dims=1)), covaa, color=palette10[inits[init]], linecolor=:match, label = label, seriesalpha = 0.7)
                end
            end
        end
    end
    plot!([log10(79697674)], seriestype = :vline, color=:grey, alpha=0.25, linestyle=:dash, label=nothing)
    plot!([log10(1256682)], seriestype = :vline, color=:grey, alpha=0.25, linestyle=:dash, label=nothing)
    plot!(legend=:bottomright, ylabel="Test Accuracy")
    savefig(joinpath(folder,"figure4vggacc.png"))
    savefig(joinpath(folder,"figure4vggacc.pdf"))
    l = @layout [
    grid(3,1){0.99h} grid(3,6){0.99w, 0.99h}
    e      grid(1,6){0.99w}
    ]   
    plot(layout=l, size=(1700,600), link=:all, legend=false)
    vgg11actives = [alldata[name]["Medium Static"]["Static"][1]["allactives"][i][1] for i in 1:10]
    init2sp = Dict("NORTH-Select"=>0,"NORTH-Rand"=>1,"NORTH-Weight"=>2)
    allacts=[]
    xaxis = []
    for (j, trig) in enumerate(sort(collect(keys(alldata[name])), by=trig->trigs[trig]))
        for (i, init) in enumerate(sort(collect(keys(alldata[name][trig])), by=init->inits[init]))
            if occursin("tatic", trig) || init == "NeST" || init == "GradMax" || init == "Firefly"
                continue
            end
            for (k, seed) in enumerate(keys(alldata[name][trig][init]))
                xaxis = alldata[name][trig][init][seed]["xaxis"]
                allacts = alldata[name][trig][init][seed]["allactives"]./(vgg11actives)
                bottom = 0*allacts[1]
                for l in 1:10
                    top = bottom + allacts[l]
                    plot!(subplot=init2sp[init]*6+seed+3, xaxis, top, fillrange=bottom, color=palette(:rainbow,10)[l], fillalpha=0.7)
                    bottom = top
                end
            end
        end
    end
    for l in 10:-1:1
        plot!(subplot=15, [0], [0], fillrange=[0], color=palette(:rainbow,10)[l], label=string("Layer ", l, l>8 ? " (Dense)" : " (Conv)"))
    end
    for sp in [9,15,21]
        plot!(subplot=sp,axis=nothing,showaxis=false)
    end
    for sp in 3:15
        plot!(subplot=sp,xformatter=_->"")
    end
    for sp in 23:27
        plot!(subplot=sp, xguide = sp == 25 ? "Epochs" : "", axis=nothing, showaxis = false, xguideposition=:top, top_margin = 40px)
    end
    for sp in 1:3
        plot!(subplot=sp, yguide = sp == 2 ? "Relative Layer Width" : "", axis=nothing, showaxis = false, yguideposition=:right, right_margin = 30px)
    end
    for sp in 4:21
        if (sp - 1) % 6 != 3
            plot!(subplot=sp,yformatter=_->"")
        end
    end
    plot!(subplot=15,legend=:left,legendfontsize=10)
    plot!(subplot=6, title = "NORTH-Select", xguideposition=:top)
    plot!(subplot=12, title = "NORTH-Rand", xguideposition=:top)
    plot!(subplot=18, title = "NORTH-Weight", xguideposition=:top)
    plot!(subplot=22, axis=nothing, showaxis = false)
    plot!(subplot=28, axis=nothing, showaxis = false)
    savefig(joinpath(folder,"figure5vggwidths.png"))
    savefig(joinpath(folder,"figure5vggwidths.pdf"))
end

bslrs = Dict("vgglr"=>[3e-4, 1e-3, 3e-3, 1e-2], "mnistlr"=>[1e-4, 3e-4, 1e-3, 3e-3, 1e-2], "mnistbatchsize"=>[32, 64, 128, 256, 512, 1024, 2048], "simdyno"=>[1,2,4,8,16,32,64])
for name in ["mnistlr", "mnistbatchsize"]
    lrs=bslrs[name]
    if occursin("lr", name)
        xlabel = "Learning Rate"
    elseif occursin("batchsize", name)
        xlabel = "Batchsize"
    end
    plot(xlabel = xlabel, ylabel="Validation Accuracy", legend = occursin("lr", name) ? :bottomright : :bottomleft, xscale=:log, xticks = (lrs, string.(lrs)), layout=(2,1), link=:x, size=(500, 700))
    for (j, trig) in enumerate(sort(collect(keys(alldata[name])), by=trig->trigs[trig]))
        for (i, init) in enumerate(sort(collect(keys(alldata[name][trig])), by=init->inits[init]))
            aa = alldata[name][trig][init]
            seedaccs = fill!(Array{Float32}(undef,5,length(lrs)),NaN)
            seedacts = fill!(Array{Float32}(undef,5,length(lrs)),NaN)
            for (l, lr) in enumerate(sort(collect(keys(aa))))
                for (s, seed) in enumerate(keys(aa[lr]))
                    toadd = aa[lr][seed]["allaccs"][end]
                    if toadd == 0
                        toadd = NaN
                    end
                    seedaccs[s,l] = toadd
                    seedacts[s,l] = sum(aa[lr][seed]["allactives"][i][end] for i in 1:length(aa[lr][seed]["allactives"]))
                end
            end
            if occursin("tatic", trig)
                label = trig
                linestyle = linestyles[j]
            else
                label = init
                linestyle = linestyles[1]
            end
            lrs = lrs[1:length(filter(!isnan,nanmean(seedaccs, 1)[1,:]))]
            if length(filter(!isnan,nanmean(seedaccs, 1)[1,:])) == length(lrs)
                plot!(subplot=1, lrs, filter(!isnan,nanmean(seedaccs, 1)[1,:]), ribbon=filter(!isnan,nanstd(seedaccs, 1)[1,:]), fillalpha=.25, label=label, color=palette10[inits[init]], linestyle=linestyle)
            end
            if length(filter(!isnan,nanmean(seedacts, 1)[1,:])) == length(lrs)
                plot!(subplot=2, lrs, filter(!isnan,nanmean(seedacts, 1)[1,:]), ribbon=filter(!isnan,nanstd(seedacts, 1)[1,:]), fillalpha=.25, color=palette10[inits[init]], linestyle = linestyle)
            end
        end
    end
    plot!(subplot=2, legend = false, ylabel="Total Hidden Neurons")
    if occursin("lr", name)
        savefig(joinpath(folder,"figure7mnistlr.png"))
        savefig(joinpath(folder,"figure7mnistlr.pdf"))
    else
        savefig(joinpath(folder,"figure7mnistbatchsize.png"))
        savefig(joinpath(folder,"figure7mnistbatchsize.pdf"))
    end
end

plot(xlabel = "Independent features", ylabel="Hidden Neurons", xscale=:log, xticks = (bslrs["simdyno"][1:6], string.(bslrs["simdyno"][1:6])), size=(1400, 400), layout=(1,3), link=:all, bottom_margin=25px, left_margin=25px)
for (sp, name) in enumerate(["simdyno", "simdyno2"])
    local lrs=bslrs["simdyno"][1:6]
    for (j, trig) in enumerate(sort(collect(keys(alldata[name])), by=trig->trigs[trig]))
        for (i, init) in enumerate(sort(collect(keys(alldata[name][trig])), by=init->inits[init]))
            aa = alldata[name][trig][init]
            seedacts = fill!(Array{Float32}(undef,5,length(lrs), sp),NaN)
            for (l, lr) in enumerate(sort(collect(keys(aa))))
                for (s, seed) in enumerate(keys(aa[lr]))
                    for i in 1:sp
                        seedacts[s,l,i] = aa[lr][seed]["allactives"][i][end]
                    end
                end
            end
            if occursin("tatic", trig)
                label = trig
                linestyle = linestyles[j]
            else
                label = init
                linestyle = linestyles[1]
            end
            if sp == 2
                label = nothing
            end
            lrs = lrs[1:length(filter(!isnan,nanmean(seedacts[:,:,1], 1)[1,:]))]
            if length(filter(!isnan,nanmean(seedacts[:,:,1], 1)[1,:])) == length(lrs)
                plot!(subplot=sp, lrs, filter(!isnan,nanmean(seedacts[:,:,1], 1)[1,:]), ribbon=filter(!isnan,nanstd(seedacts[:,:,1], 1)[1,:]), fillalpha=.25, label=label, color=palette10[inits[init]], linestyle = linestyle)
                if sp == 2
                    plot!(subplot=3, lrs, filter(!isnan,nanmean(seedacts[:,:,2], 1)[1,:]), ribbon=filter(!isnan,nanstd(seedacts[:,:,2], 1)[1,:]), fillalpha=.25, label=label, color=palette10[inits[init]], linestyle = linestyle)
                end
            end
        end
    end
end
plot!(subplot=1, legend = :right, title = "Single Hidden Layer")
plot!(subplot=2, legend = false, title = "Two Hidden Layers: 1st Layer")
plot!(subplot=3, legend = false, title = "Two Hidden Layers: 2nd Layer")
for sp in 1:3
    plot!([512], seriestype = :hline, color=:grey, alpha = 0.25, linestyle=:dash, subplot=sp, label=nothing)
    plot!([8], seriestype = :hline, color=:grey, alpha = 0.25, linestyle=:dash, subplot=sp, label=nothing)
end
savefig(joinpath(folder,"figure1simdynoact.png"))
savefig(joinpath(folder,"figure1simdynoact.pdf"))

for names in [("mniststaticsched","mnistdynosched")]
    for name in names
        if occursin("dyno", name) 
            plot(xlabel="Epochs", ylabel="Layer 1 Hidden Neurons", legend=:topright, layout=(1,2), size=(1400,400), bottom_margin=25px, left_margin=30px, link=:all)
            plot!(subplot=2, ylabel="Layer 2 Hidden Neurons")
            for (j, trig) in enumerate(sort(collect(keys(alldata[name])), by=trig->trigs[trig]))
                for (i, init) in enumerate(sort(collect(keys(alldata[name][trig])), by=init->inits[init]))
                    if length(values(alldata[name][trig][init])) > 0 && haskey(first(values(alldata[name][trig][init])), "allactives")
                        for l in 1:2
                            aa = alldata[name][trig][init]
                            seedaccs = fill!(Array{Float32}(undef,length(keys(aa)),maximum([length(aaval["allactives"][l]) for aaval in values(aa)])),NaN)
                            for (k, seed) in enumerate(keys(aa))
                                toadd = aa[seed]["allactives"][l]
                                seedaccs[k,1:length(toadd)] .= toadd
                            end
                            xaxis = sort([aaval["xaxis"] for aaval in values(aa)], by=x->length(x))[end]
                            if occursin("tatic", trig)
                                label = trig
                                linestyle = linestyles[j]
                            else
                                label = init
                                linestyle = linestyles[1]
                            end
                            if occursin("mnist", name)
                                if l == 2
                                    label = nothing
                                end
                            else
                                if l == 1
                                    label = nothing
                                end
                            end
                            if ndims(filter(!isnan,nanmean(seedaccs, 1)[1,:])) == 2
                                xaxis = xaxis[1:length(filter(!isnan,nanmean(seedaccs, 1)[1,:]))]
                                plot!(subplot=l, xaxis, filter(!isnan,nanmean(seedaccs, 1)[1,:]), ribbon=filter(!isnan,nanstd(seedaccs, 1)[1,:]), fillalpha=.25, label=label, color=palette10[inits[init]], linestyle=linestyle)
                            else
                                try
                                    xaxis = xaxis[1:length(filter(!isnan,nanmean(seedaccs, 1)))]
                                    plot!(subplot=l, xaxis, filter(!isnan,nanmean(seedaccs, 1)), ribbon=filter(!isnan,nanstd(seedaccs, 1)), fillalpha=.25, label=label, color=palette10[inits[init]], linestyle=linestyle)
                                catch end
                            end
                        end
                    end
                end
            end
            savefig(joinpath(folder, "figure3mnistdynoactives.png"))
            savefig(joinpath(folder, "figure3mnistdynoactives.pdf"))
        end
    end
    sname=names[1]
    dname=names[2]
    plot(layout=(2,2),size=(1000,600),link=:x, bottom_margin=8px)
    allaccs, alltrigs, allinits, alltimes = [],[],[],[],[]
    statictimes, staticaccs =[[],[],[]], [[],[],[]]
    for (j, trig) in enumerate(sort(collect(keys(alldata[sname])), by=trig->trigs[trig]))
        if occursin("tatic", trig)# && ["time"] in keys(first(values(first(values(alldata[sname][trig])))))
            staticaccs[statics[trig]] = [adctseed["allaccs"][end] for adctseed in values(first(values(alldata[sname][trig])))]
            statictimes[statics[trig]] = [log10(adctseed["time"]/(60*60))  for adctseed in values(first(values(alldata[sname][trig])))]
        else
            for (i, init) in enumerate(sort(collect(keys(alldata[sname][trig])), by=init->inits[init]))
                for adctseed in values(alldata[sname][trig][init])
                    if adctseed["allaccs"][end] > 0 #&& ["time"] in keys(adctseed)
                        push!(allaccs, adctseed["allaccs"][end])
                        push!(alltrigs, trig)
                        push!(allinits, init)
                        push!(alltimes, log10(adctseed["time"]/(60*60)) )
                    end
                end
            end
        end
    end
    df = DataFrame(allaccs=allaccs, alltrigs=alltrigs, allinits=allinits, alltimes=alltimes)
    if nrow(df) > 0
        @df df groupedboxplot!(subplot=2, :alltrigs, :allaccs, group = :allinits, box_position = :dodge, ylabel="Test Accuracy", linecolor = :match, fillalpha = 0.5, legend = false, palette=[palette10[inits[init]] for init in sort(unique(df.allinits))], ylims=[0.962,0.983])
        @df df groupedboxplot!(subplot=4, :alltrigs, :alltimes, group = :allinits, box_position = :dodge, ylabel="CPU Hours", linecolor = :match, fillalpha = 0.5, legend = false, palette=[palette10[inits[init]] for init in sort(unique(df.allinits))])
    end
    for si in 1:3
        if length(staticaccs[si]) > 0
            plot!(subplot=2,[mean(staticaccs[si])], ribbon = [std(staticaccs[si])], seriestype = :hline, color=palette10[inits[initnames["static"]]], label=Dict(v => k for (k, v) in statics)[si], fillalpha=0.3, seriesalpha=0.7)
        end
        if length(statictimes[si]) > 0
            plot!(subplot=4,[mean(statictimes[si])], ribbon = [std(statictimes[si])], seriestype = :hline, color=palette10[inits[initnames["static"]]], label=Dict(v => k for (k, v) in statics)[si], fillalpha=0.3, seriesalpha=0.7)
        end
    end
    plot!(subplot=4, yticks=(log10.([3e-1,1e0,3e0,1e1,3e1]), ["3e-1","1e0","3e0","1e1","3e1"]),ylims=log10.([2e-1,5e1]), grid=false)
    plot!(subplot=2, title="Preset Scheduled Strategies", grid=false)
    for (j, trig) in enumerate(sort(collect(keys(alldata[dname])), by=trig->trigs[trig]))
        for (i, init) in enumerate(sort(collect(keys(alldata[dname][trig])), by=init->inits[init]))
            actacc = zeros(Float32, 0, 2)
            acttime = zeros(Float32, 0, 2)
            for (k, seed) in enumerate(keys(alldata[dname][trig][init]))
                if haskey(alldata[dname][trig][init][seed], "allaccs") && haskey(alldata[dname][trig][init][seed], "allactives") && alldata[dname][trig][init][seed]["allaccs"][end] > 0
                    testaccs = alldata[dname][trig][init][seed]["allaccs"]
                    time = log10(alldata[dname][trig][init][seed]["time"]/(60*60))
                    testacc = [testaccs[testaccs .> 0][end]]
                    actives = [sum(alldata[dname][trig][init][seed]["allactives"][i][end] for i in 1:length(alldata[dname][trig][init][seed]["allactives"]))]
                    actacc = vcat(actacc, [actives testacc])
                    acttime = vcat(acttime, [actives time])
                end
            end
            if occursin("tatic", trig)
                label = trig
            elseif occursin("andom", init)
                label = string("Dynamic ", init)
            else
                label = init
            end
            if size(actacc)[1] > 1
                covaa = cov(actacc)
                if covaa[1,1] < 0.001
                    scatter!(subplot=1,[mean(actacc[:,1])], [mean(actacc[:,2])], yerror=[std(actacc[:,2])], label=label, markershape=markershapes[j], markersize=6, color=palette10[inits[init]], markerstrokecolor=palette10[inits[init]], alpha=0.7)
                elseif covaa[2,2] < 0.000000001
                    scatter!(subplot=1,[mean(actacc[:,1])], [mean(actacc[:,2])], xerror=[std(actacc[:,1])], label=label, markershape=markershapes[j], markersize=6, color=palette10[inits[init]], markerstrokecolor=palette10[inits[init]], alpha=0.7)
                else
                    covellipse!(subplot=1,vec(mean(actacc, dims=1)), covaa, color=palette10[inits[init]], linecolor=:match, label = label, seriesalpha = 0.7)
                end
            end
            if size(acttime)[1] > 1
                covat = cov(acttime)
                if covat[1,1] < 0.001
                    scatter!(subplot=3,[mean(acttime[:,1])], [mean(acttime[:,2])], yerror=[std(acttime[:,2])], label=label, markershape=markershapes[j], markersize=6, color=palette10[inits[init]], markerstrokecolor=palette10[inits[init]], alpha=0.7)
                elseif covat[2,2] < 0.000000001
                    scatter!(subplot=3,[mean(acttime[:,1])], [mean(acttime[:,2])], xerror=[std(acttime[:,1])], label=label, markershape=markershapes[j], markersize=6, color=palette10[inits[init]], markerstrokecolor=palette10[inits[init]], alpha=0.7)
                else
                    covellipse!(subplot=3,vec(mean(acttime, dims=1)), covat, color=palette10[inits[init]], linecolor=:match, label = label, linestyle=linestyles[j], seriesalpha = 0.7)
                end
            end
        end
    end
    for sp in [1,3]
        plot!(subplot=sp,[28*28*2], seriestype = :vline, color=:grey, alpha=0.25, linestyle=:dash, label=nothing)
        plot!(subplot=sp,[64*2], seriestype = :vline, color=:grey, alpha=0.25, linestyle=:dash, label=nothing)
    end
    plot!(subplot=1,legend=:bottomright, ylabel="Test Accuracy", title="Dynamic Strategies", ylims=[0.962,0.983], xlabel="Total Hidden Neurons")
    plot!(subplot=3,legend=false, ylabel="CPU Hours", yticks=(log10.([3e-1,1e0,3e0,1e1,3e1]), ["3e-1","1e0","3e0","1e1","3e1"]),ylims=log10.([2e-1,5e1]), xlabel="Total Hidden Neurons")
    savefig(joinpath(folder,"figure2mnistacctime.pdf"))
    savefig(joinpath(folder,"figure2mnistacctime.png"))
end

for si in ["mnist"]
    allorthogs, allinits, alltimes, allmetrics = [], [], [], []
    metrics = ["Effective Dimension", "Orthogonality Gap"]
    plot(xlabel="Total Hidden Neurons", ylabel="Validation Accuracy", layout=(2,3), link=:all, size=(900,500))
    for (n, name) in enumerate([string(si,"csthresh"), "mnistogthresh"])
        for (j, trig) in enumerate([trig for trig in sort(collect(keys(alldata[name])), by=trig->trigs[trig]) if trig == "Dynamic"])
            for (i, init) in enumerate(sort(collect(keys(alldata[name][trig])), by=init->inits[init]))
                for (o, orthog) in enumerate(sort(collect(keys(alldata[name][trig][init]))))
                    actacc = zeros(Float32, 0, 2)
                    for (k, seed) in enumerate(keys(alldata[name][trig][init][orthog]))
                        if haskey(alldata[name][trig][init][orthog][seed], "allaccs") & haskey(alldata[name][trig][init][orthog][seed], "allactives")
                            testaccs = alldata[name][trig][init][orthog][seed]["allaccs"]
                            testacc = [testaccs[testaccs .> 0][end]]
                            actives = [sum(alldata[name][trig][init][orthog][seed]["allactives"][i][end] for i in 1:length(alldata[name][trig][init][orthog][seed]["allactives"]))]
                            actacc = vcat(actacc, [actives testacc])
                            push!(allorthogs, orthog)
                            push!(allmetrics, metrics[n])
                            push!(allinits, init)
                            push!(alltimes, alldata[name][trig][init][orthog][seed]["time"]/(60*60) )
                        end
                    end
                    if 3*(n-1)+i == 4
                        label = string(orthog)
                    else
                        label = nothing
                    end
                    if occursin("andom", init)
                        init_title = string("Dynamic ", init)
                    else
                        init_title = init
                    end
                    if size(actacc)[1] > 1
                        covaa = cov(actacc)
                        if covaa[1,1] < 0.001
                            scatter!(subplot=3*(n-1)+i,[mean(actacc[:,1])], [mean(actacc[:,2])], yerror=[std(actacc[:,2])], label=label, markershape=markershapes[o], markersize=6, color=palette(:viridis, 6)[o], markerstrokecolor=palette(:viridis, 6)[o], alpha=0.5)
                        elseif covaa[2,2] < 0.000000001
                            scatter!(subplot=3*(n-1)+i,[mean(actacc[:,1])], [mean(actacc[:,2])], xerror=[std(actacc[:,1])], label=label, markershape=markershapes[o], markersize=6, color=palette(:viridis, 6)[o], markerstrokecolor=palette(:viridis, 6)[o], alpha=0.5)
                        else
                            covellipse!(subplot=3*(n-1)+i,vec(mean(actacc, dims=1)), covaa, color=palette(:viridis, 6)[o], linecolor=:match, title = n==1 ? init_title : "", label = label, linestyle=linestyles[o], seriesalpha = 0.7)
                            #scatter!(subplot=3*(n-1)+i, actacc[:,1], actacc[:,2], label=nothing, markershape=markershapes[o], markersize=6, color=palette(:viridis, 6)[o])    
                        end
                    end
                end
            end
        end
    end
    for sp in 1:6
        plot!([28*28*2], seriestype = :vline, color=:grey, alpha = 0.25, linestyle=:dash, subplot=sp, label=nothing)
        plot!([64*2], seriestype = :vline, color=:grey, alpha = 0.25, linestyle=:dash, subplot=sp, label=nothing)
    end
    plot!(subplot=4, legend=:bottomright)
    plot!(subplot=1, ylabel="Effective Dimension", yguidefontsize=13, yguidefonthalign=:right, yguidefontvalign=:bottom)
    plot!(subplot=4, ylabel="Orthogonal Gap", yguidefontsize=13, yguidefonthalign=:right, yguidefontvalign=:bottom)
    savefig(joinpath(folder,"figure6mnistthreshacc.png"))
    savefig(joinpath(folder,"figure6mnistthreshacc.pdf"))

    metrics = ["Effective Dimension", "Orthogonality Gap"]
    plot(xlabel="Total Hidden Neurons", ylabel="CPU Hours", layout=(2,3), link=:all, size=(900,500))
    for (n, name) in enumerate([string(si,"csthresh"), "mnistogthresh"])
        for (j, trig) in enumerate([trig for trig in sort(collect(keys(alldata[name])), by=trig->trigs[trig]) if trig == "Dynamic"])
            for (i, init) in enumerate(sort(collect(keys(alldata[name][trig])), by=init->inits[init]))
                for (o, orthog) in enumerate(sort(collect(keys(alldata[name][trig][init]))))
                    actacc = zeros(Float32, 0, 2)
                    for (k, seed) in enumerate(keys(alldata[name][trig][init][orthog]))
                        if haskey(alldata[name][trig][init][orthog][seed], "allaccs") && haskey(alldata[name][trig][init][orthog][seed], "allactives") && alldata[name][trig][init][orthog][seed]["time"] > 0
                            time = log10(alldata[name][trig][init][orthog][seed]["time"]/(60*60))
                            actives = [sum(alldata[name][trig][init][orthog][seed]["allactives"][i][end] for i in 1:length(alldata[name][trig][init][orthog][seed]["allactives"]))]
                            actacc = vcat(actacc, [actives time])
                        end
                    end
                    if 3*(n-1)+i == 4
                        label = string(orthog)
                    else
                        label = nothing
                    end
                    if occursin("andom", init)
                        init_title = string("Dynamic ", init)
                    else
                        init_title = init
                    end
                    if size(actacc)[1] > 1
                        covaa = cov(actacc)
                        if covaa[1,1] < 0.001
                            scatter!(subplot=3*(n-1)+i,[mean(actacc[:,1])], [mean(actacc[:,2])], yerror=[std(actacc[:,2])], label=label, markershape=markershapes[o], markersize=6, color=palette(:viridis, 6)[o], markerstrokecolor=palette(:viridis, 6)[o], alpha=0.5)
                        elseif covaa[2,2] < 0.000000001
                            scatter!(subplot=3*(n-1)+i,[mean(actacc[:,1])], [mean(actacc[:,2])], xerror=[std(actacc[:,1])], label=label, markershape=markershapes[o], markersize=6, color=palette(:viridis, 6)[o], markerstrokecolor=palette(:viridis, 6)[o], alpha=0.5)
                        else
                            covellipse!(subplot=3*(n-1)+i,vec(mean(actacc, dims=1)), covaa, color=palette(:viridis, 6)[o], linecolor=:match, title = n==1 ? init_title : "", label = label, linestyle=linestyles[o], seriesalpha = 0.7)
                        end
                    end
                end
            end
        end
    end
    plot!(subplot=4, legend=:bottomright)
    plot!(subplot=1, ylabel="Effective Dimension", yguidefontsize=13, yguidefonthalign=:right, yguidefontvalign=:bottom)
    plot!(subplot=4, ylabel="Orthogonal Gap", yguidefontsize=13, yguidefonthalign=:right, yguidefontvalign=:bottom)
    plot!(yticks=(log10.([3e0,1e1,3e1]), ["3e0", "1e1", "3e1"]))
    savefig(joinpath(folder,"figure6mnistthreshtime.png"))
    savefig(joinpath(folder,"figure6mnistthreshtime.pdf"))
end
cputime/(60*60), gputime/(60*60)
