expdir="mnistbatchsize"
trials=5
all=("static randomin" "bigstatic randomin" "smallstatic randomin" "svdacts orthogact" "svdacts randomin" "svdweights orthogweights")
bss=("32" "64" "128" "256" "512" "1024" "2048")
for seed in $(seq 1 $trials) ; do
    for ti in "${all[@]}" ; do
        for bs in "${bss[@]}" ; do
            sbatch scripts/ngm2batchsize.sh $ti $expdir $seed $bs
        done
    done
done

