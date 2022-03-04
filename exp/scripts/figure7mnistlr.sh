expdir="mnistlr"
trials=5
all=("static randomin" "bigstatic randomin" "smallstatic randomin" "svdacts orthogact" "svdacts randomin" "svdweights orthogweights")
lrs=("1e-4" "3e-4" "1e-3" "3e-3" "1e-2")
for seed in $(seq 1 $trials) ; do
    for ti in "${all[@]}" ; do
        for lr in "${lrs[@]}" ; do
            sbatch scripts/ngm2lr.sh $ti $expdir $seed $lr
        done
    done
done

