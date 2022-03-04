expdir="mnistcsthresh"
trials=5
all=("svdacts orthogact" "svdacts randomin" "svdacts solveorthogact")
lrs=("0.7" "0.9" "0.97" "0.99" "0.997")
for seed in $(seq 1 $trials) ; do
    for ti in "${all[@]}" ; do
        for lr in "${lrs[@]}" ; do
            sbatch scripts/ngm2cst.sh $ti $expdir $seed $lr
        done
    done
done

