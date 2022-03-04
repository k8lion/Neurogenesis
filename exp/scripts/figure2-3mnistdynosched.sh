expdir="mnistdynosched"
trials=5
all=("static randomin" "smallstatic randomin" "bigstatic randomin" "gsvdc firefly" "gsvdc gradmax" "gsvdc nest" "gsvdc randomout" "svdacts solveorthogact" "svdacts orthogact" "svdacts randomin" "svdweights orthogweights")
for seed in $(seq 1 $trials) ; do
    for ti in "${all[@]}" ; do
        sbatch scripts/ngm2.sh $ti $expdir $seed
    done
done

