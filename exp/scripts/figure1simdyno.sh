expdir="simdyno"
trials=5
eds=("1" "2" "4" "8" "16" "32")
all=("gsvdc firefly" "gsvdc gradmax" "gsvdc nest" "gsvdc randomout" "svdacts solveorthogact" "svdacts orthogact" "svdacts randomin" "svdweights orthogweights")
for seed in $(seq 1 $trials) ; do
    for ed in "${eds[@]}" ; do
        for ti in "${all[@]}" ; do
            sbatch scripts/ngsim2.sh $ti $expdir $seed $ed
            sbatch scripts/ngsim.sh $ti $expdir $seed $ed
        done
    done
done

