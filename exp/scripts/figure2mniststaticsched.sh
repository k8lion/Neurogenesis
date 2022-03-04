expdir="mniststaticsched"
trials=5
all=("static randomin" "linear nest" "linear gradmax" "linear firefly" "linear solveorthogact" "linear randomin" "linear orthogact" "linear orthogweights" "batched orthogweights" "batched nest" "batched gradmax" "batched firefly" "batched solveorthogact" "batched randomin" "batched orthogact")
for seed in $(seq 1 $trials) ; do
    for ti in "${all[@]}" ; do
        sbatch scripts/ngm2.sh $ti $expdir $seed
    done
done

