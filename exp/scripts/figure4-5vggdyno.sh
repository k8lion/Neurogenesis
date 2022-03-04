expdir="vggdynosched"
trials=5
all=("static randomin" "smallstatic randomin" "bigstatic randomin" "svdweights orthogweights" "svdacts orthogact" "svdacts randomin") #"gsvdc firefly" "gsvdc gradmax" "gsvdc nest"
for seed in $(seq 1 $trials) ; do
    for ti in "${all[@]}" ; do
        sbatch scripts/ngvgg.sh $ti $expdir $seed
    done
done

