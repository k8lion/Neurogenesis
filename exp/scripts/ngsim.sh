#!/bin/sh
#SBATCH --job-name=ngsim
#SBATCH -n 4
#SBATCH -N 1
#SBATCH --mem=64000
note="${SLURM_JOB_NAME}"
trial_name="${SLURM_JOB_ID}_${note}"
cd ~/Neurogenesis.jl
module purge
julia --project exp/runneurogenesis.jl \
 --trigger $1 \
 --init $2 \
 --name $trial_name \
 --expdir $3 \
 --seed $4 \
 --dataset sim \
 --effdim $5 \
 --epochs -1