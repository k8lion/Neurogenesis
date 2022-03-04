#!/bin/sh
#SBATCH --job-name=ngvgg
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=GPUNodes
#SBATCH --gres=gpu:1
#SBATCH --gres-flags=enforce-binding
note="${SLURM_JOB_NAME}"
trial_name="${SLURM_JOB_ID}_${note}"
cd ~/Neurogenesis.jl
module purge
julia --project exp/runneurogenesis.jl \
 --trigger $1 \
 --init $2 \
 --epochs 100 \
 --batchsize 128 \
 --name $trial_name \
 --expdir $3 \
 --vgg \
 --dataset cifar10 \
 --gpu \
 --seed $4 
 

