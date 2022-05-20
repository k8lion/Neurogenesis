# NORTH*: Neurogenesis driven by Neural Orthogonality 
This repository accompagnies our [work](https://openreview.net/attachment?id=SWOg-arIg9&name=main_paper_and_supplementary_material) accepted at the [1st AutoML conference](https://automl.cc).

Kaitlin Maile, Emmanuel Rachelson, Hervé Luga, Dennis G. Wilson, "When, where, and how to add new neurons to ANNs." AutoML Conference, 2022. 

## Code navigation
The main program for training a growing neural network is found in `exp/runneurogenesis.jl`. This script accepts command line arguments, detailed in `exp/utilities.jl`, such as the trigger and initialization strategies, base architecture, dataset, and hyperparameters. Models and basic operations are defined in `src/models.jl`. Trigger scoring functions are defined in `src/scores.jl`. Initialization functions are defined `src/initializations.jl`.

## Running experiments

0. If you do not have Julia >= 1.6.0, [download and install Julia](https://julialang.org/downloads/) and add it to your `PATH`.

1. Clone this repository.

2. From the main directory of this repository, run: 
```
julia --project -e "using Pkg; Pkg.instantiate()"
```

3. To run a single trial of NORTH-Select neurogenesis on a 2 hidden layer MLP on the generated toy data, run:
```
julia --project exp/runneurogenesis.jl  \
 --trigger svdacts \
 --init orthogact \
 --name simtrial \
 --expdir outputs \
 --seed 1 \
 --dataset sim \
 --effdim 8 \
 --epochs 50
```

4. To run a single trial of NORTH-Weight neurogenesis on a 2 hidden layer MLP on MNIST, run the following line. Note that you will be prompted to whether you would like to download MNIST.
```
julia --project exp/runneurogenesis.jl  \
 --trigger svdweights \
 --init orthogweights \
 --hidden 2 \
 --name mnisttrial \
 --expdir outputs \
 --seed 1 
```

5. All output files of experiments that generated the plots in the paper are already stored in `outputs`. If you wish to rerun any or all of the full experiments, adapt the slurm-based scripts in `exp/scripts` to your system and use the appropriate lines in `exp/scripts/run_all.sh`. Note that `exp/scripts/figure4-5vggdyno.sh` requires a GPU.

6. Generate all plots:
```
julia --project plot.jl
```
