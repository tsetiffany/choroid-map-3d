#!/bin/bash

#SBATCH --time=48:00:00
#SBATCH --account=st-xtang19-1-gpu
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --job-name=optuna_unet
#SBATCH --output=optuna_output_%j.out
#SBATCH --error=optuna_error_%j.err

module load gcc/9.4.0
module load cuda

source ~/miniconda3/etc/profile.d/conda.sh
conda activate vessel_unet

cd $SLURM_SUBMIT_DIR
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

N_TRIALS=${1:-50}

python optuna_search.py run $N_TRIALS
