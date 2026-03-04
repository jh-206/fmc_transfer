#!/bin/bash


#SBATCH --job-name=twarp
#SBATCH --partition=math-alderaan
#SBATCH --output=logs/reps_%j.out
#SBATCH --ntasks=4
#SBATCH --mem=64G

# Shell file to run a replications of time warp, all other weights frozen 


SEED="$SLURM_ARRAY_TASK_ID"
CONF_PATH="$1"

# Set up environment
source ~/.bashrc
conda activate fmc

python src/transfer_twarp_analysis.py $CONF_PATH $SEED 
