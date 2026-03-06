#!/bin/bash


#SBATCH --job-name=twarpf
#SBATCH --partition=math-alderaan
#SBATCH --output=logs/reps_%j.out
#SBATCH --ntasks=4
#SBATCH --mem=64G

# Shell file to run a replications of full fine tune twarp 


SEED="$SLURM_ARRAY_TASK_ID"
CONF_PATH="$1"

# Set up environment
source ~/.bashrc
conda activate fmc

python src/transfer_twarp_finetune.py $CONF_PATH $SEED 
