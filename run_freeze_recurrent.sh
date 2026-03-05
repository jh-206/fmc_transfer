#!/bin/bash


#SBATCH --job-name=frzr
#SBATCH --partition=math-alderaan
#SBATCH --output=logs/reps_%j.out
#SBATCH --ntasks=4
#SBATCH --mem=64G

# Shell file to run a replications of transfer learning, freeze recurrent layer 


SEED="$SLURM_ARRAY_TASK_ID"
CONF_PATH="$1"

# Set up environment
source ~/.bashrc
conda activate fmc

python src/transfer_freeze_recurrent.py $CONF_PATH $SEED 
