#!/bin/bash


#SBATCH --job-name=twarp
#SBATCH --partition=math-alderaan
#SBATCH --output=logs/reps_%j.out
#SBATCH --ntasks=4
#SBATCH --mem=64G

# Shell file to run replications of no transfer, direct train of RNN


SEED="$SLURM_ARRAY_TASK_ID"
CONF_PATH="$1"

# Set up environment
source ~/.bashrc
conda activate fmc

python src/src/notransfer_rnn.py $CONF_PATH $SEED 
