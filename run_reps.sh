#!/bin/bash


#SBATCH --job-name=reps
#SBATCH --partition=math-alderaan
#SBATCH --output=logs/reps_%j.out
#SBATCH --ntasks=4
#SBATCH --mem=64G

# Shell file to run a python module with different replications. 
# The modules need to be able to accept an optional 
# integer argument from the slurm array task, and that will be used to set seed


if [ "$#" -ne 3 ]; then
    echo "Error: Expected exactly 3 arguments, but got $#."
    echo "Usage: $0 <module_shell> <config_file> <nreps>"
    echo "Example: $0 run_10h_zeroshot.sh 'etc/test_dead.yaml' 100"
    exit 1
fi

SHELL_PATH="$1"
CONF_PATH="$2"
NREPS="$3"

sbatch --array=0-$((NREPS-1)) --output=logs/%x_%A_%a.out --ntasks=4 --mem=64G $SHELL_PATH $CONF_PATH 
