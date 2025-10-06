#!/bin/bash


#SBATCH --job-name=fems
#SBATCH --partition=math-alderaan
#SBATCH --output=logs/retrieve_%j.out
#SBATCH --ntasks=2
#SBATCH --mem=16G

# Control script to query FEMS api for fuels samples and save

if [ "$#" -ne 2 ]; then
    echo "Error: Expected exactly 2 arguments, but got $#."
    echo "Usage: $0 <config_file> <output_path>"
    echo "Example: $0 'etc/test_dead.yaml' 'data/test.csv'"
    exit 1
fi

CONF_PATH="$1"
OUT_PATH="$2"

# Set up environment
source ~/.bashrc
conda activate fems

export PYTHONUNBUFFERED=1
python src/retrieve_fems_dead.py $CONF_PATH $OUT_PATH
