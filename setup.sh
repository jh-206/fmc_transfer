#!/bin/bash

set -u

RNN_REPS_DIR="models/rnn_rocky_23-24_reps"
RNN_REPS_ZIP="models/rnn_rocky_23-24_reps.zip"
RNN_REPS_URL="https://zenodo.org/api/records/19927293/files-archive"

warn_missing() {
    local path="$1"
    echo "Warning: expected file not found: $path"
}

count_seed_dirs() {
    find "$1" -maxdepth 1 -type d -name 'seed_*' | wc -l | tr -d ' '
}

echo "Initializing conda environment: fmc"
eval "$(conda shell.bash hook)"
conda activate fmc

echo "Checking expected processed data files"
for path in \
    "data/processed_data/weather.csv" \
    "data/processed_data/ok_1h.csv" \
    "data/processed_data/ok_10h.csv" \
    "data/processed_data/ok_100h.csv" \
    "data/processed_data/ok_1000h.csv"
do
    if [ ! -f "$path" ]; then
        warn_missing "$path"
    fi
done

echo "Checking pretrained RNN replication directory"
seed_count=0
if [ -d "$RNN_REPS_DIR" ]; then
    seed_count=$(count_seed_dirs "$RNN_REPS_DIR")
fi

if [ "$seed_count" -ne 100 ]; then
    echo "RNN replication directory missing or incomplete, downloading archive"
    curl -L "$RNN_REPS_URL" -o "$RNN_REPS_ZIP"
    unzip -o "$RNN_REPS_ZIP" -d models

    seed_count=0
    if [ -d "$RNN_REPS_DIR" ]; then
        seed_count=$(count_seed_dirs "$RNN_REPS_DIR")
    fi

    if [ "$seed_count" -ne 100 ]; then
        echo "Error: expected 100 seed directories in $RNN_REPS_DIR after extraction, found $seed_count"
        exit 1
    fi
fi

echo "Checking expected config files"
for path in \
    "etc/thesis_config.yaml" \
    "etc/params_models.yaml"
do
    if [ ! -f "$path" ]; then
        warn_missing "$path"
    fi
done

echo "Setup completed successfully. Required files were found."
