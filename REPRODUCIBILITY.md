# Reproducibility Guide

This document describes the inputs, environment, and execution workflow needed to reproduce the main transfer-learning analyses in this repository. The primary targets are the PhD thesis preprint, *Time-Warping Recurrent Neural Networks for Transfer Learning* ([arXiv:2604.02474](https://arxiv.org/abs/2604.02474)), and the corresponding transfer-learning paper based on the same analysis workflow, currently in preparation for publication.

## Starting Point

- `100` pretrained RNN replications: available at `PLACEHOLDER`. Download the archive, unzip it, and place the resulting directory at `models/rnn_rocky_23-24_reps/`.
- `data/processed_data/` files: these are included in the GitHub repository as the starting analysis datasets. The processed datasets can also be regenerated interactively from `data/oklahoma_Carlson_data.xlsx` by opening and running `docs/process_carlson_data.ipynb`.

## System Requirements

This project is designed to be run on a Unix-like system with the following available:

* A SLURM workload manager for submitting array jobs used in the replicated analyses
* Conda available on the command line, so the provided shell scripts can activate project environments
* `curl` available on the command line for downloading the hosted pretrained RNN replication archive

The full replicated analyses were designed for a computing-cluster workflow.
Some individual python modules can be run directly for spot checks or smaller tests, but reproducing the main results efficiently assumes access to a SLURM-compatible system.
# Reproducibility Guide

This document describes the inputs, environment, and execution workflow needed to reproduce the main transfer-learning analyses in this repository. The primary targets are the PhD thesis preprint, *Time-Warping Recurrent Neural Networks for Transfer Learning* ([arXiv:2604.02474](https://arxiv.org/abs/2604.02474)), and the corresponding transfer-learning paper based on the same analysis workflow, currently in preparation for publication.

## Starting Point

- `100` pretrained RNN replications: available at `PLACEHOLDER`. Download the archive, unzip it, and place the resulting directory at `models/rnn_rocky_23-24_reps/`.
- `data/processed_data/` files: these are included in the GitHub repository as the starting analysis datasets, including `weather.csv`, `ok_1h.csv`, `ok_10h.csv`, `ok_100h.csv`, and `ok_1000h.csv`. They can also be regenerated interactively from `data/oklahoma_Carlson_data.xlsx` by opening and running `docs/process_carlson_data.ipynb`.

## Replicating Outputs

The analyses are run with statistical replications using SLURM arrays. To recreate efficiently you will need access to a computing cluster with slurm workflow. Individual python modules could be run with individual seeds as a check, but it won't be efficient to recreate the entire analysis. The slurm array number defines a random seed. This is used to extract the right pretrained RNN replication from the directory `models/rnn_rocky_23-24_reps/seed_*`, and also it sets the random seed in that local environment. 

Additional outputs, including summary tables and figures, were created using interactive jupyter notebooks. The directory `docs/` contains several notebooks that generated outputs used in the thesis and the paper. Notebooks with the prefix `analyze_*` were run to create tables, which were manually copied into latex form for the papers, and figures that are saved to the `outputs/` directory

### FM10 Zeroshot

`./run_reps.sh run_10h_zeroshot.sh etc/thesis_config.yaml 100`


`python src/analyze_zeroshot_reps.py`


### Stability of the Learned Gate Bias Structure

Open jupyter notebook and run all cells in `docs/analyze_bias_reps.ipynb`

## Transfer Learning Scenarios

### No Transfer Baselines - Static Models

`python src/notransfer_static.py etc/thesis_config.yaml`

`analyze_static_results.ipynb`

### No Transfer Baselines - RNN Direct Train

`./run_reps.sh run_notransfer.sh etc/thesis_config.yaml 100`

`python src/notransfer_rnn.py etc/thesis_config.yaml`

### FMC Time Warp Transfer - No Fine Tune

Fits timewarp params, all other weights frozen

Makes grid of timewarp params and picks best on validation set. Uses to predict test set.

`./run_reps.sh run_twarp.sh etc/thesis_config.yaml 100`

`python src/analyze_twarp0_reps.py`

### Transfer - Full Fine Tune

No time warp, no frozen layers

`./run_reps.sh run_finetune.sh etc/thesis_config.yaml 100`

`python src/analyze_finetune.py`

### Transfer - Freeze Recurrent Layer

Transfer learning taking pretrained RNN and fine-tuning to OK field data with frozen recurrent layer. No time warping

`./run_reps.sh run_freeze_recurrent.sh etc/thesis_config.yaml 100`

`python src/analyze_freeze_recurrent.py`

### Transfer - Freeze Dense Layer

Transfer learning taking pretrained RNN and fine-tuning to OK field data with frozen dense layer. No time warping

`./run_reps.sh run_freeze_dense.sh etc/thesis_config.yaml 100`

`python src/analyze_freeze_dense.py`


### Transfer - TimeWarp and Full Fine Tune

`./run_reps.sh run_twarp_finetune.sh etc/thesis_config.yaml 100`

`python src/transfer_twarp_finetune.py etc/thesis_config.yaml`
