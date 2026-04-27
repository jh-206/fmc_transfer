# Transfer Learning for Different Fuel Sizes

The goal of this repo is to effectively train machine learning models of dead fuel moisture content for more fuel sizes than the standard 10h stick, including 1h and 100h fuels. We test transfer learning methods to adapt a RNN pre-trained on 10h sensor data to predict other fuel types, including a new approach based on time-warping the learned recurrent dynamics.

Corresponding email: jonathon.hirschi@ucdenver.edu

Advisor: Jan Mandel, CU Denver

## Required Citation for Carlson Data

If you use `data/oklahoma_Carlson_data.xlsx` or derivative datasets based on it, please cite the original data-study source:

Carlson, J. D., Bradshaw, L. S., Nelson, R. M., Jr., Bensch, R. R., and Jabrzemski, R. (2007). Application of the Nelson model to four timelag fuel classes using Oklahoma field observations: Model evaluation and comparison with National Fire Danger Rating System algorithms. *International Journal of Wildland Fire*, 16, 204--216. https://doi.org/10.1071/WF06073

## Related Thesis

This repository formed the basis of the PhD thesis:

Hirschi, J. (defended April 2026). *Time-Warping Recurrent Neural Networks for Transfer Learning*.
Preprint: [https://arxiv.org/abs/2604.02474](https://arxiv.org/abs/2604.02474)
An official ProQuest link will be added when available.

The thesis studies transfer learning for fuel moisture prediction using a time-warping approach applied to recurrent neural networks.
In particular, it investigates transfer learning by modifying the learned dynamics of a pre-trained LSTM to adapt models across fuel classes.

## Relationship to Prior RNN Work

The transfer-learning experiments in this repository build on pre-trained models developed in `openwfm/ml_fmda`:
[https://github.com/openwfm/ml_fmda](https://github.com/openwfm/ml_fmda)

Those source models are associated with:

Mandel, J., Farguell, A., Haley, C., and colleagues. *A Recurrent Neural Network for Forecasting Dead Fuel Moisture Content with Inputs from Numerical Weather Models*. *Fire*, 9(1), 26. [https://doi.org/10.3390/fire9010026](https://doi.org/10.3390/fire9010026)

Reproducing the thesis results in this repository requires the corresponding pre-trained model weights from that upstream project.

## Data

The 1996-1997 field study by Carlson et al is a foundational dataset in FMC modeling. It was used to calibrate the Nelson model for operational use. As far as we can tell, this is the only controlled study of 100h fuels in CONUS. This will be used as the main dataset to adapt a model pre-trained on 10h sensors to other fuels.

### Source Data Provenance

The file `data/oklahoma_Carlson_data.xlsx` contains the Oklahoma field observations used in this project, including FMC measurements and associated weather-sensor data.
The version distributed in this repository was provided in this formatted form by Derek W. van der Kamp for the study:

Van der Kamp, D. W., Moore, R. D., and McKendry, I. G. (2017). A model for simulating the moisture content of standardized fuel sticks of various sizes. *Agricultural and Forest Meteorology*, 236, 123--134. https://doi.org/10.1016/j.agrformet.2017.01.013

The underlying study data originate from Carlson et al.
This repository includes the formatted file with permission from the original author.

- Larger Fuels:
  - Oklahoma field study for 100h and 1000h fuels
  - FEMS field samples built using the FEMS API with code originally developed by Angel Farguell and collaborators at WIRC
- Fine fuels datasets: these data come from small-scale academic studies. Please email the listed correspondance in this document for access to these datasets
  - Oklahoma field study 1996-1997 (Carlson 2007): study used to calibrate Nelson model
  - Hawaii field study 2000-2001 (Weise 2004): study used to compare 1h models, including Nelson
  - FEMS field samples

The code expects the following datasets to exist:

* `data/oklahoma_Carlson_data.xlsx` : formatted Oklahoma field-study data including FMC measurements and weather-sensor observations
* processed analysis data created by the notebooks in this repository

The datasets for analysis are created in interactive jupyter notebooks.
These processed files reflect my own cleaning and preparation workflow and are the versions used in the reproducible analyses in this repository.
The notebooks can be opened and run with all cells.
They also contain informative print statements and explain the logic used for processing the data.
To recreate the processed datasets used in analysis, open and run: `process_carlson_data.ipynb`

...weise data future work...

## Pretrained RNNs

The required models to recreate the thesis are the set of weights for each 100 replication, and live in the directory `models/reps/seed_i` for i=0,...,99

The pretrained RNN used as the source learning task is from the project `openwfm/ml_fmda`. Instructions on how to recreate those models are in the README. The steps involve retrieving and formatting all HRRR and RAWS within a spatial domain (Rocky Mountain GACC for the paper and this thesis). Then, `train_cpu_reps.sh` is run with the config file `etc/train_config.yaml`, generating 100 replications of the RNN. The replications vary train/val split in training and initial weights of the RNN. No test set is utilized for the source domain in this case. The target tasks make the test set in this project. The source domain RNN was validated with an extensive spatiotemporal cross validation and reported in the paper. 

## System Requirements

This project is designed to be run on a Unix-like system with the following available:

* A SLURM workload manager for submitting array jobs used in the replicated analyses
* Conda available on the command line, so the provided shell scripts can activate project environments

The full replicated analyses were designed for a computing-cluster workflow.
Some individual python modules can be run directly for spot checks or smaller tests, but reproducing the main results efficiently assumes access to a SLURM-compatible system.

## Replicating Outputs

The analyses are run with statistical replications using SLURM arrays. To recreate efficiently you will need access to a computing cluster with slurm workflow. Individual python modules could be run with individual seeds as a check, but it won't be efficient to recreate the entire analysis. 

Additional outputs, including summary tables and figures, were created using interactive jupyter notebooks. The directory `docs/` contains several notebooks that generated outputs used in the thesis and the paper. Notebooks with the prefix `analyze_*` were run to create tables, which were manually copied into latex form for the papers, and figures that are saved to the `outputs/` directory

### FM10 Zeroshot

`./run_reps.sh run_10h_zeroshot.sh etc/thesis_config.yaml 100`


`python src/analyze_zeroshot_reps.py`


### Stability of the Learned Gate Bias Structure

Run `rnn_timewarp_reps.ipynb`

### Steady State Analysis

`python src/steady_state_reps.py etc/thesis_config.yaml`

`analyze_steadystate_results.ipynb`

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

## AI Assistance Disclosure

AI-based tools, including ChatGPT and Codex, were used in a limited support role during this project.
They were used for coding assistance, literature search support, and technical editing of project text and documentation.

These tools did not independently design the study, perform the scientific analysis, or determine the conclusions.
All substantive analytical decisions, data preparation choices, model evaluation, result interpretation, and final written claims were made and verified by the author.
