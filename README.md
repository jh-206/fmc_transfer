# Transfer Learning for Different Fuel Sizes

The goal of this repo is to effectively train machine learning models of dead fuel moisture content for more fuel sizes than the standard 10h stick, including 1h and 100h fuels. We will test various transfer learning methods to adapt a RNN pre-trained on 10h sensor data to predict other fuel types.

Corresponding email: jonathon.hirschi@ucdenver.edu

Advisor: Jan Mandel, CU Denver

## Data

The 1996-1997 field study by Carlson et al is a foundational dataset in FMC modeling. It was used to calibrate the Nelson model for operational use. As far as we can tell, this is the only controlled study of 100h fuels in CONUS. This will be used as the main dataset to adapt a model pre-trained on 10h sensors to other fuels.

- Larger Fuels: 100h and 1000h datasets are built using the FEMS API with code originally developed by Angel Farguell and collaborators at WIRC.
- Fine fuels datasets: these data come from small-scale academic studies. Please email the listed correspondance in this document for access to these datasets
	- Oklahoma field study 1996-1997 (Carlson 2007): study used to calibrate Nelson model
	- Hawaii field study 2000-2001 (Weise 2004): study used to compare 1h models, including Nelson

The code expects the following datasets to exist. Please reach out to corresponding author for access:

* `data/oklahoma_Carlson_data.xlsx` : formatted data delivered by Derek van der Kamp, original FMC measurements from Carlson and hourly data from portable weather station
* `data/Slapout_96-97_weather.csv`, `data/Slapout_96-97_rain.csv` : formatted half-hourly weather from nearby Mesonet weather station

The datasets for analysis are created in interactive jupyter notebooks. The notebooks can be opened and run with all cells. They also contain informative print statements and explain the logic used for processing the data. To recreate the datasets used in analysis, open and run: `process_carlson_data.ipynb`

...weise data future work...

## Pretrained RNNs

The pretrained RNN used as the source learning task is from the project `openwfm/ml_fmda`. Instructions on how to recreate those models are in the README. The steps involve retrieving and formatting all HRRR and RAWS within a spatial domain (Rocky Mountain GACC for the paper and this thesis). Then, `train_cpu_reps.sh` is run with the config file `etc/train_config.yaml`, generating 100 replications of the RNN. The replications vary train/val split in training and initial weights of the RNN. No test set is utilized for the source domain in this case. The target tasks make the test set in this project. The source domain RNN was validated with an extensive spatiotemporal cross validation and reported in the paper. 

The required models to recreate the thesis are the set of weights for each 100 replication, and live in the directory `models/reps/seed_i` for i=0,...,99

## Recreating Outputs

The analyses are run with statistical replications using SLURM arrays. To recreate efficiently you will need access to a computing cluster with slurm workflow. Individual python modules could be run with individual seeds as a check, but it won't be efficient to recreate the entire analysis. 

### FM10 Zeroshot

`./run_reps.sh run_10h_zeroshot.sh etc/thesis_config.yaml 100`


`python src/analyze_zeroshot_reps.py`

To recreate the accuracy metrics and visualizations associated with the FM10 zeroshot analysis, run all cells in `rnn_10h_zeroshot.ipynb`. The accuracy metrics are printed in-line and two figures are saved:
- `outputs/ts_rnn_zeroshot.png`
- `outputs/ts_BAWC2.png`

### Stability of the Learned Gate Bias Structure

Run `rnn_timewarp_reps.ipynb`

### Steady State Analysis

`python src/steady_state_reps.py etc/thesis_config.yaml`

`analyze_steadystate_results.ipynb`

### FMC Transfer - No Fine Tune


`./run_reps.sh run_twarp.sh etc/thesis_config.yaml 100`

`python src/analyze_twarp0_reps.py`


To recreate the accuracy metrics and visualizations associated with transfer learning, no fine-tune, run python module with config:

`python src/transfer_twarp_analysis.py etc/thesis_config.yaml`

To create the files `fm1_results.pkl`, `f100_results.pkl`, `fm1000_results.pkl`, `results_test_set.pkl`

Then, tables and figures of results can be recreated with `analyze_transfer_results.ipynb`

### No Transfer Baselines - Static Models

`python src/notransfer_static.py etc/thesis_config.yaml`

`analyze_static_results.ipynb`

### No Transfer Baselines - RNN Direct Train

`./run_reps.sh run_notransfer.sh etc/thesis_config.yaml 100`

`python src/notransfer_rnn.py etc/thesis_config.yaml`


### Transfer - Freeze Recurrent Layer

`./run_reps.sh run_twarp_freeze_recurrent.sh etc/thesis_config.yaml 100`

`python src/transfer_twarp_finetune_freeze_recurrent.py etc/thesis_config.yaml`


### Transfer - Freeze Dense Layer

`./run_reps.sh run_twarp_freeze_dense.sh etc/thesis_config.yaml 100`

`python src/transfer_twarp_finetune_freeze_dense.py etc/thesis_config.yaml`





