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

The datasets for analysis are created in interactive jupyter notebooks. The notebooks can be opened and run with all cells. They also contain informative print statements and explain the logic used for processing the data. To recreate the datasets used in analysis, open and run:

1. `process_carlson_data.ipynb`
2. `process_ok_mesonet.ipynb`
3. ...weise data notebook coming soon...


## Recreating Outputs

### FM10 Zeroshot

To recreate the accuracy metrics and visualizations associated with the FM10 zeroshot analysis, run all cells in `rnn_10h_zeroshot.ipynb`. The accuracy metrics are printed in-line and two figures are saved:
- `outputs/ts_rnn_zeroshot.png`
- `outputs/ts_BAWC2.png`

### Stability of the Learned Gate Bias Structure

Run `rnn_timewarp_reps.ipynb`

### FMC Transfer - No Fine Tune

To recreate the accuracy metrics and visualizations associated with transfer learning, no fine-tune, run python module with config:

`python src/transfer_twarp_analysis.py etc/thesis_config.yaml`

To create the files `fm1_results.pkl`, `f100 this _results.pkl`, `fm1000_results.pkl`, `results_test_set.pkl`




