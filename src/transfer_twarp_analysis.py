# Executable module to run zero-shot transfer analysis
# Methodology: for each fuel class construct grid of time warp params, modify LSTM and generate predictions
### for training+validation period. Pick best fitting accuracy. Generate predictions for test set and compute as final accuracy
# No fine-tuning, only the time-warp param is fit
# This method doesn't utilize a train/validation set split, so combining them together
# Fine-tuning requires the validation set split


import requests as requests
import pandas as pd
import yaml
import time
import sys
import os
import os.path as osp

# Set Project Paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")
DATA_DIR = osp.join(PROJECT_ROOT, "data")

# Local Modules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import read_yml, Dict, time_range

# Metadata files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Module Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Executed Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(f"Usage: {sys.argv[0]} <config_path>")
        print("Example: python src/transfer_zeroshot_analysis.py etc/thesis_config.yaml")
        sys.exit(-1)  

    # Setup 
    confpath = sys.argv[1]
    conf = Dict(read_yml(confpath))
    output_dir = conf.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Time params
    train_times = time_range(conf.train_start, conf.train_end, freq="1h")
    val_times = time_range(conf.val_start, conf.val_end, freq="1h")
    test_times = time_range(conf.f_start, conf.f_end, freq="1h")
    print(f"Train Period: {conf.train_start} to {conf.train_end}")
    print(f"    N. Hours: {train_times.shape[0]}")
    print(f"Val Period:   {conf.val_start} to {conf.val_end}")
    print(f"    N. Hours: {val_times.shape[0]}")
    print(f"Test Period:  {conf.f_start} to {conf.f_end}")
    print(f"    N. Hours: {test_times.shape[0]}")

    # Data
    weather = pd.read_excel(osp.join(DATA_DIR, "processed_data/dvdk_weather.xlsx"))
    fm1 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_1h.xlsx"))
    fm10 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_10h.xlsx"))
    fm100 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_100h.xlsx"))
    fm1000 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_1000h.xlsx"))
    


