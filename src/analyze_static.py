# Executable module to analyze results of static model baselines, XGB and LR
# Requires existing output: outputs/transfer_baseline_static
# Reads in results and summarizes with tables and plots

import numpy as np
import pandas as pd
import yaml
import time
import sys
import os
import os.path as osp
import matplotlib.pyplot as plt
from pathlib import Path


# Set Project Paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")
DATA_DIR = osp.join(PROJECT_ROOT, "data")

# Local Modules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import read_yml, Dict


# Executed Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':

    tdir = "outputs/transfer_baseline_static"
    if not osp.exists(tdir):
        print(f"Can't find required output directory: {tdir}")
        sys.exit(-1)

    print(f"Summarizing Static Results from directory: {tdir}")
    conf = Dict(read_yml(osp.join(CONFIG_DIR, "thesis_config.yaml")))
   
    xgb = pd.read_pickle(osp.join(tdir, "results_xgb_testset.pkl"))
    lm  = pd.read_pickle(osp.join(tdir, "results_lm_testset.pkl"))

    # FM1 Results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~
    tab1 = pd.DataFrame({
        "Metric": [r"$R^2$", r"Bias ($\%$)", r"RMSE ($\%$)"],
        "XGBoost": np.array([xgb["FM1"]["base"]["r2"], xgb["FM1"]["base"]["bias"], xgb["FM1"]["base"]["rmse"]]).round(2),
        "LM": np.array([lm["FM1"]["base"]["r2"], lm["FM1"]["base"]["bias"], lm["FM1"]["base"]["rmse"]]).round(2)
    })

    tab1_30 = pd.DataFrame({
        "Metric": [r"$R^2$", r"Bias ($\%$)", r"RMSE ($\%$)"],
        "XGBoost": np.array([xgb["FM1"]["lt30"]["r2"], xgb["FM1"]["lt30"]["bias"], xgb["FM1"]["lt30"]["rmse"]]).round(2),
        "LM": np.array([lm["FM1"]["lt30"]["r2"], lm["FM1"]["lt30"]["bias"], lm["FM1"]["lt30"]["rmse"]]).round(2)
    })

    # FM10 Results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    tab10 = pd.DataFrame({
        "Metric": [r"$R^2$", r"Bias ($\%$)", r"RMSE ($\%$)"],
        "XGBoost": np.array([xgb["FM10"]["base"]["r2"], xgb["FM10"]["base"]["bias"], xgb["FM10"]["base"]["rmse"]]).round(2),
        "LM": np.array([lm["FM10"]["base"]["r2"], lm["FM10"]["base"]["bias"], lm["FM10"]["base"]["rmse"]]).round(2)
    })

    tab10_30 = pd.DataFrame({
        "Metric": [r"$R^2$", r"Bias ($\%$)", r"RMSE ($\%$)"],
        "XGBoost": np.array([xgb["FM10"]["lt30"]["r2"], xgb["FM10"]["lt30"]["bias"], xgb["FM10"]["lt30"]["rmse"]]).round(2),
        "LM": np.array([lm["FM10"]["lt30"]["r2"], lm["FM10"]["lt30"]["bias"], lm["FM10"]["lt30"]["rmse"]]).round(2)
    })

    # FM100 Results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    tab100 = pd.DataFrame({
        "Metric": [r"$R^2$", r"Bias ($\%$)", r"RMSE ($\%$)"],
        "XGBoost": np.array([xgb["FM100"]["base"]["r2"], xgb["FM100"]["base"]["bias"], xgb["FM100"]["base"]["rmse"]]).round(2),
        "LM": np.array([lm["FM100"]["base"]["r2"], lm["FM100"]["base"]["bias"], lm["FM100"]["base"]["rmse"]]).round(2)
    })


    # FM1000 Results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~

    tab1000 = pd.DataFrame({
        "Metric": [r"$R^2$", r"Bias ($\%$)", r"RMSE ($\%$)"],
        "XGBoost": np.array([xgb["FM1000"]["base"]["r2"], xgb["FM1000"]["base"]["bias"], xgb["FM1000"]["base"]["rmse"]]).round(2),
        "LM": np.array([lm["FM1000"]["base"]["r2"], lm["FM1000"]["base"]["bias"], lm["FM1000"]["base"]["rmse"]]).round(2)
    })

    # Write Outputs
    # ~~~~~~~~~~~~~~~~~~~~~
    print(f"Writing outputs to {tdir}")
    tab1.to_csv(osp.join(tdir, "fm1_accuracy_testset.csv"), index=False)
    tab1_30.to_csv(osp.join(tdir, "fm1_30_accuracy_testset.csv"), index=False)

    tab10.to_csv(osp.join(tdir, "fm10_accuracy_testset.csv"), index=False)
    tab10_30.to_csv(osp.join(tdir, "fm10_30_accuracy_testset.csv"), index=False)
    
    tab100.to_csv(osp.join(tdir, "fm100_accuracy_testset.csv"), index=False)
    
    tab1000.to_csv(osp.join(tdir, "fm1000_accuracy_testset.csv"), index=False)



