# Executable module to analyze results of timewarp that only tunes bias shift
# Requires existing output: outputs/transfer0_reps
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

    reps_dir = "outputs/transfer0_reps"
    if not osp.exists(reps_dir):
        print(f"Can't find required output directory: {reps_dir}")
        sys.exit(-1)

    print(f"Summarizing Time-Warp Transfer Results from directory: {reps_dir}")
    conf = Dict(read_yml(osp.join(CONFIG_DIR, "thesis_config.yaml")))
    
    # Get all results files across replications
    files = sorted(Path(reps_dir).glob("seed_*/results_test_set.pkl"),key=lambda x: int(x.parent.name.split("_")[-1]))
    results = [pd.read_pickle(f) for f in files]
    
    # FM1 Results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Data Count Summaries
    fm1 = [r["FM1"] for r in results]
    n_preds = [r["preds1"].shape[0] for r in fm1];      assert len(np.unique(n_preds)) == 1, f"Preds length don't match"
    n_obs   = [r["preds1_intp"].shape[0] for r in fm1]; assert len(np.unique(n_obs)) == 1, f"Observed length don't match"
    tab1 = pd.DataFrame({
        'Metric': ["Start Time", "End Time", "N. Hours", "N. Obs"],
        'Value' : [conf.f_start, conf.f_end, int(n_preds[0]), int(n_obs[0])]
    })

    # Accuracy Summaries, average metric and calc pm 1 std
    metrics = ['rmse', 'bias', 'r2', 'rmse_30', 'bias_30', 'r2_30']
    rows = []
    for m in metrics:
        vals = np.array([r[m] for r in fm1], dtype=float)
        
        rows.append({
            "Metric": m,
            "Mean Value": vals.mean(),
            "Std": vals.std(),
            "Median": np.median(vals),
            "Low": vals.min(),
            "High": vals.max()
        })
    df = pd.DataFrame(rows)
    # Median RMSE Case
    ## will be used for plotting
    vals = np.array([r['rmse_30'] for r in fm1], dtype=float)
    median_val = np.median(vals)
    median_idx = np.argmin(np.abs(vals - median_val))
    seed_path = files[median_idx].parent

    # Write output
    print(f"Writing summary counts to: {osp.join(reps_dir, 'fm1_summary_counts.csv')}")
    tab1.to_csv(osp.join(reps_dir, "fm1_summary_counts.csv"))
    print(f"Writing accuracy metrics to: {osp.join(reps_dir, 'fm1_accuracy_testset.csv')}")
    df.to_csv(osp.join(reps_dir, "fm1_accuracy_testset.csv"))

    print(f"Writing median replication report to: {osp.join(reps_dir, 'fm1_median_rep_report.txt')}")
    with open(osp.join(reps_dir, "fm1_median_rep_report.txt"), "w") as f:
        f.write(f"Median replication index: {median_idx}\n")
        f.write(f"Seed directory: {seed_path.resolve()}\n\n")
        f.write("Metrics:\n")
        
        for m in metrics:
            f.write(f"{m}: {fm1[median_idx][m]}\n")


    # FM100 Results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Data Count Summaries
    fm100 = [r["FM100"] for r in results]
    n_preds = [r["preds100"].shape[0] for r in fm100];      assert len(np.unique(n_preds)) == 1, f"Preds length don't match"
    n_obs   = [r["preds100_intp"].shape[0] for r in fm100]; assert len(np.unique(n_obs)) == 1, f"Observed length don't match"
    tab100 = pd.DataFrame({
        'Metric': ["Start Time", "End Time", "N. Hours", "N. Obs"],
        'Value' : [conf.f_start, conf.f_end, int(n_preds[0]), int(n_obs[0])]
    })


    # Accuracy Summaries, average metric and calc pm 1 std
    metrics = ['rmse', 'bias', 'r2']
    rows = []
    for m in metrics:
        vals = np.array([r[m] for r in fm100], dtype=float)

        rows.append({
            "Metric": m,
            "Mean Value": vals.mean(),
            "Std": vals.std(),
            "Median": np.median(vals),
            "Low": vals.min(),
            "High": vals.max()
        })
    df = pd.DataFrame(rows)
    # Median RMSE Case
    ## will be used for plotting
    vals = np.array([r['rmse'] for r in fm100], dtype=float)
    median_val = np.median(vals)
    median_idx = np.argmin(np.abs(vals - median_val))
    seed_path = files[median_idx].parent

    # Write output
    print(f"Writing summary counts to: {osp.join(reps_dir, 'fm100_summary_counts.csv')}")
    tab100.to_csv(osp.join(reps_dir, "fm100_summary_counts.csv"))
    print(f"Writing accuracy metrics to: {osp.join(reps_dir, 'fm100_accuracy_testset.csv')}")
    df.to_csv(osp.join(reps_dir, "fm100_accuracy_testset.csv"))

    print(f"Writing median replication report to: {osp.join(reps_dir, 'fm100_median_rep_report.txt')}")
    with open(osp.join(reps_dir, "fm100_median_rep_report.txt"), "w") as f:
        f.write(f"Median replication index: {median_idx}\n")
        f.write(f"Seed directory: {seed_path.resolve()}\n\n")
        f.write("Metrics:\n")

        for m in metrics:
            f.write(f"{m}: {fm100[median_idx][m]}\n")



    # FM1000 Results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Data Count Summaries
    fm1000 = [r["FM1000"] for r in results]
    n_preds = [r["preds1000"].shape[0] for r in fm1000];      assert len(np.unique(n_preds)) == 1, f"Preds length don't match"
    n_obs   = [r["preds1000_intp"].shape[0] for r in fm1000]; assert len(np.unique(n_obs)) == 1, f"Observed length don't match"
    tab1000 = pd.DataFrame({
        'Metric': ["Start Time", "End Time", "N. Hours", "N. Obs"],
        'Value' : [conf.f_start, conf.f_end, int(n_preds[0]), int(n_obs[0])]
    })


    # Accuracy Summaries, average metric and calc pm 1 std
    metrics = ['rmse', 'bias', 'r2']
    rows = []
    for m in metrics:
        vals = np.array([r[m] for r in fm1000], dtype=float)

        rows.append({
            "Metric": m,
            "Mean Value": vals.mean(),
            "Std": vals.std(),
            "Median": np.median(vals),
            "Low": vals.min(),
            "High": vals.max()
        })
    df = pd.DataFrame(rows)
    # Median RMSE Case
    ## will be used for plotting
    vals = np.array([r['rmse'] for r in fm1000], dtype=float)
    median_val = np.median(vals)
    median_idx = np.argmin(np.abs(vals - median_val))
    seed_path = files[median_idx].parent


    # Write output
    print(f"Writing summary counts to: {osp.join(reps_dir, 'fm1000_summary_counts.csv')}")
    tab1000.to_csv(osp.join(reps_dir, "fm1000_summary_counts.csv"))
    print(f"Writing accuracy metrics to: {osp.join(reps_dir, 'fm1000_accuracy_testset.csv')}")
    df.to_csv(osp.join(reps_dir, "fm1000_accuracy_testset.csv"))

    print(f"Writing median replication report to: {osp.join(reps_dir, 'fm1000_median_rep_report.txt')}")
    with open(osp.join(reps_dir, "fm1000_median_rep_report.txt"), "w") as f:
        f.write(f"Median replication index: {median_idx}\n")
        f.write(f"Seed directory: {seed_path.resolve()}\n\n")
        f.write("Metrics:\n")

        for m in metrics:
            f.write(f"{m}: {fm1000[median_idx][m]}\n")

