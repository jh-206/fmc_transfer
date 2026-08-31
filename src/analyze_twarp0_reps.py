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

    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(f"Usage: {sys.argv[0]} <reps_directory>")
        print("Example: python src/transfer_twarp_analysis.py outputs/transfer0_reps")
        sys.exit(1)        

    reps_dir = sys.argv[1]
    if not osp.exists(reps_dir):
        print(f"Can't find required output directory: {reps_dir}")
        sys.exit(-1)

    print(f"Summarizing Time-Warp Transfer Results from directory: {reps_dir}")
    conf = Dict(read_yml(osp.join(reps_dir, "config.yaml")))
    
    # Get all results files across replications
    files = sorted(Path(reps_dir).glob("seed_*/results_test_set.pkl"),key=lambda x: int(x.parent.name.split("_")[-1]))
    results = [pd.read_pickle(f) for f in files]

    # Overall Seed Summary
    # Tabular results for all 100
    fm1 = [r["FM1"] for r in results]
    fm100 = [r["FM100"] for r in results]
    fm1000 = [r["FM1000"] for r in results]

    df_all = pd.DataFrame([
        {
            "seed": i,

            "bf_1": d1["params"]["bf"],
            "bi_1": d1["params"]["bi"],
            "rmse_1": d1["rmse_30"],

            "bf_100": d100["params"]["bf"],
            "bi_100": d100["params"]["bi"],
            "rmse_100": d100["rmse"],

            "bf_1000": d1000["params"]["bf"],
            "bi_1000": d1000["params"]["bi"],
            "rmse_1000": d1000["rmse"],
        }
        for i, (d1, d100, d1000) in enumerate(zip(fm1, fm100, fm1000))
    ])
    print(f"Writing overall seed summary to: {osp.join(reps_dir, 'all_seeds_summary.csv')}")
    df_all.to_csv(osp.join(osp.join(reps_dir, 'all_seeds_summary.csv')))

    # FM1 Results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Data Count Summaries
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

    # Summarize Gate Bias Params
    # Accuracy Summaries, average metric and calc pm 1 std
    values = ['bf', 'bi']
    bfs = np.array([r["params"]["bf"] for r in fm1])
    bis = np.array([r["params"]["bi"] for r in fm1])
    rows = []
    dfb = pd.DataFrame({
        'Time-Warp Parameter': ['bf', 'bi'],
        "Mean Value": [bfs.mean(), bis.mean()],
        "Std": [bfs.std(), bis.std()],
        "Median": [np.median(bfs), np.median(bis)],
        "Low": [bfs.min(), bis.min()],
        "High": [bfs.max(), bis.max()]
    })

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

    print(f"Writing Twarp Param Summary to: {osp.join(reps_dir, 'fm1_twarps.csv')}")
    dfb.to_csv(osp.join(reps_dir, 'fm1_twarps.csv'))
    
    print(f"Writing median replication report to: {osp.join(reps_dir, 'fm1_median_rep_report.txt')}")
    with open(osp.join(reps_dir, "fm1_median_rep_report.txt"), "w") as f:
        f.write(f"Median replication index: {median_idx}\n")
        f.write(f"Seed directory: {seed_path.resolve()}\n\n")
        f.write(f"Twarp Params: {fm1[median_idx]['params']}")
        f.write("Metrics:\n")
        
        for m in metrics:
            f.write(f"{m}: {fm1[median_idx][m]}\n")


    # FM100 Results
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Data Count Summaries
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

    # Summarize Gate Bias Params
    # Accuracy Summaries, average metric and calc pm 1 std
    values = ['bf', 'bi']
    bfs = np.array([r["params"]["bf"] for r in fm100])
    bis = np.array([r["params"]["bi"] for r in fm100])
    rows = []
    dfb = pd.DataFrame({
        'Time-Warp Parameter': ['bf', 'bi'],
        "Mean Value": [bfs.mean(), bis.mean()],
        "Std": [bfs.std(), bis.std()],
        "Median": [np.median(bfs), np.median(bis)],
        "Low": [bfs.min(), bis.min()],
        "High": [bfs.max(), bis.max()]
    }) 

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

    print(f"Writing Twarp Param Summary to: {osp.join(reps_dir, 'fm100_twarps.csv')}")
    dfb.to_csv(osp.join(reps_dir, 'fm100_twarps.csv'))

    print(f"Writing median replication report to: {osp.join(reps_dir, 'fm100_median_rep_report.txt')}")
    with open(osp.join(reps_dir, "fm100_median_rep_report.txt"), "w") as f:
        f.write(f"Median replication index: {median_idx}\n")
        f.write(f"Seed directory: {seed_path.resolve()}\n\n")
        f.write(f"Twarp Params: {fm100[median_idx]['params']}")
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

    # Summarize Gate Bias Params
    # Accuracy Summaries, average metric and calc pm 1 std
    values = ['bf', 'bi']
    bfs = np.array([r["params"]["bf"] for r in fm1000])
    bis = np.array([r["params"]["bi"] for r in fm1000])
    rows = []
    dfb = pd.DataFrame({
        'Time-Warp Parameter': ['bf', 'bi'],
        "Mean Value": [bfs.mean(), bis.mean()],
        "Std": [bfs.std(), bis.std()],
        "Median": [np.median(bfs), np.median(bis)],
        "Low": [bfs.min(), bis.min()],
        "High": [bfs.max(), bis.max()]
    })

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

    print(f"Writing Twarp Param Summary to: {osp.join(reps_dir, 'fm1000_twarps.csv')}")
    dfb.to_csv(osp.join(reps_dir, 'fm1000_twarps.csv'))
    print(f"Writing median replication report to: {osp.join(reps_dir, 'fm1000_median_rep_report.txt')}")
    with open(osp.join(reps_dir, "fm1000_median_rep_report.txt"), "w") as f:
        f.write(f"Median replication index: {median_idx}\n")
        f.write(f"Seed directory: {seed_path.resolve()}\n\n")
        f.write(f"Twarp Params: {fm1000[median_idx]['params']}")
        f.write("Metrics:\n")

        for m in metrics:
            f.write(f"{m}: {fm1000[median_idx][m]}\n")

