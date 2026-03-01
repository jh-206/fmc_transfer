# Executable module to run transfer analysis 
# Baseline XGBoost regression fit to data

import numpy as np
import pandas as pd
import yaml
import time
import sys
import os
import os.path as osp
import joblib
import pickle
from itertools import product
from sklearn.metrics import mean_squared_error, r2_score

# Set Project Paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")
DATA_DIR = osp.join(PROJECT_ROOT, "data")

# Local Modules
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import read_yml, Dict, time_range, time_intp
from models import moisture_rnn as mrnn
import reproducibility
from models.moisture_static import XGB, LM

# Metadata files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

params_xgb = Dict(read_yml(osp.join(CONFIG_DIR, "params_static.yaml"), subkey="xgb"))
params_lm = Dict(read_yml(osp.join(CONFIG_DIR, "params_static.yaml"), subkey="lm"))


# Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """
    Helper func =eturns rmse, r2, bias using NaN-safe means.
    - NaNs are excluded based on y_true.
    - Optional `mask` further subsets the valid points.
    """
    valid = ~np.isnan(y_true)
    if mask is not None:
        valid = valid & mask

    # If no valid points, return NaNs (keeps downstream code from crashing unexpectedly)
    if not np.any(valid):
        return {"rmse": np.nan, "r2": np.nan, "bias": np.nan, "n": 0}

    err = y_pred[valid] - y_true[valid]
    mse = np.mean(err**2)              # safe: err has no NaN
    bias = np.mean(y_true[valid] - y_pred[valid])
    r2 = r2_score(y_true[valid], y_pred[valid])

    return {"rmse": np.sqrt(mse), "r2": r2, "bias": bias, "n": int(np.sum(valid))}


# Executed Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(f"Usage: {sys.argv[0]} <config_path>")
        print("Example: python src/transfer_zeroshot_analysis.py etc/thesis_config.yaml")
        sys.exit(-1)  

    # Setup 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    confpath = sys.argv[1]
    conf = Dict(read_yml(confpath))
    output_dir = osp.join(conf.output_dir, "transfer_baseline_static")
    os.makedirs(output_dir, exist_ok=True)
    print(f"~"*50)
    print(f"Running Transfer-Learning, No-Fine-Tune with config file: {confpath}")
    print(f"~"*50)
    reproducibility.set_seed(11001000) # arbitrary, made it by combining 1-100-1000
    
    # Time params
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
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
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    weather = pd.read_excel(osp.join(DATA_DIR, "processed_data/dvdk_weather.xlsx"))
    fm1 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_1h.xlsx"))
    fm10 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_10h.xlsx"))
    fm100 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_100h.xlsx"))
    fm1000 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_1000h.xlsx"))

    # Set up output object
    xgb_results = {}
    lm_results = {}
    
    # FM1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f"~"*50)
    print(f"FM1 Train")

    # Combine weather and fm, fill na, add geographic features
    df1 = weather.merge(
        fm1[["utc_rounded", "utc_prov", "fm1"]],
        left_on="utc",
        right_on="utc_rounded",
        how="left"
    ).drop(columns="utc_rounded")
    df1["elev"] = conf.ok_elev
    df1["lon"] = conf.ok_lon
    df1["lat"] = conf.ok_lat

    # For training set, static models can only use the particular hours that the FMC exist, very sparse
    # For test set, we can still generate predictions for each hour and examine resulting time series
    df1_train = df1[(df1.utc >= conf.train_start) & (df1.utc <= conf.val_end)]
    df1_train = df1_train[df1_train.fm1.notna()]
    df1_test  = df1[(df1.utc >= conf.f_start) & (df1.utc <= conf.f_end)]
    print(f"    {df1_train.shape=}")
    print(f"    {df1_test.shape=}")
    X_train = df1_train[params_xgb.features_list]
    y_train = df1_train["fm1"].to_numpy()
    X_test = df1_test[params_xgb.features_list]
    y_test = df1_test["fm1"].to_numpy()
    
    # Fit XGB
    xgb_fm1 = XGB(params=params_xgb)
    xgb_fm1.fit(X_train, y_train)
    preds1 = xgb_fm1.predict(X_test)
    base_metrics = calc_metrics(y_test, preds1)
    lt30_metrics = calc_metrics(y_test, preds1, mask=(y_test <= 30))
    xgb_results["FM1"] = {
        "preds1": preds1,
        "base": base_metrics,
        "lt30": lt30_metrics,
    }
    

    # Fit LM
    lm_fm1 = LM(params=params_lm)
    lm_fm1.fit(X_train, y_train)    
    preds1 = lm_fm1.predict(X_test)
    base_metrics = calc_metrics(y_test, preds1)
    lt30_metrics = calc_metrics(y_test, preds1, mask=(y_test <= 30))
    lm_results["FM1"] = {
        "preds1": preds1,
        "base": base_metrics,
        "lt30": lt30_metrics,
    }

    # FM10
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f"~"*50)
    print(f"FM10 Train")

    # Combine weather and fm, fill na, add geographic features
    df10 = weather.merge(
        fm10[["utc_rounded", "utc_prov", "fm10"]],
        left_on="utc",
        right_on="utc_rounded",
        how="left"
    ).drop(columns="utc_rounded")
    df10["elev"] = conf.ok_elev
    df10["lon"] = conf.ok_lon
    df10["lat"] = conf.ok_lat

    # For training set, static models can only use the particular hours that the FMC exist, very sparse
    # For test set, we can still generate predictions for each hour and examine resulting time series
    df10_train = df10[(df10.utc >= conf.train_start) & (df10.utc <= conf.val_end)]
    df10_train = df10_train[df10_train.fm10.notna()]
    df10_test  = df10[(df10.utc >= conf.f_start) & (df10.utc <= conf.f_end)]
    print(f"    {df10_train.shape=}")
    print(f"    {df10_test.shape=}")
    X_train = df10_train[params_xgb.features_list]
    y_train = df10_train["fm10"].to_numpy()
    X_test = df10_test[params_xgb.features_list]
    y_test = df10_test["fm10"].to_numpy()
    
    # Fit XGB
    xgb_fm10 = XGB(params=params_xgb)
    xgb_fm10.fit(X_train, y_train)
    preds10 = xgb_fm10.predict(X_test)
    base_metrics = calc_metrics(y_test, preds10)
    lt30_metrics = calc_metrics(y_test, preds10, mask=(y_test <= 30))
    xgb_results["FM10"] = {
        "preds10": preds10,
        "base": base_metrics,
        "lt30": lt30_metrics,
    }

    # Fit LM
    lm_fm10 = LM(params=params_lm)
    lm_fm10.fit(X_train, y_train)    
    preds10 = lm_fm10.predict(X_test)
    base_metrics = calc_metrics(y_test, preds10)
    lt30_metrics = calc_metrics(y_test, preds10, mask=(y_test <= 30))
    lm_results["FM10"] = {
        "preds10": preds10,
        "base": base_metrics,
        "lt30": lt30_metrics,
    }
    
    
    # FM100
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f"~"*50)
    print(f"FM100 Train")
    
    # Combine weather and fm, fill na, add geographic features
    df100 = weather.merge(
        fm100[["utc_rounded", "utc_prov", "fm100"]],
        left_on="utc",
        right_on="utc_rounded",
        how="inner"
    ).drop(columns="utc_rounded")    
    df100["elev"] = conf.ok_elev
    df100["lon"] = conf.ok_lon
    df100["lat"] = conf.ok_lat
    
    # For training set, static models can only use the particular hours that the FMC exist, very sparse
    # For test set, we can still generate predictions for each hour and examine resulting time series
    df100_train = df100[(df100.utc >= conf.train_start) & (df100.utc <= conf.val_end)]
    df100_train = df100_train[df100_train.fm100.notna()]
    df100_test  = df100[(df100.utc >= conf.f_start) & (df100.utc <= conf.f_end)]
    print(f"    {df100_train.shape=}")
    print(f"    {df100_test.shape=}")
    X_train = df100_train[params_xgb.features_list]
    y_train = df100_train["fm100"].to_numpy()
    X_test = df100_test[params_xgb.features_list]
    y_test = df100_test["fm100"].to_numpy()
    
    # Fit XGB
    xgb_fm100 = XGB(params=params_xgb)
    xgb_fm100.fit(X_train, y_train)
    preds100 = xgb_fm100.predict(X_test)
    base_metrics = calc_metrics(y_test, preds100)
    xgb_results["FM100"] = {
        "preds100": preds100,
        "base": base_metrics
    }
    

    # Fit LM
    lm_fm100 = LM(params=params_lm)
    lm_fm100.fit(X_train, y_train)    
    preds100 = lm_fm100.predict(X_test)
    base_metrics = calc_metrics(y_test, preds100)
    lm_results["FM100"] = {
        "preds100": preds100,
        "base": base_metrics
    }   

    # FM1000
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f"~"*50)
    print(f"FM1000 Train")
    
    # Combine weather and fm, fill na, add geographic features
    df1000 = weather.merge(
        fm1000[["utc_rounded", "utc_prov", "fm1000"]],
        left_on="utc",
        right_on="utc_rounded",
        how="inner"
    ).drop(columns="utc_rounded")    
    df1000["elev"] = conf.ok_elev
    df1000["lon"] = conf.ok_lon
    df1000["lat"] = conf.ok_lat
    
    # For training set, static models can only use the particular hours that the FMC exist, very sparse
    # For test set, we can still generate predictions for each hour and examine resulting time series
    df1000_train = df1000[(df1000.utc >= conf.train_start) & (df1000.utc <= conf.val_end)]
    df1000_train = df1000_train[df1000_train.fm1000.notna()]
    df1000_test  = df1000[(df1000.utc >= conf.f_start) & (df1000.utc <= conf.f_end)]
    print(f"    {df1000_train.shape=}")
    print(f"    {df1000_test.shape=}")
    X_train = df1000_train[params_xgb.features_list]
    y_train = df1000_train["fm1000"].to_numpy()
    X_test = df1000_test[params_xgb.features_list]
    y_test = df1000_test["fm1000"].to_numpy()
    
    # Fit XGB
    xgb_fm1000 = XGB(params=params_xgb)
    xgb_fm1000.fit(X_train, y_train)
    preds1000 = xgb_fm1000.predict(X_test)
    base_metrics = calc_metrics(y_test, preds1000)
    xgb_results["FM1000"] = {
        "preds1000": preds1000,
        "base": base_metrics
    }
    

    # Fit LM
    lm_fm1000 = LM(params=params_lm)
    lm_fm1000.fit(X_train, y_train)    
    preds1000 = lm_fm1000.predict(X_test)
    base_metrics = calc_metrics(y_test, preds1000)
    lm_results["FM1000"] = {
        "preds1000": preds1000,
        "base": base_metrics
    }   

    
    # Output
    out_file = osp.join(output_dir, "results_xgb_testset.pkl")
    print(f"Writing Output to: {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(xgb_results, f, protocol=pickle.HIGHEST_PROTOCOL)

    out_file = osp.join(output_dir, "results_lm_testset.pkl")
    print(f"Writing Output to: {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(lm_results, f, protocol=pickle.HIGHEST_PROTOCOL)

