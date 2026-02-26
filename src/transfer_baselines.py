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

    breakpoint()
    ## TODO: generate full predictions in test set for each hour to observe timeseries
    
    df1_train = df1[(df1.utc >= conf.train_start) & (df1.utc <= conf.val_end)]
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

    # Predict test and save for XGB
    preds1 = xgb_fm1.predict(X_test)
    inds = np.where(y_test < 30)[0]
    xgb_results["FM1"] = {
        'preds1': preds1,
        'rmse': np.sqrt(mean_squared_error(y_test, preds1)),
        'bias': np.mean(y_test - preds1),
        'r2': r2_score(y_test, preds1),
        'rmse_30': np.sqrt(mean_squared_error(y_test[inds], preds1[inds])),
        'bias_30': np.mean(y_test[inds] - preds1[inds]),
        'r2_30': r2_score(y_test[inds], preds1[inds])        
    }

    # Fit LM
    lm_fm1 = LM(params=params_lm)
    lm_fm1.fit(X_train, y_train)    

    # Predict test and save for LM
    preds1 = lm_fm1.predict(X_test)
    inds = np.where(y_test < 30)[0]
    lm_results["FM1"] = {
        'preds1': preds1,
        'rmse': np.sqrt(mean_squared_error(y_test, preds1)),
        'bias': np.mean(y_test - preds1),
        'r2': r2_score(y_test, preds1),
        'rmse_30': np.sqrt(mean_squared_error(y_test[inds], preds1[inds])),
        'bias_30': np.mean(y_test[inds] - preds1[inds]),
        'r2_30': r2_score(y_test[inds], preds1[inds])        
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
    
    df100_train = df100[(df100.utc >= conf.train_start) & (df100.utc <= conf.val_end)]
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

    # Predict test
    preds100 = xgb_fm100.predict(X_test)
    xgb_results["FM100"] = {
        'preds100': preds100,
        'rmse': np.sqrt(mean_squared_error(y_test, preds100)),
        'bias': np.mean(y_test - preds100),
        'r2': r2_score(y_test, preds100)    
    }

    # Fit LM
    lm_fm100 = LM(params=params_lm)
    lm_fm100.fit(X_train, y_train)    

    # Predict test and save for LM
    preds100 = lm_fm100.predict(X_test)
    lm_results["FM100"] = {
        'preds100': preds100,
        'rmse': np.sqrt(mean_squared_error(y_test, preds100)),
        'bias': np.mean(y_test - preds100),
        'r2': r2_score(y_test, preds100)      
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
    
    df1000_train = df1000[(df1000.utc >= conf.train_start) & (df1000.utc <= conf.val_end)]
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

    # Predict test
    preds1000 = xgb_fm1000.predict(X_test)
    xgb_results["FM1000"] = {
        'preds1000': preds1000,
        'rmse': np.sqrt(mean_squared_error(y_test, preds1000)),
        'bias': np.mean(y_test - preds1000),
        'r2': r2_score(y_test, preds1000)    
    }

    # Fit LM
    lm_fm1000 = LM(params=params_lm)
    lm_fm1000.fit(X_train, y_train)    

    # Predict test and save for LM
    preds1000 = lm_fm1000.predict(X_test)
    lm_results["FM1000"] = {
        'preds1000': preds1000,
        'rmse': np.sqrt(mean_squared_error(y_test, preds1000)),
        'bias': np.mean(y_test - preds1000),
        'r2': r2_score(y_test, preds1000)      
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

