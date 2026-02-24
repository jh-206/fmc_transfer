# Executable module to run transfer analysis 
# All weights frozen - only time-warp param fit
# Methodology: for each fuel class construct grid of time warp params, modify LSTM and generate predictions
### for training+validation period. Pick best fitting accuracy. Generate predictions for test set and compute as final accuracy
# No fine-tuning, only the time-warp param is fit
# This method doesn't utilize a train/validation set split, so combining them together
# Fine-tuning requires the validation set split

import numpy as np
import requests as requests
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

# Metadata files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Module Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_param_grid(**grids):
    """
    Creates all combos of input param grids. Input are lists of grids of any shape, put them in with named arg, name determined by user

    Usage: make_param_grid(bi=bi_grid, bf=bf_grid)
    """
    keys = grids.keys()
    values = grids.values()
    
    return [
        dict(zip(keys, combo))
        for combo in product(*values)
    ]

def warp_weights(weights0, bi_warp, bf_warp):
    """
    Given LSTM layer weights and time-warp parameters, return a new list
    of time-warped LSTM weights without modifying the input weights.
    """
    # Copy all arrays to avoid mutating the originals
    w_warped = [w.copy() for w in weights0]
    # Bias vector (Keras LSTM layout: [i, f, c, o])
    b = w_warped[2]
    # Infer number of LSTM units from bias length
    if b.ndim != 1 or b.shape[0] % 4 != 0:
        raise ValueError("Unexpected LSTM bias shape.")
    lstm_units = b.shape[0] // 4
    # Input gate biases (i)
    b[0:lstm_units] += bi_warp
    # Forget gate biases (f)
    b[lstm_units:2 * lstm_units] += bf_warp

    return w_warped

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
    output_dir = conf.output_dir
    os.makedirs(output_dir, exist_ok=True)
    print(f"~"*50)
    print(f"Running Transfer-Learning, No-Fine-Tune with config file: {confpath}")
    print(f"~"*50)
    
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

    # RNN
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    params = read_yml(osp.join(conf.rnn_dir, "params.yaml"))
    rnn = mrnn.RNN_Flexible(params=params)
    scaler = joblib.load(osp.join(conf.rnn_dir, "scaler.joblib"))
    rnn.load_weights(osp.join(conf.rnn_dir, 'rnn.keras'))
    ## Extract LSTM weights
    lstm = rnn.get_layer("lstm")
    lstm_units = lstm.units
    weights10 = lstm.get_weights()    

    # Data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    weather = pd.read_excel(osp.join(DATA_DIR, "processed_data/dvdk_weather.xlsx"))
    fm1 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_1h.xlsx"))
    fm10 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_10h.xlsx"))
    fm100 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_100h.xlsx"))
    fm1000 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_1000h.xlsx"))
    
    # Build Data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Geographic Variables from Slapout station
    # NOTE: combining train and val periods, this method doesn't need train/val split
    wtrain = weather[(weather.utc >= conf.train_start) & (weather.utc <= conf.val_end)]
    wtest  = weather[(weather.utc >= conf.f_start) & (weather.utc <= conf.f_end)]
    
    X_train = pd.DataFrame({
        "Ed": wtrain.Ed,
        "Ew": wtrain.Ew,
        "solar": wtrain["solar"],
        "wind": wtrain["wind"],
        "elev": conf.ok_elev,
        "lon": conf.ok_lon,
        "lat": conf.ok_lat,
        "rain": wtrain["rain"],
        "hod": wtrain.hod_utc,
        "doy": wtrain.doy_utc
    })
    X_test = pd.DataFrame({
        "Ed": wtest.Ed,
        "Ew": wtest.Ew,
        "solar": wtest["solar"],
        "wind": wtest["wind"],
        "elev": conf.ok_elev,
        "lon": conf.ok_lon,
        "lat": conf.ok_lat,
        "rain": wtest["rain"],
        "hod": wtest.hod_utc,
        "doy": wtest.doy_utc
    })

    
    assert X_train.columns.equals(pd.Index(params['features_list'])), f"Features list doesn't match built data columns, {params['features_list']=}, \n {X_train.columns}"
    assert X_train.columns.equals(X_test.columns), f"Train and Test columns don't match, {X_train.columns=}, \n, {X_test.columns}"

    
    # Scale using saved scaler object from RNN, reshape to 3d array
    XX_train = scaler.transform(X_train)
    XX_train = XX_train.reshape(1, *XX_train.shape)

    XX_test = scaler.transform(X_test)
    XX_test = XX_test.reshape(1, *XX_test.shape)

    print(f"Training Data Shape: {XX_train.shape}")
    print(f"Test Data Shape: {XX_test.shape}")

    # FM1
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f"~"*50)
    print(f"FM1 Train")
    
    fm1_train = fm1[(fm1.utc_rounded >= conf.train_start) & (fm1.utc_rounded <= conf.val_end)]
    fm1_test  = fm1[(fm1.utc_rounded >= conf.f_start) & (fm1.utc_rounded <= conf.f_end)]
    print(f"    {fm1_train.shape=}")
    print(f"    {fm1_test.shape=}")
    
    bf_grid = np.linspace(conf.fm1_bf_low, conf.fm1_bf_high, num=conf.ngrid)
    bi_grid = np.linspace(conf.fm1_bi_low, conf.fm1_bi_high, num=conf.ngrid)
    fm1_grid = make_param_grid(bf=bf_grid, bi=bi_grid)
    print()
    print(f"N time-warp Param Combos: {len(fm1_grid)}")

    ## Set up output object, loop over configs and run
    results_1 = {}
    for i, bs in enumerate(fm1_grid):
        print("~"*50)
        print(f"Param Combo {i+1} out of {len(fm1_grid)}")
        print(f"Params: {bs}")

        print(f"Warping LSTM params")
        weightsi = warp_weights(weights10, bi_warp = bs["bi"], bf_warp = bs["bf"])
        rnn.get_layer("lstm").set_weights(weightsi)

        print(f"Training Set Predictions")
        preds = rnn.predict(XX_train).flatten()
        
        # Interp to exact time of observed data
        preds2 = time_intp(
            t1 = wtrain.utc.to_numpy(),
            v1 = preds,
            t2 = fm1_train.utc_prov.to_numpy()
        )

        # Accuracy
        df = fm1_train.copy()
        df["preds"] = preds2
        rmse = np.sqrt(mean_squared_error(df.fm1, df.preds))
        bias = np.mean(df.fm1 - df.preds)
        r2   = r2_score(df.fm1, df.preds)
        
        # Save to results
        results_1[i] = {}
        results_1[i]["params"] = bs
        results_1[i]["preds"] = preds
        results_1[i]["preds_intp"] = preds2
        results_1[i]["rmse"] = rmse
        results_1[i]["bias"] = bias
        results_1[i]["r2"] = r2

        # Accuracy <30
        inds = np.where(df.fm1<30)[0]
        rmse_30 = np.sqrt(mean_squared_error(df.fm1.iloc[inds], df.preds.iloc[inds]))
        bias_30 = np.mean(df.fm1.iloc[inds] - df.preds.iloc[inds])
        r2_30   = r2_score(df.fm1.iloc[inds], df.preds.iloc[inds])

        results_1[i]["rmse_30"] = rmse_30
        results_1[i]["bias_30"] = bias_30
        results_1[i]["r2_30"] = r2_30

        print(f"Accuracy Metrics:")
        print(f"RMSE: {rmse.round(4)},   R2: {np.round(r2, 4)}")
        print(f"RMSE (FM1<30): {rmse_30.round(4)},   R2 (FM1<30): {np.round(r2_30, 4)}")

        
    # Output
    out_file = osp.join(output_dir, "fm1_results.pkl")
    print(f"Writing Output to: {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(results_1, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Find min RMSE config, basing on RMSE less than 30 for 1h alone
    fm1_best_key = min(results_1, key=lambda ci: results_1[ci]["rmse_30"])
    fm1_best = results_1[fm1_best_key]
    print()
    print("Best Config from Training Error:")
    print(f"Min Training RMSE (FM1<30): {np.round(fm1_best['rmse_30'], 4)}")
    print(f"Time-Warp Params: {fm1_best['params']}")
    
    # FM100
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f"~"*50)
    print(f"FM100 Train")
    
    fm100_train = fm100[(fm100.utc_rounded >= conf.train_start) & (fm100.utc_rounded <= conf.val_end)]
    fm100_test  = fm100[(fm100.utc_rounded >= conf.f_start) & (fm100.utc_rounded <= conf.f_end)]
    print(f"    {fm100_train.shape=}")
    print(f"    {fm100_test.shape=}")
    
    bf_grid = np.linspace(conf.fm100_bf_low, conf.fm100_bf_high, num=conf.ngrid)
    bi_grid = np.linspace(conf.fm100_bi_low, conf.fm100_bi_high, num=conf.ngrid)
    fm100_grid = make_param_grid(bf=bf_grid, bi=bi_grid)
    print()
    print(f"N time-warp Param Combos: {len(fm100_grid)}")

    ## Set up output object, loop over configs and run
    results_100 = {}
    for i, bs in enumerate(fm100_grid):
        print("~"*50)
        print(f"Param Combo {i+1} out of {len(fm100_grid)}")
        print(f"Params: {bs}")

        print(f"Warping LSTM params")
        weightsi = warp_weights(weights10, bi_warp = bs["bi"], bf_warp = bs["bf"])
        rnn.get_layer("lstm").set_weights(weightsi)

        print(f"Training Set Predictions")
        preds = rnn.predict(XX_train).flatten()
        
        # Interp to exact time of observed data
        preds2 = time_intp(
            t1 = wtrain.utc.to_numpy(),
            v1 = preds,
            t2 = fm100_train.utc_prov.to_numpy()
        )

        # Accuracy
        df = fm100_train.copy()
        df["preds"] = preds2
        rmse = np.sqrt(mean_squared_error(df.fm100, df.preds))
        bias = np.mean(df.fm100 - df.preds)
        r2   = r2_score(df.fm100, df.preds)
        
        # Save to results
        results_100[i] = {}
        results_100[i]["params"] = bs
        results_100[i]["preds"] = preds
        results_100[i]["preds_intp"] = preds2
        results_100[i]["rmse"] = rmse
        results_100[i]["bias"] = bias
        results_100[i]["r2"] = r2

        print(f"Accuracy Metrics:")
        print(f"RMSE: {rmse.round(4)},   R2: {np.round(r2, 4)}")
        
    # Output
    out_file = osp.join(output_dir, "fm100_results.pkl")
    print(f"Writing Output to: {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(results_100, f, protocol=pickle.HIGHEST_PROTOCOL)
        
    # Find min RMSE config
    fm100_best_key = min(results_100, key=lambda ci: results_100[ci]["rmse"])
    fm100_best = results_100[fm100_best_key]
    print()
    print("Best Config from Training Error:")
    print(f"Min Training RMSE: {np.round(fm100_best['rmse'], 4)}")
    print(f"Time-Warp Params: {fm100_best['params']}")
    

    # FM1000
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f"~"*50)
    print(f"FM1000 Train")
    
    fm1000_train = fm1000[(fm1000.utc_rounded >= conf.train_start) & (fm1000.utc_rounded <= conf.val_end)]
    fm1000_test  = fm1000[(fm1000.utc_rounded >= conf.f_start) & (fm1000.utc_rounded <= conf.f_end)]
    print(f"    {fm1000_train.shape=}")
    print(f"    {fm1000_test.shape=}")
    
    bf_grid = np.linspace(conf.fm1000_bf_low, conf.fm1000_bf_high, num=conf.ngrid)
    bi_grid = np.linspace(conf.fm1000_bi_low, conf.fm1000_bi_high, num=conf.ngrid)
    fm1000_grid = make_param_grid(bf=bf_grid, bi=bi_grid)
    print()
    print(f"N time-warp Param Combos: {len(fm1000_grid)}")

    ## Set up output object, loop over configs and run
    results_1000 = {}
    for i, bs in enumerate(fm1000_grid):
        print("~"*50)
        print(f"Param Combo {i+1} out of {len(fm1000_grid)}")
        print(f"Params: {bs}")

        print(f"Warping LSTM params")
        weightsi = warp_weights(weights10, bi_warp = bs["bi"], bf_warp = bs["bf"])
        rnn.get_layer("lstm").set_weights(weightsi)

        print(f"Training Set Predictions")
        preds = rnn.predict(XX_train).flatten()
        
        # Interp to exact time of observed data
        preds2 = time_intp(
            t1 = wtrain.utc.to_numpy(),
            v1 = preds,
            t2 = fm1000_train.utc_prov.to_numpy()
        )

        # Accuracy
        df = fm1000_train.copy()
        df["preds"] = preds2
        rmse = np.sqrt(mean_squared_error(df.fm1000, df.preds))
        bias = np.mean(df.fm1000 - df.preds)
        r2   = r2_score(df.fm1000, df.preds)
        
        # Save to results
        results_1000[i] = {}
        results_1000[i]["params"] = bs
        results_1000[i]["preds"] = preds
        results_1000[i]["preds_intp"] = preds2
        results_1000[i]["rmse"] = rmse
        results_1000[i]["bias"] = bias
        results_1000[i]["r2"] = r2

        print(f"Accuracy Metrics:")
        print(f"RMSE: {rmse.round(4)},   R2: {np.round(r2, 4)}")
        
    # Output
    out_file = osp.join(output_dir, "fm1000_results.pkl")
    print(f"Writing Output to: {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(results_1000, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Find min RMSE config
    fm1000_best_key = min(results_1000, key=lambda ci: results_1000[ci]["rmse"])
    fm1000_best = results_1000[fm1000_best_key]
    print()
    print("Best Config from Training Error:")
    print(f"Min Training RMSE: {np.round(fm1000_best['rmse'], 4)}")
    print(f"Time-Warp Params: {fm1000_best['params']}")


    # Run Test Set
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    print()
    print("~"*50)
    print("Running Test Set with Best Config")

    # Output object
    results_test = {}

    # 1hr
    bs = fm1_best['params']
    weights1 = warp_weights(weights10, bi_warp = bs["bi"], bf_warp = bs["bf"])
    rnn.get_layer("lstm").set_weights(weights1)
    preds1 = rnn.predict(XX_test).flatten()
    # Accuracy
    df = fm1_test.copy()
    df["preds"] = preds1
    rmse = np.sqrt(mean_squared_error(df.fm1, df.preds))
    bias = np.mean(df.fm1 - df.preds)
    r2   = r2_score(df.fm1, df.preds)  
    # Accuracy <30
    inds = np.where(df.fm1<30)[0]
    rmse_30 = np.sqrt(mean_squared_error(df.fm1.iloc[inds], df.preds.iloc[inds]))
    bias_30 = np.mean(df.fm1.iloc[inds] - df.preds.iloc[inds])
    r2_30   = r2_score(df.fm1.iloc[inds], df.preds.iloc[inds])
    results_test["FM1"] = {
        'rmse': rmse,
        'bias': bias,
        'r2': r2,
        'rmse_30': rmse_30,
        'bias_30': bias_30,
        'r2_30': r2_30        
    }

    
    
    # 100hr
    bs = fm100_best['params']
    weights100 = warp_weights(weights10, bi_warp = bs["bi"], bf_warp = bs["bf"])
    rnn.get_layer("lstm").set_weights(weights100)
    preds100 = rnn.predict(XX_test).flatten()

    # Accuracy
    df = fm100_test.copy()
    df["preds"] = preds100
    rmse = np.sqrt(mean_squared_error(df.fm100, df.preds))
    bias = np.mean(df.fm100 - df.preds)
    r2   = r2_score(df.fm100, df.preds)  
    results_test["FM100"] = {
        'rmse': rmse,
        'bias': bias,
        'r2': r2    
    }

    
    # 1000hr
    bs = fm1000_best['params']
    weights1000 = warp_weights(weights10, bi_warp = bs["bi"], bf_warp = bs["bf"])
    rnn.get_layer("lstm").set_weights(weights1000)
    preds1000 = rnn.predict(XX_test).flatten()

    # Accuracy
    df = fm1000_test.copy()
    df["preds"] = preds1000
    rmse = np.sqrt(mean_squared_error(df.fm1000, df.preds))
    bias = np.mean(df.fm1000 - df.preds)
    r2   = r2_score(df.fm1000, df.preds)  
    results_test["FM1000"] = {
        'rmse': rmse,
        'bias': bias,
        'r2': r2    
    }

    # Output
    out_file = osp.join(output_dir, "results_test_set.pkl")
    print(f"Writing Output to: {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(results_test, f, protocol=pickle.HIGHEST_PROTOCOL)

        


