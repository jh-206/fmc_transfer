# Executable module to run transfer analysis 
# All weights frozen - only time-warp param fit
# Methodology: for each fuel class construct grid of time warp params, modify LSTM and generate predictions
### for training+validation period. Pick best fitting accuracy. Generate predictions for test set and compute as final accuracy
# No fine-tuning, only the time-warp param is fit
# This method doesn't utilize a train/validation set split, so combining them together
# Fine-tuning requires the validation set split

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
from models.moisture_rnn import RNN_Flexible, mse_masked, build_training_batches_univariate
import reproducibility

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
    output_dir = osp.join(conf.output_dir, "transfer_finetune")
    os.makedirs(output_dir, exist_ok=True)
    print(f"~"*50)
    print(f"Running Transfer-Learning, Full Fine-Tune with config file: {confpath}")
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

    # RNN
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    params = Dict(read_yml(osp.join(conf.rnn_dir, "params.yaml")))
    seed = 42
    reproducibility.set_seed(seed)
    rnn = RNN_Flexible(params=params, loss=mse_masked, random_state=seed)
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
    df1["fm1"] = df1["fm1"].fillna(-9999)
    
    df1_train = df1[(df1.utc >= conf.train_start) & (df1.utc <= conf.train_end)]
    df1_val   = df1[(df1.utc >= conf.val_start) & (df1.utc <= conf.val_end)]
    df1_test  = df1[(df1.utc >= conf.f_start) & (df1.utc <= conf.f_end)]
    print(f"    {df1_train.shape=}")
    print(f"    {df1_val.shape=}")
    print(f"    {df1_test.shape=}")
    X_train = df1_train[params.features_list]
    y_train = df1_train["fm1"].to_numpy()
    X_val = df1_val[params.features_list]
    y_val = df1_val["fm1"].to_numpy()
    X_test = df1_test[params.features_list]
    y_test = df1_test["fm1"].to_numpy()

    # Scale using saved scaler object from RNN, reshape val and test to 3d array
    X_train_scaled = scaler.transform(X_train)
    
    XX_val = scaler.transform(X_val)
    XX_val = XX_val.reshape(1, *XX_val.shape)
    yy_val = y_val[np.newaxis, :, np.newaxis]
    
    XX_test = scaler.transform(X_test)
    XX_test = XX_test.reshape(1, *XX_test.shape)

    # Build training samples
    X_train_samples, y_train_samples, masks = build_training_batches_univariate(X = X_train_scaled, y=y_train)
    print(f"    {X_train_samples.shape=}")

    bf_grid = np.linspace(conf.fm1_bf_low, conf.fm1_bf_high, num=conf.ngrid).round(4)
    bi_grid = np.linspace(conf.fm1_bi_low, conf.fm1_bi_high, num=conf.ngrid).round(4)
    fm1_grid = make_param_grid(bf=bf_grid, bi=bi_grid)
    print()
    print(f"N time-warp Param Combos: {len(fm1_grid)}")

    #d Loop over param grids, fit to training data w early stop, calculate RMSE on val
    results_1 = {}
    for i, bs in enumerate(fm1_grid):
        print("~"*50)
        print(f"FM1 Param Combo {i+1} out of {len(fm1_grid)}")
        print(f"Params: {bs}")    
        weightsi = warp_weights(weights10, bi_warp = bs["bi"], bf_warp = bs["bf"])
        rnn.get_layer("lstm").set_weights(weightsi)
        rnn.fit(X_train_samples, y_train_samples, validation_data = (XX_val, yy_val), verbose_fit = False, plot_history=False)
        
        print(f"Predicting Val Set")
        preds = rnn.predict(XX_val)
        val_mse = np.float32(mse_masked(y_val.flatten(), preds.flatten()))
        print(f"    {val_mse=}")
        results_1[i]={}
        results_1[i]["val_rmse"] = np.sqrt(val_mse)
        results_1[i]["params"] = bs
        results_1[i]["preds"] = preds  


    # Find min val_rmse case
    fm1_best_key = min(results_1, key=lambda ci: results_1[ci]["val_rmse"])
    fm1_best = results_1[fm1_best_key]
    print()
    print("Best Config from Val Error:")
    print(f"Min Val RMSE: {np.round(fm1_best['val_rmse'], 4)}")
    print(f"Time-Warp Params: {fm1_best['params']}")

    breakpoint()
    # Compute Test Error and save, using params
    weights1 = warp_weights(weights10, bi_warp = fm1_best["params"]["bi"], bf_warp = fm1_best["params"]["bf"])
    rnn.get_layer("lstm").set_weights(weights1)
    preds1 = rnn.predict(XX_test)

    # Interp to exact time of observed data
    inds= np.where(y_test != -9999)
    preds2 = time_intp(
        t1 = df1_test.utc.to_numpy(),
        v1 = preds1.flatten(),
        t2 = df1_test.utc_prov.iloc[inds].to_numpy()
    )

    # Accuracy
    fm1_best["test_rmse"] = np.sqrt(mean_squared_error(y_test[inds], preds2))
    fm1_best["test_bias"]        = np.mean(y_test[inds] - preds2)
    fm1_best["test_r2"]          = r2_score(y_test[inds], preds2)

    # Accuracy <=30
    y_test2 = y_test[inds]
    inds2 = np.where(y_test2<=30)[0]
    fm1_best["test_rmse_30"] = np.sqrt(mean_squared_error(y_test2[inds2], preds2[inds2]))
    fm1_best["test_bias_30"]        = np.mean(y_test2[inds2] - preds2[inds2])
    fm1_best["test_r2_30"]          = r2_score(y_test2[inds2], preds2[inds2])    
    
    breakpoint()
    

















