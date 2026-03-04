# Executable module to run directly train RNN on Oklahoma field data
# No pretrain, no transfer

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
from sklearn.preprocessing import MinMaxScaler, StandardScaler

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
from models.moisture_rnn import RNN_Flexible, mse_masked, build_training_batches_univariate, scale_3d

# Metadata files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

params = Dict(read_yml(osp.join(PROJECT_ROOT, "models/params.yaml")))




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
    output_dir = osp.join(conf.output_dir, "notransfer_rnn")
    os.makedirs(output_dir, exist_ok=True)
    print(f"~"*50)
    print(f"Running Transfer-Learning, No-Fine-Tune with config file: {confpath}")
    print(f"~"*50)
    seed = 11001000 # arbitrary, made it by combining 1-100-1000
    reproducibility.set_seed(seed) 
    
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
    results = {}
    
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
    df1.loc[:, "hod"] = df1.hod_utc
    df1.loc[:, "doy"] = df1.doy_utc
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

    # Scale by fitting scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)    
    
    XX_val = scaler.transform(X_val)
    XX_val = XX_val.reshape(1, *XX_val.shape)
    yy_val = y_val[np.newaxis, :, np.newaxis]
    
    XX_test = scaler.transform(X_test)
    XX_test = XX_test.reshape(1, *XX_test.shape)
    
    # Build Samples
    X_train_samples, y_train_samples, masks = build_training_batches_univariate(X = X_train_scaled, y=y_train)
    print(f"    {X_train_samples.shape=}")
    
    # Build RNN and Train
    rnn = RNN_Flexible(params=params, loss=mse_masked, random_state=seed)
    rnn.fit(X_train_samples, y_train_samples, validation_data = (XX_val, yy_val), batch_size=params.batch_size, epochs=params.epochs, verbose_fit = True, plot_history=False)

    # Predict Test
    preds1 = rnn.predict(XX_test)
    
    # Interp to exact time of observed data
    inds= np.where(y_test != -9999)[0]
    preds2 = time_intp(
        t1 = df1_test.utc.to_numpy(),
        v1 = preds1.flatten(),
        t2 = df1_test.utc_prov.iloc[inds].to_numpy()
    )

    # Calc accuracy in output object
    results = {}
    results["FM1"] = {}
    # Accuracy
    results["FM1"]["rmse"] = np.sqrt(mean_squared_error(y_test[inds], preds2))
    results["FM1"]["bias"]        = np.mean(y_test[inds] - preds2)
    results["FM1"]["r2"]          = r2_score(y_test[inds], preds2)

    # Accuracy <=30
    y_test2 = y_test[inds]
    inds2 = np.where(y_test2<=30)[0]
    results["FM1"]["rmse_30"] = np.sqrt(mean_squared_error(y_test2[inds2], preds2[inds2]))
    results["FM1"]["bias_30"]        = np.mean(y_test2[inds2] - preds2[inds2])
    results["FM1"]["r2_30"]          = r2_score(y_test2[inds2], preds2[inds2])    
    results["FM1"]["preds1"] = preds1
    results["FM1"]["preds1_intp"] = preds2
    print("FM1 Accuracy Metrics Test Set")
    print(f'    RMSE: {results["FM1"]["rmse"]}')
    print(f'    RMSE 30: {results["FM1"]["rmse_30"]}')

    
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
    df10.loc[:, "hod"] = df10.hod_utc
    df10.loc[:, "doy"] = df10.doy_utc
    df10["elev"] = conf.ok_elev
    df10["lon"] = conf.ok_lon
    df10["lat"] = conf.ok_lat
    df10["fm10"] = df10["fm10"].fillna(-9999)
    
    df10_train = df10[(df10.utc >= conf.train_start) & (df10.utc <= conf.train_end)]
    df10_val   = df10[(df10.utc >= conf.val_start) & (df10.utc <= conf.val_end)]
    df10_test  = df10[(df10.utc >= conf.f_start) & (df10.utc <= conf.f_end)]
    print(f"    {df10_train.shape=}")
    print(f"    {df10_val.shape=}")
    print(f"    {df10_test.shape=}")
    X_train = df10_train[params.features_list]
    y_train = df10_train["fm10"].to_numpy()
    X_val = df10_val[params.features_list]
    y_val = df10_val["fm10"].to_numpy()
    X_test = df10_test[params.features_list]
    y_test = df10_test["fm10"].to_numpy()

    # Scale by fitting scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)    
    
    XX_val = scaler.transform(X_val)
    XX_val = XX_val.reshape(1, *XX_val.shape)
    yy_val = y_val[np.newaxis, :, np.newaxis]
    
    XX_test = scaler.transform(X_test)
    XX_test = XX_test.reshape(1, *XX_test.shape)
    
    # Build Samples
    X_train_samples, y_train_samples, masks = build_training_batches_univariate(X = X_train_scaled, y=y_train)
    print(f"    {X_train_samples.shape=}")
    
    # Build RNN and Train
    rnn = RNN_Flexible(params=params, loss=mse_masked, random_state=seed)
    rnn.fit(X_train_samples, y_train_samples, validation_data = (XX_val, yy_val), batch_size=params.batch_size, epochs=params.epochs, verbose_fit = True, plot_history=False)

    # Predict Test
    preds10 = rnn.predict(XX_test)
    
    # Interp to exact time of observed data
    inds= np.where(y_test != -9999)[0]
    preds2 = time_intp(
        t1 = df10_test.utc.to_numpy(),
        v1 = preds10.flatten(),
        t2 = df10_test.utc_prov.iloc[inds].to_numpy()
    )

    # Calc accuracy in output object
    results = {}
    results["FM10"] = {}
    # Accuracy
    results["FM10"]["rmse"] = np.sqrt(mean_squared_error(y_test[inds], preds2))
    results["FM10"]["bias"]        = np.mean(y_test[inds] - preds2)
    results["FM10"]["r2"]          = r2_score(y_test[inds], preds2)

    # Accuracy <=30
    y_test2 = y_test[inds]
    inds2 = np.where(y_test2<=30)[0]
    results["FM10"]["rmse_30"] = np.sqrt(mean_squared_error(y_test2[inds2], preds2[inds2]))
    results["FM10"]["bias_30"]        = np.mean(y_test2[inds2] - preds2[inds2])
    results["FM10"]["r2_30"]          = r2_score(y_test2[inds2], preds2[inds2])    
    results["FM10"]["preds10"] = preds10
    results["FM10"]["preds10_intp"] = preds2
    print("FM10 Accuracy Metrics Test Set")
    print(f'    RMSE: {results["FM10"]["rmse"]}')
    print(f'    RMSE 30: {results["FM10"]["rmse_30"]}')    



    # FM100
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f"~"*50)
    print(f"FM100 Train")

    # Combine weather and fm, fill na, add geographic features
    df100 = weather.merge(
        fm100[["utc_rounded", "utc_prov", "fm100"]],
        left_on="utc",
        right_on="utc_rounded",
        how="left"
    ).drop(columns="utc_rounded")
    df100.loc[:, "hod"] = df100.hod_utc
    df100.loc[:, "doy"] = df100.doy_utc
    df100["elev"] = conf.ok_elev
    df100["lon"] = conf.ok_lon
    df100["lat"] = conf.ok_lat
    df100["fm100"] = df100["fm100"].fillna(-9999)
    
    df100_train = df100[(df100.utc >= conf.train_start) & (df100.utc <= conf.train_end)]
    df100_val   = df100[(df100.utc >= conf.val_start) & (df100.utc <= conf.val_end)]
    df100_test  = df100[(df100.utc >= conf.f_start) & (df100.utc <= conf.f_end)]
    print(f"    {df100_train.shape=}")
    print(f"    {df100_val.shape=}")
    print(f"    {df100_test.shape=}")
    X_train = df100_train[params.features_list]
    y_train = df100_train["fm100"].to_numpy()
    X_val = df100_val[params.features_list]
    y_val = df100_val["fm100"].to_numpy()
    X_test = df100_test[params.features_list]
    y_test = df100_test["fm100"].to_numpy()

    # Scale by fitting scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)    
    
    XX_val = scaler.transform(X_val)
    XX_val = XX_val.reshape(1, *XX_val.shape)
    yy_val = y_val[np.newaxis, :, np.newaxis]
    
    XX_test = scaler.transform(X_test)
    XX_test = XX_test.reshape(1, *XX_test.shape)
    
    # Build Samples
    X_train_samples, y_train_samples, masks = build_training_batches_univariate(X = X_train_scaled, y=y_train)
    print(f"    {X_train_samples.shape=}")
    
    # Build RNN and Train
    rnn = RNN_Flexible(params=params, loss=mse_masked, random_state=seed)
    rnn.fit(X_train_samples, y_train_samples, validation_data = (XX_val, yy_val), batch_size=params.batch_size, epochs=params.epochs, verbose_fit = True, plot_history=False)

    # Predict Test
    preds100 = rnn.predict(XX_test)
    
    # Interp to exact time of observed data
    inds= np.where(y_test != -9999)[0]
    preds2 = time_intp(
        t1 = df100_test.utc.to_numpy(),
        v1 = preds100.flatten(),
        t2 = df100_test.utc_prov.iloc[inds].to_numpy()
    )

    # Calc accuracy in output object
    results = {}
    results["FM100"] = {}
    # Accuracy
    results["FM100"]["rmse"] = np.sqrt(mean_squared_error(y_test[inds], preds2))
    results["FM100"]["bias"]        = np.mean(y_test[inds] - preds2)
    results["FM100"]["r2"]          = r2_score(y_test[inds], preds2)

    print("FM100 Accuracy Metrics Test Set")
    print(f'    RMSE: {results["FM100"]["rmse"]}')


    # FM1000
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f"~"*50)
    print(f"FM1000 Train")

    # Combine weather and fm, fill na, add geographic features
    df1000 = weather.merge(
        fm1000[["utc_rounded", "utc_prov", "fm1000"]],
        left_on="utc",
        right_on="utc_rounded",
        how="left"
    ).drop(columns="utc_rounded")
    df1000.loc[:, "hod"] = df1000.hod_utc
    df1000.loc[:, "doy"] = df1000.doy_utc
    df1000["elev"] = conf.ok_elev
    df1000["lon"] = conf.ok_lon
    df1000["lat"] = conf.ok_lat
    df1000["fm1000"] = df1000["fm1000"].fillna(-9999)
    
    df1000_train = df1000[(df1000.utc >= conf.train_start) & (df1000.utc <= conf.train_end)]
    df1000_val   = df1000[(df1000.utc >= conf.val_start) & (df1000.utc <= conf.val_end)]
    df1000_test  = df1000[(df1000.utc >= conf.f_start) & (df1000.utc <= conf.f_end)]
    print(f"    {df1000_train.shape=}")
    print(f"    {df1000_val.shape=}")
    print(f"    {df1000_test.shape=}")
    X_train = df1000_train[params.features_list]
    y_train = df1000_train["fm1000"].to_numpy()
    X_val = df1000_val[params.features_list]
    y_val = df1000_val["fm1000"].to_numpy()
    X_test = df1000_test[params.features_list]
    y_test = df1000_test["fm1000"].to_numpy()

    # Scale by fitting scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)    
    
    XX_val = scaler.transform(X_val)
    XX_val = XX_val.reshape(1, *XX_val.shape)
    yy_val = y_val[np.newaxis, :, np.newaxis]
    
    XX_test = scaler.transform(X_test)
    XX_test = XX_test.reshape(1, *XX_test.shape)
    
    # Build Samples
    X_train_samples, y_train_samples, masks = build_training_batches_univariate(X = X_train_scaled, y=y_train)
    print(f"    {X_train_samples.shape=}")
    
    # Build RNN and Train
    rnn = RNN_Flexible(params=params, loss=mse_masked, random_state=seed)
    rnn.fit(X_train_samples, y_train_samples, validation_data = (XX_val, yy_val), batch_size=params.batch_size, epochs=params.epochs, verbose_fit = True, plot_history=False)

    # Predict Test
    preds1000 = rnn.predict(XX_test)
    
    # Interp to exact time of observed data
    inds= np.where(y_test != -9999)[0]
    preds2 = time_intp(
        t1 = df1000_test.utc.to_numpy(),
        v1 = preds1000.flatten(),
        t2 = df1000_test.utc_prov.iloc[inds].to_numpy()
    )

    # Calc accuracy in output object
    results = {}
    results["FM1000"] = {}
    # Accuracy
    results["FM1000"]["rmse"] = np.sqrt(mean_squared_error(y_test[inds], preds2))
    results["FM1000"]["bias"]        = np.mean(y_test[inds] - preds2)
    results["FM1000"]["r2"]          = r2_score(y_test[inds], preds2)

    print("FM1000 Accuracy Metrics Test Set")
    print(f'    RMSE: {results["FM1000"]["rmse"]}')
    
    

    # Write Output
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output
    out_file = osp.join(output_dir, "results_test_set.pkl")
    print(f"Writing Output to: {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
