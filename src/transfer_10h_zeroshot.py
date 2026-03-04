# Executable module to run zeroshot FM10
# If run directly with config file, will default to rnn_dir in config
# If run with optional seed, will run with reps_dir and assume seed_i file exists 


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
from models.moisture_rnn import RNN_Flexible


# Executed Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        print(f"Invalid arguments. {len(sys.argv)-1} was given but 1 or 2 expected")
        print(f"Usage: {sys.argv[0]} <config_path> [seed]")
        print("Example: python src/transfer_10h_zeroshot.py etc/thesis_config.yaml")
        print("Example: python src/transfer_10h_zeroshot.py etc/thesis_config.yaml 17")
        sys.exit(-1)

    # Setup 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    confpath = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) == 3 else None
    conf = Dict(read_yml(confpath))
    
    print(f"~"*50)
    print(f"Running FM10 Zeroshot: {confpath}")

    if seed is not None:
        reproducibility.set_seed(seed)
        output_dir = osp.join(conf.output_dir, "zeroshot_10h_reps", f"seed_{seed}")
        print(f"RNN Model Dir: {osp.join(conf.reps_dir, f'seed_{seed}')}")
        params = Dict(read_yml(osp.join(conf.reps_dir, f"seed_{seed}", "params.yaml")))
        rnn = mrnn.RNN_Flexible(params=params)
        scaler = joblib.load(osp.join(conf.rnn_dir, "scaler.joblib"))
        rnn.load_weights(osp.join(conf.reps_dir, f"seed_{seed}", 'rnn.keras'))
    else:
        seed = 11001000 # arbitrary, made it by combining 1-100-1000
        reproducibility.set_seed(seed)
        output_dir = osp.join(conf.output_dir, "zeroshot_10h")
        print(f"RNN Model Dir: {conf.rnn_dir}")
        params = Dict(read_yml(osp.join(conf.rnn_dir, "params.yaml")))
        rnn = mrnn.RNN_Flexible(params=params)
        scaler = joblib.load(osp.join(conf.rnn_dir, "scaler.joblib"))
        rnn.load_weights(osp.join(conf.rnn_dir, 'rnn.keras'))


    os.makedirs(output_dir, exist_ok=True)

    # Time params
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print(f"Time Period: {conf.train_start} to {conf.f_end}")
    print("NOTE: no train nor validation set for this analysis, the whole period can be treated as a test set")

    # Data
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    weather = pd.read_excel(osp.join(DATA_DIR, "processed_data/dvdk_weather.xlsx"))
    fm10 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_10h.xlsx"))

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
    
    print(f"    {df10.shape=}")
    X = df10[params.features_list]
    y = df10["fm10"].to_numpy()

    # Scale by prefit scaler
    XX = scaler.transform(X)
    XX = XX.reshape(1, *XX.shape) 
    
    # Predict 
    preds = rnn.predict(XX).flatten()

    # Interp to exact time of observed data
    preds2 = time_intp(
        t1 = df10.utc.to_numpy(),
        v1 = preds,
        t2 = fm10.utc_prov.to_numpy()
    )

    # Calc accuracy in output object
    results = {}
    # Accuracy
    df = fm10.copy()
    df["preds"] = preds2

    results["preds"] = preds
    results["preds_intp"] = preds2
    results["rmse"] = np.sqrt(mean_squared_error(df.fm10, df.preds))
    results["bias"] = np.mean(df.fm10 - df.preds)
    results["r2"]   = r2_score(df.fm10, df.preds)

    # Accuracy <=30
    inds = np.where(df.fm10<=30)[0]
    results["rmse_30"] = np.sqrt(mean_squared_error(df.fm10.iloc[inds], df.preds.iloc[inds]))
    results["bias_30"] = np.mean(df.fm10.iloc[inds] - df.preds.iloc[inds])
    results["r2_30"]   =  r2_score(df.fm10.iloc[inds], df.preds.iloc[inds])
    
    
    print("FM10 Zeroshot Accuracy Metrics - Full Time Period")
    print(f"    N. Obs: {fm10.shape[0]}")
    print(f'    RMSE: {results["rmse"]}')

    print(f"    N. Obs Less than equal to 30: {np.sum(fm10.fm10 <= 30)}")
    print(f'    RMSE 30: {results["rmse_30"]}')    

    # Write Output
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Output
    out_file = osp.join(output_dir, f"results_zeroshot_{seed}.pkl")
    print(f"Writing Output to: {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
