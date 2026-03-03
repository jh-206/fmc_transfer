# Executable module to run steady-state scenarios on RNN Model replications

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
import joblib

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
from models.moisture_rnn import RNN_Flexible
import reproducibility

# Module Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_const_mean_case(scaler, nsteps=168):
    """
    Given scaler, produce formatted data of constant mean zero case.

    Returns untransformed and zero-transformed versions, 
    in case you want to examine the original means
    """
    x0 = np.repeat(scaler.mean_[None, :], nsteps, axis=0)
    x0_scaled = scaler.transform(x0.copy())
    x0_scaled = x0_scaled[None, :, :] # make 3d array for RNN
    return x0, x0_scaled

def make_sine_eq_case(scaler, nsteps=168):
    mu = scaler.mean_
    var = scaler.var_
    t = np.linspace(0, 2*np.pi, nsteps, endpoint=False)
    
    # start from constant mean sequence
    xs = np.repeat(mu[None, :], nsteps, axis=0)
    
    # amplitudes for first two features
    A = np.sqrt(2 * var[:2])
    
    # apply sine only to first two features
    xs[:, :2] = mu[:2] + A[None, :] * np.sin(t)[:, None]

    xs_scaled = scaler.transform(xs.copy())
    xs_scaled = xs_scaled[None, :, :] # make 3d array for RNN
    return xs, xs_scaled

def make_raws_case(scaler, dat, nsteps=168):
    x1 = dat1[params["features_list"]].to_numpy(dtype=np.float32)
    x1 = np.repeat(x1[None, :], nsteps, axis=0)
    x1_scaled = scaler.transform(x1.copy())
    x1_scaled = x1_scaled[None, :, :] # make 3d array for RNN
    return x1, x1_scaled


# Executing Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Hard coded run params

nsteps=168 
st1 = "AENC2"
st2 = "BAWC2"
bf_warp = 0.5
bi_warp = 0.5


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 2 expected")
        print(f"Usage: {sys.argv[0]} <config_path>")
        print("Example: python src/steady_state_reps.py etc/thesis_config.yaml")
        sys.exit(-1)  

    # Setup 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    confpath = sys.argv[1]
    conf = Dict(read_yml(confpath))
    reps_dir = conf.reps_dir
    output_dir = osp.join(conf.output_dir, "steady_reps")
    os.makedirs(output_dir, exist_ok=True)
    params = Dict(read_yml(osp.join(conf.rnn_dir, "params.yaml")))
    print(f"~"*50)
    print(f"Running Steady-State Analysis on RNN Replications with config: {confpath}")
    print(f"~"*50)
    reproducibility.set_seed(11001000) # arbitrary, made it by combining 1-100-1000
    
    # Read In Reps
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    subdirs = os.listdir(reps_dir)
    subdirs = sorted(
        (d for d in os.listdir(reps_dir)
         if os.path.isdir(os.path.join(reps_dir, d)) and d.startswith("seed_")),
        key=lambda x: int(x.split("_")[1])
    )

    # Make RNN Object - modify weights from stable architecture
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    rnn = RNN_Flexible(params=params)
    units = rnn.get_layer("lstm").units

    # Read in ML data to get RAWS cases
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ml_data = pd.read_pickle(osp.join(conf.rnn_dir, "ml_data.pkl"))
    dat1 = ml_data[st1]["data"].iloc[-1] # Last available time
    dat2 = ml_data[st2]["data"].iloc[12345] 

    # Loop Over Reps - make cases with scaler, generate predictions for time-warp scenarios
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    results = {} # return object
    for subdir in subdirs:
        
        # Setup rep case - output and paths to model objects
        results[subdir] = {}
        weight_path = osp.join(reps_dir, subdir, "rnn.weights.h5")
        scaler = joblib.load(osp.join(reps_dir, subdir, "scaler.joblib")) 

        # Make data cases
        x0, x0_scaled = make_const_mean_case(scaler)
        xs, xs_scaled = make_sine_eq_case(scaler)
        x1, x1_scaled = make_raws_case(scaler, dat1)
        x2, x2_scaled = make_raws_case(scaler, dat2)
        
        # Set RNN weights based on rep, copy to make stable
        rnn.load_weights(weight_path)
        weights0 = [w.copy() for w in rnn.get_layer("lstm").get_weights()]
        # Predictions for baseline case
        p0 = rnn.predict(x0_scaled, verbose=0).flatten()
        ps = rnn.predict(xs_scaled, verbose=0).flatten()
        p1 = rnn.predict(x1_scaled, verbose=0).flatten()
        p2 = rnn.predict(x2_scaled, verbose=0).flatten()
        results[subdir]["base"] = {
            'p0': p0,
            'ps': ps,
            'p1': p1,
            'p2': p2            
        }

        
        # Set up time-warp cases
        weights_fast = rnn.get_layer("lstm").get_weights()
        weights_fast[2][0:units]       = weights0[2][0:units] + bi_warp       # input gate
        weights_fast[2][units:2*units] = weights0[2][units:2*units] - bf_warp # forget gate
        rnn.get_layer("lstm").set_weights(weights_fast)
        # Predictions for fast case
        p0 = rnn.predict(x0_scaled, verbose=0).flatten()
        ps = rnn.predict(xs_scaled, verbose=0).flatten()
        p1 = rnn.predict(x1_scaled, verbose=0).flatten()
        p2 = rnn.predict(x2_scaled, verbose=0).flatten()
        results[subdir]["fast"] = {
            'p0': p0,
            'ps': ps,
            'p1': p1,
            'p2': p2            
        }        
        
        weights_slow = rnn.get_layer("lstm").get_weights()
        weights_slow[2][0:units]       = weights0[2][0:units] - bi_warp       # input gate
        weights_slow[2][units:2*units] = weights0[2][units:2*units] + bf_warp # forget gate
        rnn.get_layer("lstm").set_weights(weights_slow)
        # Predictions for slow case
        p0 = rnn.predict(x0_scaled, verbose=0).flatten()
        ps = rnn.predict(xs_scaled, verbose=0).flatten()
        p1 = rnn.predict(x1_scaled, verbose=0).flatten()
        p2 = rnn.predict(x2_scaled, verbose=0).flatten()
        results[subdir]["slow"] = {
            'p0': p0,
            'ps': ps,
            'p1': p1,
            'p2': p2            
        }


    # Write Output
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    out_file = osp.join(output_dir, "results_reps.pkl")
    print(f"Writing Output to: {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)    




    



