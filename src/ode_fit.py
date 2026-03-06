# Executable module used to fit ODE+KF curve to entire Carlson dataset for 
# each fuel class. Grid search on params. Just used as a reanalysis, not forecast

import numpy as np
import pandas as pd
import yaml
import time
import sys
import os
import os.path as osp
from itertools import product
import pickle

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
from models.moisture_ode import ODE_FMC, model_augmented, ext_kf

# Metadata files
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

params = Dict(read_yml(osp.join(CONFIG_DIR, "params_models.yaml"), subkey="ode"))

# Module Functions
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

def make_param_grid(**grids):
    keys = grids.keys()
    values = grids.values()
    
    return [
        dict(zip(keys, combo))
        for combo in product(*values)
    ]

## Wrapper to run ODE+KF with intermittent observations
def ode_kf_fit(df, params, tstep = 1, fm_col = "fm1"):
    # Get data from df
    Ed = df.Ed
    Ew = df.Ew
    rain = df.rain
    d = df[fm_col]
    # times = df.date
    hours = df.shape[0]
    
    # Background State and Covariance Matrices
    sigmaR = params["R"]
    sigmdaQ = params["Q"]
    T = params["T"] # timelag
    
    # Initialize
    u = np.zeros((2,hours))
    u[:,0]=[d[0],0]       # initialize,background state 
    P = np.zeros((2,2,hours))
    P[:,:,0] = np.array([[1e-3, 0.],
                      [0.,  1e-3]]) # background state covariance
    Q = np.array([[sigmdaQ, 0.],
                [0,  sigmdaQ]]) # process noise covariance
    H = np.array([[1., 0.]])  # first component observed
    R = np.array([sigmaR]) # data variance

    # Run
    for t in range(1, hours):
        update_phase = ~np.isnan(d[t])
        if update_phase:
            u[:,t],P[:,:,t] = ext_kf(u[:,t-1],P[:,:,t-1],
                                      lambda uu: model_augmented(uu,Ed[t],Ew[t],rain[t],t, T=T),
                                      Q,d[t],H=H,R=R)
        else:
            u[:,t],P[:,:,t] = ext_kf(u[:,t-1],P[:,:,t-1],
                                      lambda uu: model_augmented(uu,Ed[t],Ew[t],rain[t],t, T=T),
                                      Q*0.0)    

    return u

# Executed Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

## Data
df1 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_1h.xlsx"))
df10 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_10h.xlsx"))
df100 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_100h.xlsx"))
df1000 = pd.read_excel(osp.join(DATA_DIR, "processed_data/ok_1000h.xlsx"))
weather = pd.read_excel(osp.join(DATA_DIR, "processed_data/dvdk_weather.xlsx"))

df1 = df1.rename(columns={"utc_rounded": "utc"})
df10 = df10.rename(columns={"utc_rounded": "utc"})
df100 = df100.rename(columns={"utc_rounded": "utc"})
df1000 = df1000.rename(columns={"utc_rounded": "utc"})

## Setup grid of params
ngrid = 5                                    # number of candidate values to construct grid
Rgrid = np.linspace(1e-6, 1e-1, num=ngrid)     # data covariance
Qgrid = np.linspace(1e-6, 1e-1, num=ngrid)     # process covariance
dTrgrid = np.linspace(0.9, 2.0, num=ngrid)     # Multiplicative factor to define rain-wetting timelag, default 14 in wrfx model

ngrid = 5
T1grid = np.linspace(0.9, 1.1, num=ngrid+1)    # 1h timelag param
T10grid = np.linspace(9.5, 10.5, num=ngrid+1)   # 10h timelag param
T100grid = np.linspace(95, 105, num=ngrid+1)   # 100h timelag param
T1000grid = np.linspace(950, 1050, num=ngrid+1)   # 1000h timelag param

combo1    = make_param_grid(T=T1grid, R=Rgrid, Q =Qgrid, dTr = dTrgrid)
combo10  = make_param_grid(T=T10grid, R=Rgrid, Q =Qgrid, dTr = dTrgrid)
combo100  = make_param_grid(T=T100grid, R=Rgrid, Q =Qgrid, dTr = dTrgrid)
combo1000 = make_param_grid(T=T1000grid, R=Rgrid, Q =Qgrid, dTr = dTrgrid)

## Object to store param combos and configs
results = {
    '1h': {},
    '10h': {},
    '100h': {},
    '1000h': {}
}

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Invalid arguments. {len(sys.argv)} was given but 1 expected")
        print(f"Usage: {sys.argv[0]} <output_path>")
        print("Example: python ode_fit.py outputs/test.pkl")
        sys.exit(-1)  

    outpath = sys.argv[1]

    print("~"*50)
    print(f"Running ODE Fit for FMC Reanalysis")
    print()
    print("~"*50)
    print(f"1h Fuels")

    # Get weather data for time period
    weather2 = weather.copy()
    weather2 = weather2[weather2["utc"].between(df1["utc"].min(), df1["utc"].max())]
    df = (
        df1[["fm1", "utc"]]
        .merge(
            weather2[["Ed", "Ew", "rain", "utc"]],
            on="utc",
            how="right"
        )
    )
    # Print Summaries before starting
    print(f"    Start UTC: {df.utc.min()}")
    print(f"    End UTC: {df.utc.max()}")
    print(f"    Hours: {df.shape[0]}")
    print(f"    N. FMC Observations: {df.fm1.notna().sum()}")
    print()
    print(f"Number of Param Combos to Test: {len(combo1)}")
    for i in range(0, len(combo1)):
        # Extract param config, save to results object
        print("~"*50)
        print(f"    Running combo {i} out of {len(combo1)}")
        cc = combo1[i]
        cc["Tr"] = cc["T"]*cc["dTr"]
        params.update(cc)
        results["1h"][f"{i}"] = {}
        results["1h"][f"{i}"]["params"] = params.copy()
        # Run model with current config
        ui = ode_kf_fit(df, params, tstep = 1, fm_col = "fm1")
        results["1h"][f"{i}"]["u"] = ui
        # Calculate RMSE over stretch
        inds = np.where(df.fm1 <= 20)[0]
        rmse = np.sqrt(np.mean((df.fm1 - ui[0,:])**2))
        rmse20 = np.sqrt(np.mean((df.fm1[inds] - ui[0,inds])**2))
        results["1h"][f"{i}"]["rmse"] = rmse
        results["1h"][f"{i}"]["rmse_20"] = rmse20
        print(f"    RMSE: {rmse}")
        print(f"    RMSE (<=20): {rmse20}")

    print()
    print("~"*50)
    print(f"10h Fuels")
    # Get weather data for time period
    weather2 = weather.copy()
    weather2 = weather2[weather2["utc"].between(df10["utc"].min(), df10["utc"].max())]
    df = (df10[["fm10", "utc"]].merge(weather2[["Ed", "Ew", "rain", "utc"]],on="utc",how="right"))
    # Print Summaries before starting
    print(f"    Start UTC: {df.utc.min()}")
    print(f"    End UTC: {df.utc.max()}")
    print(f"    Hours: {df.shape[0]}")
    print(f"    N. FMC Observations: {df.fm10.notna().sum()}")
    print()
    print(f"Number of Param Combos to Test: {len(combo10)}")
    for i in range(0, len(combo10)):
        # Extract param config, save to results object
        print("~"*50)
        print(f"    Running combo {i} out of {len(combo10)}")
        cc = combo10[i]
        cc["Tr"] = cc["T"]*cc["dTr"]
        params.update(cc)
        results["10h"][f"{i}"] = {}
        results["10h"][f"{i}"]["params"] = params.copy()
        # Run model with current config
        ui = ode_kf_fit(df, params, tstep = 1, fm_col = "fm10")
        results["10h"][f"{i}"]["u"] = ui
        # Calculate RMSE over stretch
        inds = np.where(df.fm10 <= 20)[0]
        rmse = np.sqrt(np.mean((df.fm10 - ui[0,:])**2))
        rmse20 = np.sqrt(np.mean((df.fm10[inds] - ui[0,inds])**2))
        results["10h"][f"{i}"]["rmse"] = rmse
        results["10h"][f"{i}"]["rmse_20"] = rmse20
        print(f"    RMSE: {rmse}")
        print(f"    RMSE (<=20): {rmse20}")    
    
    print()
    print("~"*50)
    print(f"100h Fuels")
    
    # Get weather data for time period
    weather2 = weather.copy()
    weather2 = weather2[weather2["utc"].between(df100["utc"].min(), df100["utc"].max())]
    df = (
        df100[["fm100", "utc"]]
        .merge(
            weather2[["Ed", "Ew", "rain", "utc"]],
            on="utc",
            how="right"
        )
    )
    # Print Summaries before starting
    print(f"    Start UTC: {df.utc.min()}")
    print(f"    End UTC: {df.utc.max()}")
    print(f"    Hours: {df.shape[0]}")
    print(f"    N. FMC Observations: {df.fm100.notna().sum()}")
    print()
    print(f"Number of Param Combos to Test: {len(combo100)}")
    for i in range(0, len(combo100)):
        # Extract param config, save to results object
        print("~"*50)
        print(f"    Running combo {i} out of {len(combo100)}")
        cc = combo100[i]
        cc["Tr"] = cc["T"]*cc["dTr"]
        params.update(cc)
        results["100h"][f"{i}"] = {}
        results["100h"][f"{i}"]["params"] = params.copy()
        # Run model with current config
        ui = ode_kf_fit(df, params, tstep = 1, fm_col = "fm100")
        results["100h"][f"{i}"]["u"] = ui
        # Calculate RMSE over stretch
        inds = np.where(df.fm100 <= 20)[0]
        rmse = np.sqrt(np.mean((df.fm100 - ui[0,:])**2))
        rmse20 = np.sqrt(np.mean((df.fm100[inds] - ui[0,inds])**2))
        results["100h"][f"{i}"]["rmse"] = rmse
        results["100h"][f"{i}"]["rmse_20"] = rmse20
        print(f"    RMSE: {rmse}")
        print(f"    RMSE (<=20): {rmse20}")       
    

    print()
    print("~"*50)
    print(f"1000h Fuels")


    print()
    print("~"*50)
    print(f"Writing output to: {outpath}")
    with open(outpath, 'wb') as f:
        pickle.dump(results, f)
    

    