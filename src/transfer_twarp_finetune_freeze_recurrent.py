# Executable module to run transfer analysis 
# Freeze Recurrent layer
# Methodology: for each fuel class construct grid of time warp params, modify LSTM, run fit on training set
    # Pick best time warp on val set
    # Predict test set for accuracy
# NOTE: this script will assume the form of the layer order from the pretrained model. The RNN_FLexible custom class can handle frozen weights at initiate step


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


def eval_interp_metrics(
    *,
    df_test,
    y_test,
    preds,
    time_intp,
    y_col=None,
    t_model_col="utc",
    t_obs_col="utc_prov",
    missing_value=-9999,
    threshold=None,
):
    """
    Evaluate RMSE/bias/R2 after interpolating model predictions (on t_model_col)
    to observation times (t_obs_col) for non-missing observations.

    Parameters
    ----------
    df_test : pandas.DataFrame
        Must contain columns t_model_col and t_obs_col.
    y_test : array-like
        Target values aligned with df_test rows (or pass y_col instead).
    preds : array-like
        Model predictions aligned with df_test rows at t_model_col times.
        Can be shape (1, T, 1), (T,), etc.
    time_intp : callable
        Function time_intp(t1, v1, t2) -> interpolated values at t2.
    y_col : str | None
        If provided, y_test is ignored and y is pulled from df_test[y_col].
    threshold : float | None
        If provided, metrics are computed only where y <= threshold (after missing filter).

    Returns
    -------
    out : dict
        {
          "rmse": float,
          "bias": float,
          "r2": float,
          "n": int,
          "preds_intp": np.ndarray,   # length n
          "inds": np.ndarray          # indices into original df_test/y_test
        }
    """
    # Pull y
    if y_col is not None:
        y = df_test[y_col].to_numpy()
    else:
        y = np.asarray(y_test)

    # Flatten preds to 1D aligned with df_test rows
    v1 = np.asarray(preds).reshape(-1)
    if v1.shape[0] != len(df_test):
        raise ValueError(f"preds length {v1.shape[0]} does not match df_test length {len(df_test)}")

    # Non-missing observation indices
    inds = np.where(y != missing_value)[0]
    if inds.size == 0:
        return {"rmse": np.nan, "bias": np.nan, "r2": np.nan, "n": 0, "preds_intp": np.array([]), "inds": inds}

    # Optional threshold filter (applied after missing filter)
    if threshold is not None:
        inds = inds[y[inds] <= threshold]
        if inds.size == 0:
            return {"rmse": np.nan, "bias": np.nan, "r2": np.nan, "n": 0, "preds_intp": np.array([]), "inds": inds}

    # Interpolate predictions to observed times
    preds_intp = time_intp(
        t1=df_test[t_model_col].to_numpy(),
        v1=v1,
        t2=df_test[t_obs_col].iloc[inds].to_numpy(),
    )

    y_eval = y[inds]
    rmse = float(np.sqrt(mean_squared_error(y_eval, preds_intp)))
    bias = float(np.mean(y_eval - preds_intp))
    r2 = float(r2_score(y_eval, preds_intp))

    return {"rmse": rmse, "bias": bias, "r2": r2, "n": int(inds.size), "preds_intp": np.asarray(preds_intp), "inds": inds}


# Executed Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
if __name__ == '__main__':
    if len(sys.argv) not in [2, 3]:
        print(f"Invalid arguments. {len(sys.argv)-1} was given but 1 or 2 expected")
        print(f"Usage: {sys.argv[0]} <config_path> [seed]")
        print("Example: python src/transfer_twarp_finetune_freeze_recurrent.py etc/thesis_config.yaml")
        print("Example: python src/transfer_twarp_finetune_freeze_recurrent.py etc/thesis_config.yaml 17")
        sys.exit(-1)


    # Setup 
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    confpath = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) == 3 else None
    conf = Dict(read_yml(confpath))
    print(f"~"*50)
    print(f"Running Transfer-Learning, Fine-Tune Freeze Recurrent with config file: {confpath}")

    if seed is not None:
        reproducibility.set_seed(seed)
        output_dir = osp.join(conf.output_dir, "freeze_recurrent_reps", f"seed_{seed}")
        print(f"RNN Model Dir: {osp.join(conf.reps_dir, f'seed_{seed}')}")
        params = Dict(read_yml(osp.join(conf.reps_dir, f"seed_{seed}", "params.yaml")))
        params["freeze_layers"] = [1, 0, 0, 0] # matches hidden_layers=[lstm, dense, dense, dropout]
        rnn = RNN_Flexible(params=params, loss=mse_masked, random_state=seed)
        scaler = joblib.load(osp.join(conf.reps_dir, f"seed_{seed}", "scaler.joblib"))
        rnn.load_weights(osp.join(conf.reps_dir, f"seed_{seed}", 'rnn.keras'))
    else:
        seed = 11001000 # arbitrary, made it by combining 1-100-1000
        reproducibility.set_seed(seed)
        output_dir = osp.join(conf.output_dir, "freeze_recurrent")
        print(f"RNN Model Dir: {conf.rnn_dir}")
        params = Dict(read_yml(osp.join(conf.rnn_dir, "params.yaml")))
        params["freeze_layers"] = [1, 0, 0, 0] # matches hidden_layers=[lstm, dense, dense, dropout]
        rnn = RNN_Flexible(params=params, loss=mse_masked, random_state=seed)
        scaler = joblib.load(osp.join(conf.rnn_dir, "scaler.joblib"))
        rnn.load_weights(osp.join(conf.rnn_dir, 'rnn.keras'))

    os.makedirs(output_dir, exist_ok=True)
    
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
        rnn.load_weights(osp.join(conf.rnn_dir, 'rnn.keras')) # reset weights to baseline
        weightsi = warp_weights(weights10, bi_warp = bs["bi"], bf_warp = bs["bf"])
        rnn.get_layer("lstm").set_weights(weightsi)
        rnn.fit(X_train_samples, y_train_samples, validation_data = (XX_val, yy_val), batch_size=params.batch_size, epochs=params.epochs, verbose_fit = False, plot_history=False)
        
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

    # Re-train using best params (clean run)
    # Reset to baseline pretrained weights
    reproducibility.set_seed(seed)
    rnn.load_weights(osp.join(conf.rnn_dir, "rnn.keras"))
    
    # Apply best warp to the LSTM weights
    weights_best = warp_weights(
        weights10,
        bi_warp=fm1_best["params"]["bi"],
        bf_warp=fm1_best["params"]["bf"],
    )
    rnn.get_layer("lstm").set_weights(weights_best)
    # Train again (fresh run from baseline+warp)
    rnn.fit(
        X_train_samples,
        y_train_samples,
        validation_data=(XX_val, yy_val),
        batch_size=params.batch_size,
        epochs=params.epochs,
        verbose_fit=True,
        plot_history=False,
    )

    preds1 = rnn.predict(XX_test)
    
    # Calc Accuracy
    fm1_test = eval_interp_metrics(
        df_test=df1_test,
        y_test=y_test,
        preds=preds1,
        time_intp=time_intp,
        t_model_col="utc",
        t_obs_col="utc_prov",
        missing_value=-9999,
        threshold=None,
    )
    
    fm1_test_30 = eval_interp_metrics(
        df_test=df1_test,
        y_test=y_test,
        preds=preds1,
        time_intp=time_intp,
        t_model_col="utc",
        t_obs_col="utc_prov",
        missing_value=-9999,
        threshold=30,
    )
    
    # Example storing into your fm1_best dict
    fm1_best["rmse"] = fm1_test["rmse"]
    fm1_best["bias"] = fm1_test["bias"]
    fm1_best["r2"]   = fm1_test["r2"]
    
    fm1_best["rmse_30"] = fm1_test_30["rmse"]
    fm1_best["bias_30"] = fm1_test_30["bias"]
    fm1_best["r2_30"]   = fm1_test_30["r2"]

    print(f"FM1 Test Accuracy")
    print(f"    RMSE: {fm1_best['rmse']}")
    print(f"    RMSE30: {fm1_best['rmse_30']}")
    


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

    bf_grid = np.linspace(conf.fm100_bf_low, conf.fm100_bf_high, num=conf.ngrid).round(4)
    bi_grid = np.linspace(conf.fm100_bi_low, conf.fm100_bi_high, num=conf.ngrid).round(4)
    fm100_grid = make_param_grid(bf=bf_grid, bi=bi_grid)
    print()
    print(f"N time-warp Param Combos: {len(fm100_grid)}")

    #d Loop over param grids, fit to training data w early stop, calculate RMSE on val
    results_100 = {}
    for i, bs in enumerate(fm100_grid):
        print("~"*50)
        print(f"FM100 Param Combo {i+1} out of {len(fm100_grid)}")
        print(f"Params: {bs}")    
        rnn.load_weights(osp.join(conf.rnn_dir, 'rnn.keras')) # reset weights to baseline
        weightsi = warp_weights(weights10, bi_warp = bs["bi"], bf_warp = bs["bf"])
        rnn.get_layer("lstm").set_weights(weightsi)
        rnn.fit(X_train_samples, y_train_samples, validation_data = (XX_val, yy_val), batch_size=params.batch_size, epochs=params.epochs, verbose_fit = False, plot_history=False)
        
        print(f"Predicting Val Set")
        preds = rnn.predict(XX_val)
        val_mse = np.float32(mse_masked(y_val.flatten(), preds.flatten()))
        print(f"    {val_mse=}")
        results_100[i]={}
        results_100[i]["val_rmse"] = np.sqrt(val_mse)
        results_100[i]["params"] = bs
        results_100[i]["preds"] = preds  


    # Find min val_rmse case
    fm100_best_key = min(results_100, key=lambda ci: results_100[ci]["val_rmse"])
    fm100_best = results_100[fm100_best_key]
    print()
    print("Best Config from Val Error:")
    print(f"Min Val RMSE: {np.round(fm100_best['val_rmse'], 4)}")
    print(f"Time-Warp Params: {fm100_best['params']}")

    # Re-train using best params (clean run)
    # Reset to baseline pretrained weights
    reproducibility.set_seed(seed)
    rnn.load_weights(osp.join(conf.rnn_dir, "rnn.keras"))
    
    # Apply best warp to the LSTM weights
    weights_best = warp_weights(
        weights10,
        bi_warp=fm100_best["params"]["bi"],
        bf_warp=fm100_best["params"]["bf"],
    )
    rnn.get_layer("lstm").set_weights(weights_best)
    # Train again (fresh run from baseline+warp)
    rnn.fit(
        X_train_samples,
        y_train_samples,
        validation_data=(XX_val, yy_val),
        batch_size=params.batch_size,
        epochs=params.epochs,
        verbose_fit=True,
        plot_history=False,
    )

    preds100 = rnn.predict(XX_test)
    
    # Calc Accuracy
    fm100_test = eval_interp_metrics(
        df_test=df100_test,
        y_test=y_test,
        preds=preds100,
        time_intp=time_intp,
        t_model_col="utc",
        t_obs_col="utc_prov",
        missing_value=-9999,
        threshold=None,
    )
    
    # Example storing into your fm100_best dict
    fm100_best["rmse"] = fm100_test["rmse"]
    fm100_best["bias"] = fm100_test["bias"]
    fm100_best["r2"]   = fm100_test["r2"]
    
    print(f"FM100 Test Accuracy")
    print(f"    RMSE: {fm100_test['rmse']}")




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

    bf_grid = np.linspace(conf.fm1000_bf_low, conf.fm1000_bf_high, num=conf.ngrid).round(4)
    bi_grid = np.linspace(conf.fm1000_bi_low, conf.fm1000_bi_high, num=conf.ngrid).round(4)
    fm1000_grid = make_param_grid(bf=bf_grid, bi=bi_grid)
    print()
    print(f"N time-warp Param Combos: {len(fm1000_grid)}")

    #d Loop over param grids, fit to training data w early stop, calculate RMSE on val
    results_1000 = {}
    for i, bs in enumerate(fm1000_grid):
        print("~"*50)
        print(f"FM1000 Param Combo {i+1} out of {len(fm1000_grid)}")
        print(f"Params: {bs}")    
        rnn.load_weights(osp.join(conf.rnn_dir, 'rnn.keras')) # reset weights to baseline
        weightsi = warp_weights(weights10, bi_warp = bs["bi"], bf_warp = bs["bf"])
        rnn.get_layer("lstm").set_weights(weightsi)
        rnn.fit(X_train_samples, y_train_samples, validation_data = (XX_val, yy_val), batch_size=params.batch_size, epochs=params.epochs, verbose_fit = False, plot_history=False)
        
        print(f"Predicting Val Set")
        preds = rnn.predict(XX_val)
        val_mse = np.float32(mse_masked(y_val.flatten(), preds.flatten()))
        print(f"    {val_mse=}")
        results_1000[i]={}
        results_1000[i]["val_rmse"] = np.sqrt(val_mse)
        results_1000[i]["params"] = bs
        results_1000[i]["preds"] = preds  


    # Find min val_rmse case
    fm1000_best_key = min(results_1000, key=lambda ci: results_1000[ci]["val_rmse"])
    fm1000_best = results_1000[fm1000_best_key]
    print()
    print("Best Config from Val Error:")
    print(f"Min Val RMSE: {np.round(fm1000_best['val_rmse'], 4)}")
    print(f"Time-Warp Params: {fm1000_best['params']}")

    # Re-train using best params (clean run)
    # Reset to baseline pretrained weights
    reproducibility.set_seed(seed)
    rnn.load_weights(osp.join(conf.rnn_dir, "rnn.keras"))
    
    # Apply best warp to the LSTM weights
    weights_best = warp_weights(
        weights10,
        bi_warp=fm1000_best["params"]["bi"],
        bf_warp=fm1000_best["params"]["bf"],
    )
    rnn.get_layer("lstm").set_weights(weights_best)
    # Train again (fresh run from baseline+warp)
    rnn.fit(
        X_train_samples,
        y_train_samples,
        validation_data=(XX_val, yy_val),
        batch_size=params.batch_size,
        epochs=params.epochs,
        verbose_fit=True,
        plot_history=False,
    )

    preds1000 = rnn.predict(XX_test)
    
    # Calc Accuracy
    fm1000_test = eval_interp_metrics(
        df_test=df1000_test,
        y_test=y_test,
        preds=preds1000,
        time_intp=time_intp,
        t_model_col="utc",
        t_obs_col="utc_prov",
        missing_value=-9999,
        threshold=None,
    )
    
    # Example storing into your fm1000_best dict
    fm1000_best["rmse"] = fm1000_test["rmse"]
    fm1000_best["bias"] = fm1000_test["bias"]
    fm1000_best["r2"]   = fm1000_test["r2"]
    
    print(f"FM1000 Test Accuracy")
    print(f"    RMSE: {fm1000_test['rmse']}")
        

    # Output
    results_test = {}
    results_test["FM1"] = fm1_best
    results_test["FM10"] = fm10_best
    results_test["FM100"] = fm100_best
    results_test["FM1000"] = fm1000_best
    out_file = osp.join(output_dir, "results_finetune.pkl")
    print(f"Writing Output to: {out_file}")
    with open(out_file, "wb") as f:
        pickle.dump(results_test, f, protocol=pickle.HIGHEST_PROTOCOL)










