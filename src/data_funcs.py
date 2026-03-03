# Set of Functions to process and format fuel moisture model inputs
# These functions are specific to the particulars of the input data, and may not be generally applicable
# Generally applicable functions should be in utils.py
#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

from datetime import datetime
from dateutil.relativedelta import relativedelta
import numpy as np
import os
import os.path as osp
import sys
import pickle
import pandas as pd
import random
import copy
from abc import ABC, abstractmethod
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import warnings

# Set up project paths
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
CURRENT_DIR = osp.dirname(osp.normpath(osp.abspath(__file__)))
PROJECT_ROOT = osp.dirname(osp.normpath(CURRENT_DIR))
sys.path.append(osp.join(PROJECT_ROOT, "src"))
CONFIG_DIR = osp.join(PROJECT_ROOT, "etc")


# Read Project Module Code
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
from utils import Dict, read_yml, time_range, str2time, is_consecutive_hours, time_intp
import reproducibility




    
def sort_train_dict(d):
    """
    Rearrange keys based on number of observations in data subkey. Keys with most observations go first

    Used to make stateful batching easier
    
    NOTE: only intended to apply to train dict, as validation and test dicts were constructed so no missing data
    """
    return dict(sorted(d.items(), key=lambda item: item[1]["data"].shape[0], reverse=True))

def filter_empty_data(input_dict):
    return {k: v for k, v in input_dict.items() if v["data"].shape[0] > 0}


class MLData(ABC):
    """
    Abstract base class for ML Data, providing support for scaling. 
    Scaling performed on training data and applied to val and test.
    """    
    def __init__(self, train, val=None, test=None, scaler="standard", features_list=None, random_state=None):
        if random_state is not None:
            reproducibility.set_seed(random_state)
        
        self._run_checks(train, val, test, scaler)

        if scaler not in {"standard", "minmax"}:
            raise ValueError("scaler must be 'standard' or 'minmax'")
        self.scaler = StandardScaler() if scaler == "standard" else MinMaxScaler()
        self.features_list = features_list if features_list is not None else ["Ed", "Ew", "rain"]
        self.n_features = len(self.features_list)

        # Setup data fiels, e.g. X_train and y_train
        self._setup_data(train, val, test)
        # Assuming that units are all the same as it was checked in a previous step
        self.units = next(iter(train.values()))["units"]
    
    def _run_checks(self, train, val, test, scaler):
        """Validates input types for train, val, test, and scaler."""
        if not isinstance(train, dict):
            raise ValueError("train must be a dictionary")
        if val is not None and not isinstance(val, dict):
            raise ValueError("val must be a dictionary or None")
        if test is not None and not isinstance(test, dict):
            raise ValueError("test must be a dictionary or None")
        if scaler not in {"standard", "minmax"}:
            raise ValueError("scaler must be 'standard' or 'minmax'")
    
    @abstractmethod
    def _setup_data(self, train, val, test):
        """Abstract method to initialize X_train, y_train, X_val, y_val, X_test, y_test"""
        pass

    def _combine_data(self, data_dict):
        """Combines all DataFrames under 'data' keys into a single DataFrame."""
        return pd.concat([v["data"] for v in data_dict.values()], ignore_index=True)  
    
    def scale_data(self, verbose=True):
        """
        Scales the training data using the set scaler.
        NOTE: this converts pandas dataframes into numpy ndarrays.
        Tensorflow requires numpy ndarrays so this is intended behavior

        Parameters:
        -----------
        verbose : bool, optional
            If True, prints status messages. Default is True.

        Returns:
        ---------
        Nothing, modifies in place
        """        

        if not hasattr(self, "X_train"):
            raise AttributeError("No X_train within object. Run train_test_split first. This is to avoid fitting the scaler with prediction data.")
        if verbose:
            print(f"Scaling training data with scaler {self.scaler}, fitting on X_train")

        # Fit scaler on training data
        self.scaler.fit(self.X_train)
        # Transform data using fitted scaler
        self.X_train = self.scaler.transform(self.X_train)
        if hasattr(self, 'X_val'):
            if self.X_val is not None:
                self.X_val = self.scaler.transform(self.X_val)
        if self.X_test is not None:
            self.X_test = self.scaler.transform(self.X_test)    

    def inverse_scale(self, save_changes=False, verbose=True):
        """
        Inversely scales the data to its original form. Either save changes internally,
        or return tuple X_train, X_val, X_test

        Parameters:
        -----------
        return_X : str, optional
            Specifies what data to return after inverse scaling. Default is 'all_hours'.
        save_changes : bool, optional
            If True, updates the internal data with the inversely scaled values. Default is False.
        verbose : bool, optional
            If True, prints status messages. Default is True.
        """        
        if verbose:
            print("Inverse scaling data...")
        X_train = self.scaler.inverse_transform(self.X_train)
        X_val = self.scaler.inverse_transform(self.X_val)
        X_test = self.scaler.inverse_transform(self.X_test)

        if save_changes:
            print("Inverse transformed data saved")
            self.X_train = X_train
            self.X_val = X_val
            self.X_test = X_test
        else:
            if verbose:
                print("Inverse scaled, but internal data not changed.")
            return X_train, X_val, X_test    
    
    # def print_hashes(self, attrs_to_check = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test']):
    #     """
    #     Prints the hash of specified data attributes. 
    #     NOTE: AS OF FEB 3 2025 this doesn't work. data is saved in pandas and reproducibility to_numpy not guarenteed

    #     Parameters:
    #     -----------
    #     attrs_to_check : list, optional
    #         A list of attribute names to hash and print. Default includes 'X', 'y', and split data.
    #     """
        
    #     for attr in attrs_to_check:
    #         if hasattr(self, attr):
    #             value = getattr(self, attr)
    #             print(f"Hash of {attr}: {hash_ndarray(value)}") 


class StaticMLData(MLData):
    """
    Custom class to handle data scaling and extracting from dictionaries. 
    Static combines all data in train/val/test as independent observations in time. 
    So timeseries are not maintained and a single "sample" is one hour of data
    Inherits from MLData class.
    """    
    def _setup_data(self, train, val, test, y_col="fm", verbose=True):
        """
        Combines all DataFrames under 'data' keys for train, val, and test. 
        Static data does not keep track of timeseries, and throws all instantaneous samples into the same pool
        If train and val are None, still create those names as None objects

        Creates numpy ndarrays X_train, y_train, X_val, y_val, X_test, y_test
        """
        if verbose:
            print(f"Subsetting input data to {self.features_list}")

        
        X_train = self._combine_data(train)
        self.train_locs = X_train['stid'].to_numpy()
        self.train_times = X_train['date_time'].to_numpy().astype(str)
        self.y_train = X_train[y_col].to_numpy()
        self.X_train = X_train[self.features_list].to_numpy()

        self.X_val, self.y_val = (None, None)
        if val:
            X_val = self._combine_data(val)
            self.val_locs = X_val['stid'].to_numpy()
            self.val_times = X_val['date_time'].to_numpy().astype(str)
            self.y_val = X_val[y_col].to_numpy()
            self.X_val = X_val[self.features_list].to_numpy()
        
        self.X_test, self.y_test = (None, None)
        if test:
            X_test = self._combine_data(test)
            self.test_locs = X_test['stid'].to_numpy()
            self.test_times = X_test['date_time'].to_numpy().astype(str)
            self.y_test = X_test[y_col].to_numpy()
            self.X_test = X_test[self.features_list].to_numpy()

        if verbose:
            print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
            if self.X_val is not None:
                print(f"X_val shape: {self.X_val.shape}, y_val shape: {self.y_val.shape}")
            if self.X_test is not None:
                print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")
            
  
if __name__ == '__main__':

    print("Imports successful, no executable code") 

    
