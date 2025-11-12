import numpy as np
import yaml
from datetime import datetime, timezone
import pandas as pd

plot_styles = {
    'fm': {'color': '#468a29', 'linestyle': '-', 'label': 'Observed FMC'},
    'fm_preds': {'color': '#468a29', 'linestyle': '-', 'label': 'Observed FMC'},
    'Ed': {'color': '#EF847C', 'linestyle': '--', 'alpha':.8, 'label': 'drying EQ'},
    'Ew': {'color': '#7CCCEF', 'linestyle': '--', 'alpha':.8, 'label': 'wetting EQ'},
    'rain': {'color': 'b', 'linestyle': '-', 'alpha':.9, 'label': 'Rain'},
    'model': {'color': 'k', 'linestyle': '-', 'label': 'Predicted FMC'}
}


class Dict(dict):
    """
    A dictionary that allows member access to its keys.
    A convenience class.
    """

    def __init__(self, d):
        """
        Updates itself with d.
        """
        self.update(d)

    def __getattr__(self, item):
        return self[item]

    def __setattr__(self, item, value):
        self[item] = value

    def __getitem__(self, item):
        if item in self:
            return super().__getitem__(item)
        else:
            for key in self:
                if isinstance(key,(range,tuple)) and item in key:
                    return super().__getitem__(key)
            raise KeyError(item)

    def keys(self):
        if any([isinstance(key,(range,tuple)) for key in self]):
            keys = []
            for key in self:
                if isinstance(key,(range,tuple)):
                    for k in key:
                        keys.append(k)
                else:
                    keys.append(key)
            return keys
        else:
            return super().keys()

def str2time(input):
    """
    Convert string or list of strings to datetime, supporting multiple formats.
    """
    formats = [
        '%Y-%m-%dT%H:%M:%S%z',      # ISO 8601 with 'T'
        '%Y-%m-%d %H:%M:%S%z',      # ISO 8601 with space instead of 'T'
        '%Y-%m-%dT%H:%M:%S',        # No timezone
        '%Y-%m-%d %H:%M:%S',        # No timezone, space separator
    ]

    def parse(s):
        s = s.replace('Z', '+00:00')
        for fmt in formats:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unsupported datetime format: {s}")

    if isinstance(input, str):
        return parse(input)
    elif isinstance(input, list):
        return [parse(s) for s in input]
    else:
        raise ValueError("Input must be a string or a list of strings")

def filter_nan_values(t1, v1):
    # Filter out NaN values from v1 and corresponding times in t1
    valid_indices = ~np.isnan(v1)  # Indices where v1 is not NaN
    t1_filtered = np.array(t1)[valid_indices]
    v1_filtered = np.array(v1)[valid_indices]
    return t1_filtered, v1_filtered


def time_intp(t1, v1, t2):
    """
    Interpolates 1D time series values v1 from times t1 onto new times t2.
    Ensures valid 1D inputs, removes NaNs, converts datetimes to timestamps,
    and performs linear interpolation using np.interp.

    Args:
        t1 (np.ndarray): 1D array of datetime objects representing known time points.
        v1 (np.ndarray): 1D array of values corresponding to t1.
        t2 (np.ndarray): 1D array of datetime objects representing target time points.
    """
    if t1.ndim != 1 or v1.ndim != 1 or t2.ndim != 1:
        raise ValueError("Inputs must be 1D arrays")
    if len(t1) != len(v1):
        raise ValueError("t1 and v1 must have the same length")
    t1_no_nan, v1_no_nan = filter_nan_values(t1, v1)
    # Convert datetime64 arrays to POSIX seconds
    def to_seconds(arr):
        if np.issubdtype(arr.dtype, np.datetime64):
            return arr.astype("datetime64[s]").astype(float)
        # If pandas or datetime objects, convert with timestamp()
        return np.array([t.timestamp() for t in arr], dtype=float)

    t1_stamps = to_seconds(t1_no_nan)
    t2_stamps = to_seconds(t2)
    
    # Interpolate using the filtered data
    v2_interpolated = np.interp(t2_stamps, t1_stamps, v1_no_nan)
    if np.isnan(v2_interpolated).any():
        # logging.error('time_intp: interpolated output contains NaN')
        raise ValueError("")

    return v2_interpolated


def is_consecutive_hours(times):
    """
    Check whether input array are consecutive 1 hour increments
    """
    # Convert to numpy timedelta64[h] for hour differences
    time_diffs = np.diff(times).astype('timedelta64[h]')
    return np.all(time_diffs == np.timedelta64(1, 'h'))

def time_range(start, end, freq="1h"):
    """
    Wrapper function for pandas date range. Checks to allow for input of datetimes or strings
    """
    if (type(start) is str) and (type(end) is str):
        start = str2time(start)
        end = str2time(end)
    else:
        assert isinstance(start, datetime) and isinstance(end, datetime), "Args start and end must be both strings or both datetimes"

    times = pd.date_range(start, end, freq=freq)
    times = times.to_pydatetime()
    return times


# Generic helper function to read yaml files
def read_yml(yaml_path, subkey=None):
    """
    Reads a YAML file and optionally retrieves a specific subkey.

    Parameters:
    -----------
    yaml_path : str
        The path to the YAML file to be read.
    subkey : str, optional
        A specific key within the YAML file to retrieve. If provided, only the value associated 
        with this key will be returned. If not provided, the entire YAML file is returned as a 
        dictionary. Default is None.

    Returns:
    --------
    dict or any
        The contents of the YAML file as a dictionary, or the value associated with the specified 
        subkey if provided.

    """    
    with open(yaml_path, 'r') as file:
        d = yaml.safe_load(file)
        if subkey is not None:
            d = d[subkey]
    return d
    