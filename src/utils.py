import numpy as np

def time_intp(t1, v1, t2):
    # Check if t1 v1 t2 are 1D arrays
    if t1.ndim != 1:
        # logging.error("Error: t1 is not a 1D array. Dimension: %s", t1.ndim)
        # return None
        raise ValueError("")
    if v1.ndim != 1:
        # logging.error("Error: v1 is not a 1D array. Dimension %s:", v1.ndim)
        # return None
        raise ValueError("")
    if t2.ndim != 1:
        # logging.errorr("Error: t2 is not a 1D array. Dimension: %s", t2.ndim)
        # return None
        raise ValueError("")
    # Check if t1 and v1 have the same length
    if len(t1) != len(v1):
        # logging.error("Error: t1 and v1 have different lengths: %s %s",len(t1),len(v1))
        # return None
        raise ValueError("")
    t1_no_nan, v1_no_nan = filter_nan_values(t1, v1)
    # print('t1_no_nan.dtype=',t1_no_nan.dtype)
    # Convert datetime objects to timestamps
    t1_stamps = np.array([t.timestamp() for t in t1_no_nan])
    t2_stamps = np.array([t.timestamp() for t in t2])
    
    # Interpolate using the filtered data
    v2_interpolated = np.interp(t2_stamps, t1_stamps, v1_no_nan)
    if np.isnan(v2_interpolated).any():
        # logging.error('time_intp: interpolated output contains NaN')
        raise ValueError("")

    return v2_interpolated