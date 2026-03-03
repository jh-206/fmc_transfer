# Set of functions to use for loss functions and model evaluation

import numpy as np
# from scipy.interpolate import CubicSpline
# from sklearn.metrics import mean_squared_error
import pandas as pd

# Equilibrium moisture content equations
def calc_eq(rh, temp):
    rh = np.asarray(rh, dtype=float)
    temp = np.asarray(temp, dtype=float)
    Ed = 0.924 * rh**0.679 + 0.000499 * np.exp(0.1 * rh) + 0.18 * (21.1 + 273.15 - temp) * (1 - np.exp(-0.115 * rh))
    Ew = 0.618 * rh**0.753 + 0.000454 * np.exp(0.1 * rh) + 0.18 * (21.1 + 273.15 - temp) * (1 - np.exp(-0.115 * rh))
    return Ed, Ew

