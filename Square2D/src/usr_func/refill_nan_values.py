"""
Refills nan values with neighbouring values.
"""
import numpy as np
from scipy.interpolate import NearestNDInterpolator


def refill_nan_values(data: np.ndarray) -> np.ndarray:
    mask = np.where(~np.isnan(data))
    interp = NearestNDInterpolator(np.transpose(mask), data[mask])
    filled_data = interp(*np.indices(data.shape))
    return filled_data
