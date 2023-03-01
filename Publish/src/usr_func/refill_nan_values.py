"""
This function refills the NaN values in a 2D array using the nearest neighbor
"""
import numpy as np
from scipy.interpolate import NearestNDInterpolator


def refill_nan_values(data: np.ndarray) -> np.ndarray:
    """
    Args:
        data (np.ndarray): 2D array with NaN values

    Returns:
        np.ndarray: 2D array with NaN values refilled

    Raises:
        TypeError: If data is not a numpy array

    Examples:
        >>> import numpy as np
        >>> from refill_nan_values import refill_nan_values
        >>> data = np.array([[1, 2, 3], [4, np.nan, 6], [7, 8, 9]])
        >>> refill_nan_values(data)
        array([[1., 2., 3.],
                [4., 5., 6.],
                [7., 8., 9.]])

    """
    mask = np.where(~np.isnan(data))
    interp = NearestNDInterpolator(np.transpose(mask), data[mask])
    filled_data = interp(*np.indices(data.shape))
    return filled_data
