"""
Vectorize the input array.
"""
import numpy as np


def vectorize(array: np.ndarray) -> np.ndarray:
    """
    Vectorize the input array according to column first ordering.
    Args:
        array: (n, m) dimension array.
    Returns:
        vector: (n*m,  1) dimension vector.
    """
    return array.reshape(-1, 1)
