"""
Vectorize the input array according to column first ordering.
"""
import numpy as np


def vectorize(array: np.ndarray) -> np.ndarray:
    """
    Args:
        array: (n, m) dimension array.

    Returns:
        vector: (n*m,  1) dimension vector.

    Examples:
        >>> array = np.array([[1, 2, 3],
        ...                   [4, 5, 6]])
        >>> vectorize(array)
        array([[1],
               [4],
               [2],
               [5],
               [3],
               [6]])

    """
    return array.reshape(-1, 1)
