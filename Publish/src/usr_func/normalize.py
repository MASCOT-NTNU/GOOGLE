"""
Normalize a numpy array to a range of [amin, amax]
"""

import numpy as np


def normalize(x, amin=0, amax=1) -> np.ndarray:
    """
    Args:
        x: numpy array to be normalized
        amin: minimum value of the normalized array
        amax: maximum value of the normalized array

    Returns:
        Normalized numpy array

    Examples:
        >>> x = np.arange(10)
        >>> print(x)
        [0 1 2 3 4 5 6 7 8 9]
        >>> print(normalize(x))
        [0.         0.11111111 0.22222222 0.33333333 0.44444444 0.55555556
            0.66666667 0.77777778 0.88888889 1.        ]

        >>> x = np.arange(10)
        >>> print(x)
        [0 1 2 3 4 5 6 7 8 9]
        >>> print(normalize(x, amin=1, amax=2))
        [1.         1.11111111 1.22222222 1.33333333 1.44444444 1.55555556
            1.66666667 1.77777778 1.88888889 2.        ]

    """
    return (x - np.amin(x)) / (np.amax(x) - np.amin(x)) * (amax - amin) + amin


if __name__ == "__main__":
    x = np.arange(10)
    print(x)
    print(normalize(x))

