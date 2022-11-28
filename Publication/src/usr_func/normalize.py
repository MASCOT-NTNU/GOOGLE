""" Normalise the vector between min and max. """

import numpy as np


def normalize(x, amin=0, amax=1):
    return (x - np.amin(x)) / (np.amax(x) - np.amin(x)) * (amax - amin) + amin


if __name__ == "__main__":
    x = np.arange(10)
    print(x)
    print(normalize(x))

