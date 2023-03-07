"""
This module calculates the expected Bernoulli variance using a bivariate normal distribution density function.
"""

import numpy as np
from scipy.stats import multivariate_normal

def calculate_analytical_ebv(parameter_set: np.ndarray) -> float:
    """
    Calculate the expected Bernoulli variance using a bivariate normal distribution density function.

    Parameters:
        parameter_set (np.ndarray): parameter set containing the mean and variance of the bivariate normal distribution.
            mur (float): mean of the bivariate normal distribution.
            sig2r_1 (float): variance of the bivariate normal distribution.
            sig2r (float): covariance of the bivariate normal distribution.

    """
    mur, sig2r_1, sig2r = parameter_set
    ebv = multivariate_normal.cdf(np.array([0, 0]), np.array([-mur, mur]).squeeze(),
                                  np.array([[sig2r_1, -sig2r],
                                            [-sig2r, sig2r_1]]).squeeze())
    return ebv
