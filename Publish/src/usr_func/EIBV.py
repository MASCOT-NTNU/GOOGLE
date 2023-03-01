"""
This function calculates the Expected Information Benefit Value (EIBV) for a given threshold.
"""

import numpy as np
from scipy.stats import multivariate_normal, norm


def EIBV_mvn(threshold, mu, Sig, H, R) -> float:
    """
    Calculates the Expected Information Benefit Value (EIBV) using the multivariate normal distribution.

    Parameters:
        threshold (float): Threshold for characterizing the threshold between fresh water and saline water.
        mu (np.ndarray): Mean vector.
        Sig (np.ndarray): Covariance matrix.
        H (np.ndarray): Sampling vector.
        R (np.ndarray): Nugget or measurement noise matrix.

    Returns:
        float: EIBV value.

    """
    Sigxi = Sig @ H.T @ np.linalg.solve((H @ Sig @ H.T + R), H @ Sig)
    V = Sig - Sigxi
    sa2 = np.diag(V)

    IntA = 0
    for i in range(len(mu)):
        sn2 = sa2[i]
        sn = np.sqrt(sn2)
        m = mu[i][0]
        mur = (threshold - m) / sn
        IntA += multivariate_normal.cdf([0, 0], [-mur, mur], [[1, 0], [0, 1]])
    return IntA


def EIBV_approximate(threshold, mu, Sig, H, R) -> float:
    """
    Calculates the Expected Information Benefit Value (EIBV) using the univariate normal distribution.

    Parameters:
        threshold (float): Threshold for characterizing the threshold between fresh water and saline water.
        mu (np.ndarray): Mean vector.
        Sig (np.ndarray): Covariance matrix.
        H (np.ndarray): Sampling vector.
        R (np.ndarray): Nugget or measurement noise matrix.

    Returns:
        float: EIBV value.

    """
    Sigxi = Sig @ H.T @ np.linalg.solve((H @ Sig @ H.T + R), H @ Sig)
    V = Sig - Sigxi
    sa2 = np.diag(V)

    p = norm.cdf(threshold, mu.flatten(), np.sqrt(sa2))
    bv = p * (1 - p)
    return np.sum(bv)

