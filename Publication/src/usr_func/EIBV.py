"""
This function contains two different implementations for EIBV calculation.
"""

import numpy as np
from scipy.stats import multivariate_normal, norm


def EIBV_mvn(threshold, mu, Sig, H, R):
    """
    threshold: float number for characterizing the threshold between fresh water and saline water.
    mu: mean vector
    Sig: Covariance matrix
    H: sampling vector
    R: nugget or measurement noise matrix
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


def calc_eibv():
    pass



def EIBV_approximate(threshold, mu, Sig, H, R):
    """
    threshold: float number for characterizing the threshold between fresh water and saline water.
    mu: mean vector
    Sig: Covariance matrix
    H: sampling vector
    R: nugget or measurement noise matrix
    """
    Sigxi = Sig @ H.T @ np.linalg.solve((H @ Sig @ H.T + R), H @ Sig)
    V = Sig - Sigxi
    sa2 = np.diag(V)

    p = norm.cdf(threshold, mu.flatten(), np.sqrt(sa2))
    bv = p * (1 - p)
    return np.sum(bv)

