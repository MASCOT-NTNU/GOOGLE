"""
This script prepares the cholesky factorization for the CTD simulator.

Methodology:
    1. Construct the covariance matrix for the CTD simulator.
    2. Perform cholesky factorization on the covariance matrix.
    3. Save the cholesky factorization for future use.

!!! Note:
    1. The covariance matrix has fixed nugget and other coefficients.
    2. When other coefficients and sigma changes, then it should be updated as well.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-09-05
"""
from SINMOD import SINMOD
import numpy as np
from pykdtree.kdtree import KDTree
from scipy.spatial.distance import cdist
from time import time
import os


class CovarianceMatrix:

    def __init__(self, sigma: float = 1., l_range: float = 700,
                 filepath: str = os.getcwd() + "/../sinmod/samples_2022.05.11.nc") -> None:
        """
        Args:
            sigma: spatial variability of the salinity field.
            l_range: lateral correlation range of the salinity field, got from variogram analysis.
            filepath: default SINMOD path for the prior salinity field.

        """
        self.sigma = sigma
        self.l_range = l_range
        self.filepath = filepath
        self.sinmod = SINMOD(self.filepath)

        # s0, set up grid
        self.setup_grid()

        # s1, get covariance matrix
        self.get_covariance_matrix()

        # s2, save cholesky factorization
        self.save_cholesky()

        pass

    def setup_grid(self) -> None:
        self.grid = self.sinmod.get_data()[:, :3]
        ind_surface = np.where(self.grid[:, -1] == 0.5)[0]
        self.grid = self.grid[ind_surface, :2]
        self.grid_tree = KDTree(self.grid)

    def get_covariance_matrix(self) -> None:
        dm = cdist(self.grid, self.grid)
        eta = 4.5 / self.l_range
        self.cov = self.sigma ** 2 * ((1 + eta * dm) * np.exp(-eta * dm))

    def save_cholesky(self) -> None:
        t0 = time()
        self.L = np.linalg.cholesky(self.cov)
        np.savez(os.getcwd() + "/AUVSimulator/cholesky.npz", L=self.L)
        print("Saving cholesky factorization takes: ", time() - t0)


if __name__ == "__main__":
    cm = CovarianceMatrix()

