"""
This script builds the kernel for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.spatial.distance import cdist
from Config.Config import FILEPATH, NUGGET


# == Parameters
SIGMA = 1.5
LATERAL_RANGE = 700
# ==


class GRF:

    def __init__(self):
        self.load_grf_grid()
        self.load_prior_mean()
        self.get_covariance_matrix()
        self.get_simulated_truth()
        self.mu_cond = self.mu_prior
        self.Sigma_cond = self.Sigma_prior
        print("GRF1- is set up successfully!")

    def load_grf_grid(self):
        self.grf_grid = pd.read_csv(FILEPATH+"Config/GRFGrid.csv").to_numpy()
        self.N = self.grf_grid.shape[0]
        print("GRF1: Grid is loaded successfully!")

    def load_prior_mean(self):
        self.mu_prior = pd.read_csv(FILEPATH + "Config/data_interpolated.csv")['salinity'].to_numpy()
        print("GRF2: Prior mean is loaded successfully!")

    def get_covariance_matrix(self):
        self.sigma = SIGMA
        self.eta = 4.5 / LATERAL_RANGE
        self.tau = np.sqrt(NUGGET)
        self.R = np.diagflat(self.tau ** 2)
        distance_matrix = cdist(self.grf_grid, self.grf_grid)
        self.Sigma_prior = self.sigma ** 2 * (1 + self.eta * distance_matrix) * np.exp(-self.eta * distance_matrix)
        print("GRF3: Covariance matrix is computed successfully!")

    def get_simulated_truth(self):
        self.mu_truth = (self.mu_prior.reshape(-1, 1) +
                         np.linalg.cholesky(self.Sigma_prior) @
                         np.random.randn(len(self.mu_prior)).reshape(-1, 1))
        print("GRF4: Simulated truth is computed successfully!")

    def update_grf_model(self, ind_measured=0, salinity_measured=0):
        t1 = time.time()
        F = np.zeros([self.N, 1])
        F[ind_measured] = True
        C = F.T @ self.Sigma_cond @ F + self.R
        self.mu_cond = self.mu_cond + self.Sigma_cond @ F @ np.linalg.solve(C, (salinity_measured - F.T @ self.mu_cond))
        self.Sigma_cond = self.Sigma_cond - self.Sigma_cond @ F @ np.linalg.solve(C, F.T @ self.Sigma_cond)
        t2 = time.time()
        print("GRF model updates takes: ", t2 - t1)

    def get_ind_from_location(self, x, y):
        dx = (self.grf_grid[:, 0] - x)**2
        dy = (self.grf_grid[:, 1] - y)**2
        dd = dx + dy
        ind = np.argmin(dd)
        return ind

    def check_prior(self):
        plt.scatter(self.grf_grid[:, 1], self.grf_grid[:, 0], c=self.mu_prior,
                    cmap=get_cmap("BrBG", 10), s=150, vmin=20, vmax=30)
        plt.colorbar()
        plt.show()

    def check_update(self):
        self.update_grf_model(10, 100)
        plt.figure(figsize=(20, 10))
        plt.subplot(121)
        plt.scatter(self.grf_grid[:, 1], self.grf_grid[:, 0], c=self.mu_cond,
                    cmap=get_cmap("BrBG", 10), s=150, vmin=20, vmax=30)
        plt.colorbar()
        plt.subplot(122)
        plt.scatter(self.grf_grid[:, 1], self.grf_grid[:, 0], c=np.diag(self.Sigma_cond),
                    cmap=get_cmap("RdBu", 10), s=150)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    grf = GRF()
    # grf.check_prior()
    # grf.check_update()


