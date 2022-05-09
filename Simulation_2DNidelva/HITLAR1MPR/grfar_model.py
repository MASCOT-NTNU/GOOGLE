"""
This script builds the kernel for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-05-06
"""

import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.spatial.distance import cdist
from Config.Config import FILEPATH, NUGGET, CRASH
from usr_func import vectorise


# == Parameters
SIGMA = .6
LATERAL_RANGE = 450
AR1_COEF = .965 # trained based on SINMOD data on May-27-2021
# ==


class GRFAR:

    def __init__(self):
        self.load_grf_grid()
        self.load_prior_mean()
        self.load_ar1_coef()
        self.get_covariance_matrix()
        self.get_simulated_truth()
        self.mu_cond = self.mu_prior
        self.Sigma_cond = self.Sigma_prior
        if not CRASH:
            np.save(FILEPATH + "Backup/mu.npy", self.mu_cond)
            np.save(FILEPATH + "Backup/Sigma.npy", self.Sigma_cond)
            print("mu, Sigma is saved successfully!")
            print("GRF1- is set up successfully!")

    def load_grf_grid(self):
        self.grf_grid = pd.read_csv(FILEPATH+"Config/GRFGrid.csv").to_numpy()
        self.N = self.grf_grid.shape[0]
        print("GRF1: Grid is loaded successfully!")

    def load_prior_mean(self):
        self.beta1, self.beta0 = np.load(FILEPATH + "../../../MAFIA/HITL2reduced/models/Google_coef.npy")
        self.mu_prior = (vectorise(pd.read_csv(FILEPATH + "Config/data_interpolated.csv")['salinity'].to_numpy()) *
                         self.beta1 + self.beta0)
        print("beta1: ", self.beta1)
        print("beta0: ", self.beta0)
        print('mean: ', np.mean(self.mu_prior))
        print("GRF2: Prior mean is loaded successfully!")

    def load_ar1_coef(self):
        self.ar1_coef = AR1_COEF
        print("GRF3: AR1 coef is loaded successfully!")

    def get_covariance_matrix(self):
        self.sigma = SIGMA
        self.eta = 4.5 / LATERAL_RANGE
        self.tau = np.sqrt(NUGGET)
        # self.R = np.diagflat(self.tau ** 2)
        distance_matrix = cdist(self.grf_grid, self.grf_grid)
        self.Sigma_prior = self.sigma ** 2 * (1 + self.eta * distance_matrix) * np.exp(-self.eta * distance_matrix)
        print("GRF4: Covariance matrix is computed successfully!")

    def get_simulated_truth(self):
        self.mu_truth = (self.mu_prior + np.linalg.cholesky(self.Sigma_prior) @
                         np.random.randn(len(self.mu_prior)).reshape(-1, 1))
        print("GRF5: Simulated truth is computed successfully!")

    def update_grfar_model(self, ind_measured=np.array([1, 2]), salinity_measured=vectorise([0, 0]), timestep=0):
        t1 = time.time()
        # propagate
        mt0 = self.mu_prior + self.ar1_coef * (self.mu_cond - self.mu_prior)
        St0 = self.ar1_coef**2 * self.Sigma_cond + (1-self.ar1_coef**2) * self.Sigma_prior
        mts = mt0
        Sts = St0
        for s in range(timestep):
            mts = self.mu_prior + self.ar1_coef * (mts - self.mu_prior)
            Sts = self.ar1_coef**2 * Sts + (1 - self.ar1_coef**2) * self.Sigma_prior
        # update
        msamples = salinity_measured.shape[0]
        F = np.zeros([msamples, self.N])
        for i in range(msamples):
            F[i, ind_measured[i]] = True
        R = np.eye(msamples) * self.tau ** 2
        self.mu_cond = mts + Sts @ F.T @ np.linalg.solve(F @ Sts @ F.T + R, salinity_measured - F @ mts)
        # print("F.T@mts: ", (F@mts).shape)
        self.Sigma_cond = Sts - Sts @ F.T @ np.linalg.solve(F @ Sts @ F.T + R, F @ Sts)
        t2 = time.time()
        print("GRF-AR1 model updates takes: ", t2 - t1)
        np.save(FILEPATH + "Backup/mu.npy", self.mu_cond)
        np.save(FILEPATH + "Backup/Sigma.npy", self.Sigma_cond)
        print("mu, Sigma is saved successfully!")

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
        self.update_grfar_model(np.array([1]), vectorise([30]), timestep=6)
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

        self.update_grfar_model(np.array([100, 20, 500]), vectorise([30, 20, 33]), timestep=2)
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
    grf = GRFAR()
    # grf.check_prior()
    grf.check_update()


