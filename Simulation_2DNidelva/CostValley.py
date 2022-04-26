"""
This script builds the cost valley
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-25
"""
import matplotlib.pyplot as plt

from GOOGLE.Simulation_2DNidelva.Config.Config import FILEPATH, THRESHOLD, NUGGET
import pandas as pd
import numpy as np
from numba import vectorize
from scipy.stats import norm


class CostValley:


    def __init__(self, mu=None, Sigma=None):
        self.load_grf_grid()
        self.mu = mu
        self.Sigma = Sigma
        pass

    def load_grf_grid(self):
        self.grf_grid = pd.read_csv(FILEPATH+"Config/GRFGrid.csv").to_numpy()
        print("CV1: GRF Grid is loaded successfullyQ")

    def update(self):
        pass

    def get_cost_valley(self):
        pass

    def get_exploration_exploitation_field(self):
        ee_field = np.zeros_like(self.mu)
        # F = np.zeros([self.mu.shape[0], 1])
        # F[0] = True
        for i in range(self.mu.shape[0]):
            SF = self.Sigma[:, i].reshape(-1, 1)
            MD = 1 / (self.Sigma[i, i] + NUGGET)
            VR = SF @ SF.T * MD
            SP = self.Sigma - VR
        # R = np.diagflat(NUGGET)
        # A = F.T @ self.Sigma @ F + R
        # b = F.T @ self.Sigma
        # res = fast_solve(A, b)
        # Sr = self.Sigma @ F @ res
        # Su = self.Sigma - Sr
        #
        # vr = np.sum(np.diag(Sr))
        #
        # t1 = time.time()
        # self.cost_eibv = []
        # for i in range(self.coordinates_xyz.shape[0]):
        #     F = getFVector(i, self.coordinates_xyz.shape[0])
        #     self.cost_eibv.append(get_eibv_1d(self.knowledge.threshold, self.knowledge.mu_cond,
        #                                       self.knowledge.Sigma_cond, F, self.R))
        # self.cost_eibv = normalise(np.array(self.cost_eibv))
        # t2 = time.time()
        # print("EIBV field takes: ", t2 - t1)
        #
        # t1 = time.time()
        # self.cost_vr = []
        # for i in range(len(self.knowledge.coordinates_xy)):
        #     ind_F = get_ind_at_location2d_xy(self.knowledge.coordinates_xy,
        #                                      LocationXY(self.knowledge.coordinates_xy[i, 0],
        #                                                 self.knowledge.coordinates_xy[i, 1]))
        #     F = np.zeros([1, self.coordinates_xyz.shape[0]])
        #     F[0, ind_F] = True
        #     self.cost_vr.append(self.get_variance_reduction(self.knowledge.Sigma_cond, F, self.R))
        # self.cost_vr = 1 - normalise(np.array(self.cost_vr))
        # t2 = time.time()
        # print("Variance Reduction field takes: ", t2 - t1)
        #
        # pass





