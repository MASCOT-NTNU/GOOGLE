"""
This script builds the cost valley
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-25
"""
import matplotlib.pyplot as plt

from GOOGLE.Simulation_2DNidelva.Config.Config import FILEPATH, THRESHOLD, NUGGET
from GOOGLE.Simulation_2DNidelva.Budget import Budget
import pandas as pd
import numpy as np
from usr_func import get_ibv, normalise, vectorise
import time

# # == Setting
PENALTY_AZIMUTH = 0
# # ==


class CostValley:

    def __init__(self):
        self.load_grf_grid()
        self.budget = Budget()
        self.cost_valley = np.zeros_like(self.grf_grid[:, 0])

    def load_grf_grid(self):
        self.grf_grid = pd.read_csv(FILEPATH+"Config/GRFGrid.csv").to_numpy()
        print("CV1: GRF Grid is loaded successfullyQ")

    def update_cost_valley(self, mu, Sigma, x_current, y_current, x_previous, y_previous):
        self.budget.update_budget(x_current, y_current, x_previous, y_previous)
        self.mu = mu
        self.Sigma = Sigma
        self.x_current = x_current
        self.y_current = y_current
        self.x_azimuth = x_current - x_previous
        self.y_azimuth = y_current - y_previous
        self.vector_azimuth = vectorise([self.x_azimuth, self.y_azimuth])
        self.get_cost_valley()

    def get_cost_valley(self):
        t1 = time.time()
        self.budget.get_budget_field()
        self.get_directional_field()
        self.get_exploration_exploitation_field()
        # self.cost_valley = self.ee_field + self.budget.budget_field
        self.cost_valley = self.azimuth_field + self.ee_field + self.budget.budget_field
        t2 = time.time()
        print("Cost valley takes: ", t2 - t1)

    def get_exploration_exploitation_field(self):
        self.eibv_field = np.zeros_like(self.mu)
        self.vr_field = np.zeros_like(self.mu)
        t1 = time.time()
        for i in range(self.mu.shape[0]):
            self.SF = self.Sigma[:, i].reshape(-1, 1)
            self.MD = 1 / (self.Sigma[i, i] + NUGGET)
            self.VR = self.SF @ self.SF.T * self.MD
            self.SP = self.Sigma - self.VR
            self.sigma_diag = np.diag(self.SP)
            self.eibv_field[i] = get_ibv(self.mu, self.sigma_diag, THRESHOLD)
            self.vr_field[i] = np.sum(np.diag(self.VR))
        t2 = time.time()
        self.ee_field = normalise(self.eibv_field) + 1 - normalise(self.vr_field)
        print("EE field takes: ", t2 - t1)

    def get_directional_field(self):
        t1 = time.time()
        self.azimuth_field = np.zeros_like(self.grf_grid[:, 0])
        dx1 = self.grf_grid[:, 0] - self.x_current
        dy1 = self.grf_grid[:, 1] - self.y_current
        vec1 = np.vstack((dx1, dy1)).T
        res = vec1 @ self.vector_azimuth
        ind = np.where(res < 0)[0]
        self.azimuth_field[ind] = PENALTY_AZIMUTH
        t2 = time.time()
        print("Azimuth field takes: ", t2 - t1)

    def check_cost_valley(self):
        from scipy.spatial.distance import cdist
        from matplotlib.patches import Ellipse
        import math
        from matplotlib.cm import get_cmap
        self.mu = pd.read_csv(FILEPATH+"Config/data_interpolated.csv")['salinity'].to_numpy()
        DM = cdist(self.grf_grid, self.grf_grid)
        eta = 4.5/1600
        Sigma = (1 + eta * DM) * np.exp(-eta * DM)

        xp = 2000
        yp = -2000
        xn = 1990
        yn = -1900
        self.budget.budget_left = 4000
        self.update_cost_valley(self.mu, Sigma, xn, yn, xp, yp)
        # self.get_cost_valley()

        # plt.scatter(self.grf_grid[:, 1], self.grf_grid[:, 0], c=self.mu, cmap=get_cmap("BrBG", 10), vmin=15, vmax=30)
        plt.scatter(self.grf_grid[:, 1], self.grf_grid[:, 0], c=self.cost_valley, cmap=get_cmap("BrBG", 10), vmin=0, vmax=4)
        plt.plot(yn, xn, 'bs')
        plt.plot(yp, xp, 'ro')
        ellipse = Ellipse(xy=(self.budget.y_middle, self.budget.x_middle), width=2*self.budget.ellipse_a,
                          height=2*self.budget.ellipse_b, angle=math.degrees(self.budget.angle),
                          edgecolor='r', fc='None', lw=2)
        plt.gca().add_patch(ellipse)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    t = CostValley()
    t.check_cost_valley()

