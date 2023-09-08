"""
Agent abstract the AUV to conduct the path planning with sense, plan, act philosophy.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-09-02
"""
from Planner.Myopic2D.Myopic2D import Myopic2D
from AUVSimulator.AUVSimulator import AUVSimulator
from Visualiser.AgentPlotMyopic import AgentPlotMyopic
from Config import Config
from usr_func.checkfolder import checkfolder
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance
import numpy as np
import os
from time import time


class Agent:
    def __init__(self, weight_eibv: float = 1., weight_ivr: float = 1., random_seed: int = 1,
                 debug=False, name: str = "Equal") -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        self.__config = Config()
        self.__num_steps = self.__config.get_num_steps()
        self.debug = debug
        self.counter = 0

        # s0: load AUVSimulator
        self.auv = AUVSimulator(random_seed=random_seed)
        # self.ctd = CTD(loc_start=self.loc_start, random_seed=random_seed, sigma=sigma, nugget=nugget)

        # s1: set up planning strategies
        self.myopic = Myopic2D(weight_eibv=weight_eibv, weight_ivr=weight_ivr)
        self.cv = self.myopic.getCostValley()
        self.grf = self.cv.get_grf_model()
        self.threshold = self.grf.get_threshold()

        # s2: set up visualiser
        figpath = os.getcwd() + "/../../../../OneDrive - NTNU/MASCOT_PhD/Projects" \
                                "/GOOGLE/Docs/fig/Sim_2DNidelva/Simulator/Myopic/" + name + "/"
        checkfolder(figpath)
        self.ap = AgentPlotMyopic(self, figpath)

    def run(self) -> None:
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """
        # start logging the data.
        self.trajectory = np.empty([0, 2])
        N = self.grf.grid.shape[0]

        self.ibv = np.empty([self.__num_steps, ])
        self.rmse = np.empty([self.__num_steps, ])
        self.vr = np.empty([self.__num_steps, ])
        self.mu_data = np.empty([self.__num_steps, N])
        if self.__num_steps > 15:
            self.cov_data = np.empty([self.__num_steps // 15, N, N])
        else:
            self.cov_data = np.empty([1, N, N])
        self.sigma_data = np.empty([self.__num_steps, N])
        self.mu_truth_data = np.empty([self.__num_steps, N])

        t0 = time()
        for i in range(self.__num_steps):
            print(" STEP: {} / {}".format(i, self.__num_steps),
                  " Percentage: ", i / self.__num_steps * 100, "%",
                  " Time remaining: ", (time() - t0) * (self.__num_steps - i) / 60, " min")
            t0 = time()
            # s0: update simulation data and save the updated data.
            mu, cov, sigma_diag, mu_truth, ibv, rmse, vr = self.update_metrics()
            self.ibv[i] = ibv
            self.rmse[i] = rmse
            self.vr[i] = vr
            self.mu_data[i, :] = mu.flatten()
            if i % 15 == 0:
                self.cov_data[i // 15, :, :] = cov
            self.sigma_data[i, :] = sigma_diag.flatten()
            self.mu_truth_data[i, :] = mu_truth.flatten()

            if self.debug:
                self.ap.plot_agent()

            # p1: parallel move AUV to the first location
            wp_next = self.myopic.get_next_waypoint()
            self.trajectory = np.append(self.trajectory, wp_next.reshape(1, -1), axis=0)

            # s2: obtain CTD data
            self.auv.move_to_location(wp_next)
            ctd_data = self.auv.get_ctd_data()

            # s3: update pioneer waypoint
            self.myopic.update_next_waypoint(ctd_data)

            self.counter += 1

    def update_metrics(self) -> tuple:
        mu = self.grf.get_mu()
        cov = self.grf.get_covariance_matrix()
        sigma_diag = np.diag(cov)
        mu_truth = self.auv.ctd.get_salinity_at_dt_loc(dt=0, loc=self.grf.grid)
        ibv = self.get_ibv(self.threshold, mu, sigma_diag)
        rmse = mean_squared_error(mu_truth, mu, squared=False)
        vr = np.sum(sigma_diag)
        return mu, cov, sigma_diag, mu_truth, ibv, rmse, vr

    def get_ibv(self, threshold, mu, sigma_diag) -> np.ndarray:
        """ !!! Be careful with dimensions, it can lead to serious problems.
        !!! Be careful with standard deviation is not variance, so it does not cause significant issues tho.
        :param mu: (n, ) dimension
        :param sigma_diag: (n, ) dimension
        :return:
        """
        p = norm.cdf(threshold, mu.flatten(), np.sqrt(sigma_diag))
        bv = p * (1 - p)
        ibv = np.sum(bv)
        return ibv

    def get_metrics(self) -> tuple:
        """
        Return the metrics calculated during the simulation.
        """
        return (self.trajectory, self.ibv, self.rmse, self.vr, self.mu_data,
                self.cov_data, self.sigma_data, self.mu_truth_data)


if __name__ == "__main__":
    a = Agent(1.9, .1, 0, True)
    a.run(20)
    a



