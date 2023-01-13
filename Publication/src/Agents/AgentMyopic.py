"""
Agent abstract the AUV to conduct the path planning with sense, plan, act philosophy.
"""
from Planner.Myopic2D.Myopic2D import Myopic2D
from Config import Config
from Simulators.CTD import CTD
from Visualiser.AgentPlotMyopic import AgentPlotMyopic
from usr_func.checkfolder import checkfolder
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import numpy as np
import os


class Agent:
    """
    Agent
    """
    def __init__(self, weight_eibv: float = 1., weight_ivr: float = 1., sigma: float = 1., nugget: float = .4,
                 random_seed: int = 1, debug=False, name: str = "Equal") -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        # s0: load parameters
        self.config = Config()

        # s1: set the starting location.
        self.loc_start = self.config.get_loc_start()

        # s2: load CTD
        self.ctd = CTD(loc_start=self.loc_start, random_seed=random_seed, sigma=sigma, nugget=nugget)

        # s3: set up planning strategies
        self.myopic = Myopic2D(self.loc_start, weight_eibv=weight_eibv, weight_ivr=weight_ivr,
                               sigma=sigma, nugget=nugget)
        self.cv = self.myopic.getCostValley()
        self.grf = self.cv.get_grf_model()

        # s4: set up visualiser
        figpath = os.getcwd() + "/../../../../OneDrive - NTNU/MASCOT_PhD/Projects" \
                                "/GOOGLE/Docs/fig/Sim_2DNidelva/Simulator/Myopic/" + name + "/"
        checkfolder(figpath)
        self.ap = AgentPlotMyopic(self, figpath)
        self.debug = debug
        self.counter = 0

        # s5: set up monitoring metrics
        self.ibv = []
        self.vr = []
        self.rmse = []
        self.threshold = self.grf.get_threshold()
        self.mu_truth = self.ctd.get_ground_truth()

    def run(self, num_steps: int = 5) -> None:
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """
        # start logging the data.
        self.trajectory = np.empty([0, 2])

        for i in range(num_steps):
            print("Step: ", i)
            # s0: update simulation data
            ibv, vr, rmse = self.update_metrics()
            self.ibv.append(ibv)
            self.vr.append(vr)
            self.rmse.append(rmse)

            if self.debug:
                self.ap.plot_agent()

            # p1: parallel move AUV to the first location
            wp_now = self.myopic.get_current_waypoint()
            self.trajectory = np.append(self.trajectory, wp_now.reshape(1, -1), axis=0)

            # s2: obtain CTD data
            ctd_data = self.ctd.get_ctd_data(wp_now)

            # s3: update pioneer waypoint
            self.myopic.update_next_waypoint(ctd_data)

            self.counter += 1

    def update_metrics(self) -> tuple:
        mu = self.grf.get_mu()
        sigma_diag = np.diag(self.grf.get_covariance_matrix())
        ibv = self.get_ibv(self.threshold, mu, sigma_diag)
        rmse = mean_squared_error(self.mu_truth, mu, squared=False)
        vr = np.sum(sigma_diag)
        return ibv, vr, rmse

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


if __name__ == "__main__":
    a = Agent(1.9, .1, 0, True)
    a.run(20)
    a



