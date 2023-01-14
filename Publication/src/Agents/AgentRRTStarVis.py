"""
Agent abstract the AUV to conduct the path planning with sense, plan, act philosophy.
"""
from Planner.Planner import Planner
from Config import Config
from Simulators.CTD import CTD
from Visualiser.AgentPlotRRTStarVis import AgentPlotRRTStar
from usr_func.checkfolder import checkfolder
from scipy.stats import norm
from sklearn.metrics import mean_squared_error
import numpy as np
import os


class Agent:
    """
    Agent
    """
    def __init__(self, weight_eibv: float = 1., weight_ivr: float = 1., sigma: float = .1,
                 nugget: float = .01, random_seed: int = 1, debug=False, name: str = "Equal",
                 budget_mode: bool = False) -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        # s0: load parameters
        self.config = Config()
        self.__budget_mode = budget_mode

        # s1: set the starting location.
        self.loc_start = self.config.get_loc_start()

        # s2: load CTD
        self.ctd = CTD(loc_start=self.loc_start, random_seed=random_seed, sigma=sigma, nugget=nugget)

        # s3: set up planning strategies
        self.planner = Planner(self.loc_start, weight_eibv=weight_eibv, weight_ivr=weight_ivr,
                               sigma=sigma, nugget=nugget, budget_mode=budget_mode)
        self.rrtstarcv = self.planner.get_rrtstarcv()
        self.cv = self.rrtstarcv.get_CostValley()
        self.grf = self.cv.get_grf_model()

        # s4: set up visualiser
        self.debug = debug
        figpath = os.getcwd() + "/../../../../OneDrive - NTNU/MASCOT_PhD/Projects" \
                                "/GOOGLE/Docs/fig/Sim_2DNidelva/Simulator/RRT/"
        # print("figpath: ", figpath)
        checkfolder(figpath)
        self.ap = AgentPlotRRTStar(self, figpath)
        self.counter = 0

        # s5: set up monitoring metrics
        self.threshold = self.grf.get_threshold()

    def run(self, num_steps: int = 5) -> None:
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """
        # start logging the data.
        self.trajectory = np.empty([0, 2])

        for i in range(num_steps):
            print("Step: ", i)
            if self.debug:
                self.ap.plot_agent()

            """
            Add RRTStar planning strategies
            """
            # s1: update the waypoint trackers
            self.planner.update_planning_trackers()
            wp_now = self.planner.get_current_waypoint()
            self.trajectory = np.append(self.trajectory, wp_now.reshape(1, -1), axis=0)

            # s2: obtain CTD data
            ctd_data = self.ctd.get_ctd_data(wp_now)

            # s3: update pioneer waypoint
            self.planner.update_pioneer_waypoint(ctd_data=ctd_data)

            self.counter += 1


if __name__ == "__main__":
    a = Agent(1.9, .1, 0, True)
    a.run(20)
    a



