"""
Simulator handles the simulation task.
- store essential data during simulation.
- accept different parameters governing the simulation.
"""
from Planner.Planner import Planner
from Simulators.CTD import CTD
from Config import Config
import numpy as np
from time import time
import matplotlib.pyplot as plt
from Visualiser.Visualiser import plotf_vector
from matplotlib.cm import get_cmap
from usr_func.checkfolder import checkfolder
import os
print("hello")

#%%

class Simulator:
    """
    Simulator
    """
    def __init__(self, weight_eibv: float = 1., weight_ivr: float = 1.,
                 case: str = "Equal", debug: bool = False) -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        # s0: load parameters
        self.weight_eibv = weight_eibv
        self.weight_ivr = weight_ivr
        self.case = case
        self.debug = debug

        # s1: set up planner.
        self.loc_start = np.array([1200, -1500])
        self.planner = Planner(self.loc_start)

        # s2: set up ground truth
        self.ctd = CTD()

        # s4: set up data storage
        # self.traj_sim = np.empty([0, self.num_steps+1, 2])

        self.rrtstar = self.planner.get_rrtstarcv()
        self.cv = self.rrtstar.get_CostValley()
        self.cv.set_weight_eibv(self.weight_eibv)
        self.cv.set_weight_ivr(self.weight_ivr)
        self.cv.update_cost_valley()  # update right after the weights are refreshed.

        if self.debug:
            self.grf = self.cv.get_grf_model()
            self.field = self.cv.get_field()
            self.grid = self.field.get_grid()
            self.config = Config()
            self.polygon_border = self.config.get_polygon_border()
            self.polygon_obstacle = self.config.get_polygon_obstacle()
            self.figpath = os.getcwd() + "/../../fig/Sim_2DNidelva/Simulator/"
            checkfolder(self.figpath)

    def run_simulator(self, num_steps: int = 5) -> np.ndarray:
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """
        self.trajectory = np.empty([0, 2])
        self.trajectory = np.append(self.trajectory, self.loc_start.reshape(1, -1), axis=0)

        t1 = time()
        for i in range(num_steps):
            t2 = time()
            print("Step: ", i, " takes ", t2 - t1, " seconds. ")
            t1 = time()

            """ plotting seciton. """
            if self.debug:
                plt.figure(figsize=(15, 12))
                cv = self.cv.get_cost_field()
                plotf_vector(self.grid[:, 1], self.grid[:, 0], cv, xlabel='East', ylabel='North', title='RRTCV',
                             cbar_title="Cost", cmap=get_cmap("RdBu", 10), vmin=0, vmax=2.2, stepsize=.25)
                if len(self.trajectory) > 0:
                    plt.plot(self.trajectory[:, 1], self.trajectory[:, 0], 'k.-')
                plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'r-.')
                plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'r-.')
                plt.xlabel("East")
                plt.ylabel("North")
                plt.savefig(self.figpath + "P_{:03d}.png".format(i))
                plt.close("all")

            self.planner.update_planning_trackers()

            # p1: parallel move AUV to the first location
            wp_now = self.planner.get_current_waypoint()
            self.trajectory = np.append(self.trajectory, wp_now.reshape(1, -1), axis=0)

            # s2: obtain CTD data
            ctd_data = self.ctd.get_ctd_data(wp_now)

            # s3: update pioneer waypoint
            # t1 = time.time()
            self.planner.update_pioneer_waypoint(ctd_data)
            # t2 = time.time()
            # print("Update pioneer waypoint takes: ", t2 - t1)

        return self.trajectory


if __name__ == "__main__":
    s = Simulator(weight_eibv=1.9, weight_ivr=.1, case="equal")
    s.run_simulator()
    # s.run_replicates()


