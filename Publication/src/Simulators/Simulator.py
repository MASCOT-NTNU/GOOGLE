"""
Simulator generates the simulation result using three different methodolgies.
- EIBV dominant
- IVR dominant
- Equal weights
"""
from Planner.Planner import Planner
from Simulators.CTD import CTD
from Simulators.Log import Log
from Config import Config
import numpy as np
from time import time
import matplotlib.pyplot as plt
from Visualiser.Visualiser import plotf_vector
from matplotlib.cm import get_cmap
from usr_func.checkfolder import checkfolder
import os


class Simulator:
    """
    Simulator
    """
    def __init__(self, debug: bool = False) -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        self.debug = debug

        # s0: load parameters
        self.config = Config()

        # s1: set the starting location.
        self.loc_start = np.array([1200, -1500])

        # s2: set up ground truth.
        self.ctd = CTD(self.loc_start)

    def set_weights(self, weight_eibv: float = 1., weight_ivr: float = 1.):
        if weight_eibv > weight_ivr:
            self.case = "EIBV"
        elif weight_eibv < weight_ivr:
            self.case = "IVR"
        else:
            self.case = "EQUAL"

        # s0: set the planner according to their weight set.
        self.planner = Planner(self.loc_start)
        self.log = Log(ctd=self.ctd)
        self.rrtstar = self.planner.get_rrtstarcv()
        self.cv = self.rrtstar.get_CostValley()
        self.cv.set_weight_eibv(weight_eibv)
        self.cv.set_weight_ivr(weight_ivr)
        self.cv.update_cost_valley()
        self.grf = self.cv.get_grf_model()

        if self.debug:
            self.field = self.cv.get_field()
            self.grid = self.field.get_grid()
            self.polygon_border = self.config.get_polygon_border()
            self.polygon_obstacle = self.config.get_polygon_obstacle()
            self.figpath = os.getcwd() + "/../../fig/Sim_2DNidelva/Simulator/" + self.case + "/"
            checkfolder(self.figpath)

            plt.figure(figsize=(15, 12))
            plotf_vector(self.grid[:, 1], self.grid[:, 0], self.grf.get_mu(), xlabel='East', ylabel='North',
                         title='Prior', cbar_title="Salinity", cmap=get_cmap("RdBu", 10), vmin=10, vmax=33, stepsize=1.5)
            plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'r-.')
            plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'r-.')
            plt.xlabel("East")
            plt.ylabel("North")
            plt.savefig(self.figpath + "Prior_" + self.case + ".png")
            plt.show()

            plt.figure(figsize=(15, 12))
            plotf_vector(self.grid[:, 1], self.grid[:, 0], self.ctd.get_ground_truth(), xlabel='East', ylabel='North',
                         title='GroundTruth', cbar_title="Salinity", cmap=get_cmap("RdBu", 10), vmin=10, vmax=33,
                         stepsize=1.5)
            plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'r-.')
            plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'r-.')
            plt.xlabel("East")
            plt.ylabel("North")
            plt.savefig(self.figpath + "Truth_" + self.case + ".png")
            plt.show()

    def run_simulator(self, num_steps: int = 5) -> tuple:
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """
        trajectory = np.empty([0, 2])
        trajectory = np.append(trajectory, self.loc_start.reshape(1, -1), axis=0)
        self.log.append_log(self.grf)
        
        t1 = time()

        """ Debug section. """
        # rmse = []
        # from sklearn.metrics import mean_squared_error
        # mu_truth = self.ctd.get_ground_truth()
        # wps = np.array([[2000, -2000],
        #                 [2200, -1800],
        #                 [2400, -1600],
        #                 [2600, -1400],
        #                 [2800, -1200],
                        # [3000, -1000],
                        # [2900, -1100],
                        # [2800, -1300],
                        # [2700, -1200],
                        # [1500, -800]
                        # ])
        """ End """

        for i in range(num_steps):
        # for i in range(len(wps)):
            t2 = time()
            print("Step: ", i, " takes ", t2 - t1, " seconds. ")
            t1 = time()

            """ plotting seciton. """
            if self.debug:
                plt.figure(figsize=(15, 12))
                cost_valley = self.cv.get_cost_field()
                plotf_vector(self.grid[:, 1], self.grid[:, 0], cost_valley, xlabel='East', ylabel='North', title='RRTCV',
                             cbar_title="Cost", cmap=get_cmap("RdBu", 10), vmin=0, vmax=2.2, stepsize=.25)
                if len(trajectory) > 0:
                    plt.plot(trajectory[:, 1], trajectory[:, 0], 'k.-')
                plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'r-.')
                plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'r-.')
                plt.xlabel("East")
                plt.ylabel("North")
                plt.savefig(self.figpath + "P_{:03d}.png".format(i))
                plt.close("all")

            self.planner.update_planning_trackers()

            """ Debug check rmse """
            # rmse.append(mean_squared_error(mu_truth, self.grf.get_mu(), squared=False))
            # wp_now = wps[i, :]
            # ctdd = self.ctd.get_ctd_data(wp_now)
            # print("CTD: ", ctdd)
            # self.grf.assimilate_data(ctdd)
            """ End """

            # p1: parallel move AUV to the first location
            wp_now = self.planner.get_current_waypoint()
            trajectory = np.append(trajectory, wp_now.reshape(1, -1), axis=0)

            # s2: obtain CTD data
            ctd_data = self.ctd.get_ctd_data(wp_now)

            # s3: update pioneer waypoint
            t1 = time()
            # print("CTD: ", ctd_data)
            self.planner.update_pioneer_waypoint(ctd_data)
            t2 = time()
            print("Update pioneer waypoint takes: ", t2 - t1)

            # s4: update simulation data
            self.log.append_log(self.grf)

        # plt.plot(rmse)
        # plt.title("RMSE in")
        # plt.show()

        return trajectory, self.log


if __name__ == "__main__":
    s = Simulator(debug=True)
    s.set_weights(1.9, .1)
    t1, l1 = s.run_simulator(num_steps=10)
    s.set_weights(1., 1.)
    t2, l2 = s.run_simulator(num_steps=10)
    s.set_weights(.1, 1.9)
    t3, l3 = s.run_simulator(num_steps=10)

    t1, l1
