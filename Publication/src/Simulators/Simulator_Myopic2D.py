"""
Simulator generates the simulation result using three different methodologies.
- EIBV dominant
- IVR dominant
- Equal weights
"""
from Planner.Myopic2D.Myopic2D import Myopic2D
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

    def run_simulator(self, num_steps: int = 5) -> tuple:
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """
        w_eibv = [1.9, 1., .1]
        w_ivr = [.1, 1., 1.9]

        traj_sim = np.empty([0, num_steps+1, 2])
        ibv = np.empty([0, num_steps+1])
        rmse = np.empty([0, num_steps+1])
        vr = np.empty([0, num_steps+1])

        for k in range(len(w_eibv)):

            weight_eibv = w_eibv[k]
            weight_ivr = w_ivr[k]
            print("Weight_EIBV: ", weight_eibv, "Weight_IVR: ", weight_ivr)

            if weight_eibv > weight_ivr:
                case = "EIBV"
            elif weight_eibv < weight_ivr:
                case = "IVR"
            else:
                case = "EQUAL"

            # s0: set the planner according to their weight set.
            myopic = Myopic2D(self.loc_start)
            log = Log(ctd=self.ctd)
            cv = myopic.getCostValley()
            cv.set_weight_eibv(weight_eibv)
            cv.set_weight_ivr(weight_ivr)
            cv.update_cost_valley_for_locations(self.loc_start)
            grf = cv.get_grf_model()

            if self.debug:
                field = cv.get_field()
                grid = field.get_grid()
                polygon_border = self.config.get_polygon_border()
                polygon_obstacle = self.config.get_polygon_obstacle()
                figpath = os.getcwd() + "/../../fig/Sim_2DNidelva/Simulator/Myopic/" + case + "/"
                checkfolder(figpath)

                plt.figure(figsize=(15, 12))
                plotf_vector(grid[:, 1], grid[:, 0], grf.get_mu(), xlabel='East', ylabel='North',
                             title='Prior', cbar_title="Salinity", cmap=get_cmap("RdBu", 10), vmin=10, vmax=33, stepsize=1.5)
                plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'r-.')
                plt.plot(polygon_obstacle[:, 1], polygon_obstacle[:, 0], 'r-.')
                plt.xlabel("East")
                plt.ylabel("North")
                plt.savefig(figpath + "./../Prior_" + case + ".png")
                plt.show()

                plt.figure(figsize=(15, 12))
                plotf_vector(grid[:, 1], grid[:, 0], self.ctd.get_ground_truth(), xlabel='East', ylabel='North',
                             title='GroundTruth', cbar_title="Salinity", cmap=get_cmap("RdBu", 10), vmin=10, vmax=33,
                             stepsize=1.5)
                plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'r-.')
                plt.plot(polygon_obstacle[:, 1], polygon_obstacle[:, 0], 'r-.')
                plt.xlabel("East")
                plt.ylabel("North")
                plt.savefig(figpath + "./../Truth_" + case + ".png")
                plt.show()

            # start logging the data.
            trajectory = np.empty([0, 2])
            trajectory = np.append(trajectory, self.loc_start.reshape(1, -1), axis=0)
            log.append_log(grf)
            print("RMSE: ", log.rmse)

            t1 = time()

            for i in range(num_steps):
            # for i in range(len(wps)):
                t2 = time()
                print("Step: ", i, " takes ", t2 - t1, " seconds. ")
                t1 = time()

                """ plotting seciton. """
                if self.debug:
                    plt.figure(figsize=(15, 12))
                    cv.update_cost_valley()
                    cost_valley = cv.get_cost_field()
                    plotf_vector(grid[:, 1], grid[:, 0], cost_valley, xlabel='East', ylabel='North', title='RRTCV',
                                 cbar_title="Cost", cmap=get_cmap("RdBu", 10), vmin=0, vmax=2.2, stepsize=.25)
                    if len(trajectory) > 0:
                        plt.plot(trajectory[:, 1], trajectory[:, 0], 'k.-')
                    plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'r-.')
                    plt.plot(polygon_obstacle[:, 1], polygon_obstacle[:, 0], 'r-.')
                    plt.xlabel("East")
                    plt.ylabel("North")
                    plt.savefig(figpath + "P_{:03d}.png".format(i))
                    plt.close("all")

                # p1: parallel move AUV to the first location
                wp_now = myopic.get_current_waypoint()
                trajectory = np.append(trajectory, wp_now.reshape(1, -1), axis=0)

                # s2: obtain CTD data
                ctd_data = self.ctd.get_ctd_data(wp_now)

                # s3: update pioneer waypoint
                t1 = time()
                # print("CTD: ", ctd_data)
                myopic.update_next_waypoint(ctd_data)
                t2 = time()
                print("Update pioneer waypoint takes: ", t2 - t1)

                # s4: update simulation data
                log.append_log(grf)

            traj_sim = np.append(traj_sim, trajectory.reshape(1, num_steps + 1, 2), axis=0)
            rmse = np.append(rmse, np.array(log.rmse).reshape(1, -1), axis=0)
            ibv = np.append(ibv, np.array(log.ibv).reshape(1, -1), axis=0)
            vr = np.append(vr, np.array(log.vr).reshape(1, -1), axis=0)

        return traj_sim, rmse, ibv, vr


if __name__ == "__main__":
    s = Simulator(debug=True)
    t, r, i, v = s.run_simulator(num_steps=80)
    t, r

