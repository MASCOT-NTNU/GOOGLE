"""
Simulator for studying the properties of RRT star with Cost Valley.
"""
from Planner.RRTSCV.RRTStarCV import RRTStarCV
from Config import Config
from Visualiser.TreePlotter import TreePlotter
from usr_func.checkfolder import checkfolder
from Visualiser.Visualiser import plotf_vector
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import numpy as np
import os
from time import time


class Simulator:

    def __init__(self, weight_eibv: float = 1., weight_ivr: float = 1.,
                 case: str = "Equal", debug: bool = False) -> None:
        self.weight_eibv = weight_eibv
        self.weight_ivr = weight_ivr
        self.case = case
        self.debug = debug

        self.config = Config()
        self.rrtstar = RRTStarCV()
        self.tp = TreePlotter()
        self.cv = self.rrtstar.get_CostValley()
        self.cv.set_weight_eibv(self.weight_eibv)
        self.cv.set_weight_ivr(self.weight_ivr)
        self.cv.update_cost_valley()  # update right after the weights are refreshed.
        self.grf = self.cv.get_grf_model()
        self.field = self.cv.get_field()
        self.grid = self.field.get_grid()
        self.polygon_border = self.config.get_polygon_border()
        self.polygon_obstacle = self.config.get_polygon_obstacle()

        # simulation parameter set
        self.stepsizes = np.arange(60, 601, 60)
        self.max_iterations = np.arange(1000, 6000, 500)
        self.d_traj = np.zeros([len(self.stepsizes), len(self.max_iterations)])  # distance
        self.t_traj = np.zeros_like(self.d_traj)  # time
        self.c_traj = np.zeros_like(self.d_traj)  # cost

        self.figpath = os.getcwd() + "/../../fig/Sim_2DNidelva/rrtstar/Cases/"

    def run(self, value: int = 0) -> tuple:
        print("weight_EIBV: ", self.cv.get_eibv_weight(), " weight IVR: ", self.cv.get_ivr_weight())

        loc_now = np.array([1500, -2000])
        loc_end = self.cv.get_minimum_cost_location()
        counter = 0
        for i in range(len(self.stepsizes)):
            for j in range(len(self.max_iterations)):
                t1 = time()
                print("Iteration: {:d}/{:d}".format(counter, len(self.stepsizes) * len(self.max_iterations)))

                self.rrtstar.set_stepsize(self.stepsizes[i])
                self.rrtstar.set_max_expansion_iteraions(self.max_iterations[j])
                # print("stepsize: ", self.rrtstar.get_stepsize())
                # print("max iteration: ", self.rrtstar.get_max_expansion_iteraions())

                wp = self.rrtstar.get_next_waypoint(loc_now, loc_end)

                self.d_traj[i, j] = self.rrtstar.get_distance_along_trajectory()
                self.t_traj[i, j] = self.rrtstar.T.get_t_total()
                self.c_traj[i, j] = self.rrtstar.get_cost_along_trajectory()

                if self.debug:
                    nodes = self.rrtstar.get_tree_nodes()
                    traj = self.rrtstar.get_trajectory()
                    self.tp.update_trees(nodes)

                    plt.figure(figsize=(15, 12))
                    cv = self.cv.get_cost_field()
                    plotf_vector(self.grid[:, 1], self.grid[:, 0], cv, xlabel='East', ylabel='North', title='RRTCV',
                                 cbar_title="Cost", cmap=get_cmap("RdBu", 10))
                    self.tp.plot_tree()
                    plt.plot(traj[:, 1], traj[:, 0], 'k-', linewidth=10)
                    plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'r-.')
                    plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'r-.')

                    plt.plot(loc_now[1], loc_now[0], 'r.', markersize=20)
                    plt.plot(loc_end[1], loc_end[0], 'k*', markersize=20)
                    plt.xlabel("x")
                    plt.ylabel("y")
                    plt.savefig(self.figpath + "P_stepsize_{:d}_maxiter_{:d}.png".format(self.stepsizes[i],
                                                                                         self.max_iterations[j]))
                    # plt.show()
                    plt.close("all")
                else:
                    pass
                t2 = time()
                print("Time consumed: ", t2 - t1)

                counter += 1
        return self.d_traj, self.c_traj, self.t_traj


if __name__ == "__main__":
    s = Simulator(debug=False)
    s.run()
