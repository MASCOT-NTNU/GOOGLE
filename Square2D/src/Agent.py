"""
Agent object abstract the entire adaptive agent by wrapping all the other components together inside the class.
It handles the procedure of the execution by integrating all essential modules and expand its functionalities.

The goal of the agent is to conduct the autonomous sampling operation by using the following procedure:
- Sense
- Plan
- Act

Sense refers to the in-situ measurements. Once the agent obtains the sampled values in the field. Then it can plan based
on the updated knowledge for the field. Therefore, it can act according to the planned manoeuvres.
"""
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.cm import get_cmap

from RRTStar.RRTStar import RRTStar
from RRTStar.StraightLinePathPlanner import StraightLinePathPlanner
from CostValley.CostValley import CostValley
from Field import Field
from AUVSimulator.AUVSimulator import AUVSimulator
from Visualiser.TreePlotter import TreePlotter
import numpy as np
import time
import os


class Agent:

    __loc_start = np.array([.01, .01])
    __loc_end = np.array([.01, .99])

    __loc_min_cv = np.array([.01, .01])
    __loc_next = np.array([.01, .01])

    __NUM_STEP = 10
    __distance_tolerance = .05
    __counter = 0

    traj = np.empty([0, 2])

    def __init__(self) -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        # s1: setup planner.
        self.rrtstar = RRTStar()
        self.slpp = StraightLinePathPlanner()

        # s2: setup cost valley and kernel.
        self.cv = CostValley()
        self.Budget = self.cv.get_Budget()
        self.grf = self.cv.get_grf_model()
        self.grid = self.grf.grid

        # s2: setup AUV simulator.
        self.auv = AUVSimulator()

        # s3: setup Visualiser.
        self.tp = TreePlotter()
        # self.visualiser = Visualiser(self, figpath=os.getcwd() + "/../fig/Myopic3D/")

    def run(self):
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """

        loc = self.__loc_start

        for i in range(self.__NUM_STEP):
            print(i)
            # s0: append location
            self.traj = np.append(self.traj, loc.reshape(1, -1), axis=0)

            # s1: move to location
            self.auv.move_to_location(loc)

            # s2: sample
            ctd_data = self.auv.get_ctd_data()

            # s3: update grf
            self.grf.assimilate_data(ctd_data)

            # s4: update cost valley
            self.cv.update_cost_valley(loc)

            # s5: get minimum cost location.
            self.__loc_min_cv = self.cv.get_minimum_cost_location()

            # s6: plot trees
            self.tp.update_trees(self.rrtstar.get_nodes())
            self.tp.plot_tree()
            # traj = self.rrtstar.get_trajectory()
            plt.plot(loc[0], loc[1], 'y*', markersize=20)
            plt.plot(self.traj[:, 0], self.traj[:, 1], 'k.-')
            # plt.plot(traj[:, 0], traj[:, 1], 'r-')

            # s7: plan one step based on cost valley and rrt*
            if not self.Budget.get_go_home_alert():
                loc = self.rrtstar.get_next_waypoint(loc, self.__loc_min_cv, self.cv)
            else:
                loc = self.slpp.get_waypoint_from_straight_line(loc, self.__loc_end)

            # print("loc: ", loc)
            plt.plot(loc[0], loc[1], 'b*', markersize=20)
            plt.scatter(self.grid[:, 0], self.grid[:, 1], c=self.cv.get_cost_valley(),
                        cmap=get_cmap("BrBG", 10), vmin=0, vmax=4, alpha=.5)
            plt.colorbar()
            plt.savefig("/Users/yaoling/Downloads/trees/P_{:03d}.png".format(i))
            plt.close("all")

            # s8: check arrival
            dist = np.sqrt((loc[0] - self.__loc_end[0])**2 +
                           (loc[1] - self.__loc_end[1])**2)
            if dist <= self.__distance_tolerance:
                break


if __name__ == "__main__":
    a = Agent()
    a.run()


