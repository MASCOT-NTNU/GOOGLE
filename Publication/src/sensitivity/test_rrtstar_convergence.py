"""
This script tests the convergence rate of rrt star in different cost field.
"""
from Planner.RRTSCV.RRTStarCV import RRTStarCV
from Config import Config
from Visualiser.TreePlotter import TreePlotter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap
from numpy import testing
import time
import os


class RRTStarCase:

    def __init__(self):
        self.config = Config()
        self.rrtstar = RRTStarCV()
        self.tp = TreePlotter()
        self.cv = self.rrtstar.get_CostValley()
        self.field = self.cv.get_field()
        self.grid = self.cv.get_grid()
        self.polygon_border = self.config.get_polygon_border()
        self.polygon_obstacle = self.config.get_polygon_obstacle()

        self.xmin, self.ymin = map(np.amin, [self.polygon_border[:, 0], self.polygon_border[:, 1]])
        self.xmax, self.ymax = map(np.amax, [self.polygon_border[:, 0], self.polygon_border[:, 1]])

        self.figpath = os.getcwd() + "/../../fig/Sim_2DNidelva/"

    def test_updating_parameters(self) -> None:
        # c1: step size
        value = 120.458
        self.rrtstar.set_stepsize(value)
        testing.assert_equal(value, self.rrtstar.get_stepsize())

        # c2: number of iterations
        value = 12000
        self.rrtstar.set_max_expansion_iteraions(value)
        testing.assert_equal(value, self.rrtstar.get_max_expansion_iteraions())

        # c3: goal sampling rates
        value = .01123
        self.rrtstar.set_goal_sampling_rate(value)
        testing.assert_equal(value, self.rrtstar.get_goal_sampling_rate())

        # c4: neighbour radius
        value = 123.45
        self.rrtstar.set_rrtstar_neighbour_radius(value)
        testing.assert_equal(value, self.rrtstar.get_rrtstar_neighbour_radius())

        # c5: home radius
        value = 130.8
        self.rrtstar.set_home_radius(value)
        testing.assert_equal(value, self.rrtstar.get_home_radius())

    def check_cost_effects(self) -> None:

        pass

    def check_convergence_for_rrtstar(self):
        # stepsizes = (np.arange(0, .85, .05) + .05) * np.sqrt((self.xmax-self.xmin)**2 + (self.ymax-self.ymin)**2)
        # max_iterations = np.arange(1, 11) * 1000
        # goal_sampling_rates = np.arange(0.01, .8, .025)
        self.distance_traj = []

        cnt = 0
        # for ss in stepsizes:
        for ss in [0]:
        # for itr in max_iterations:
        # for gsr in goal_sampling_rates:
            """ Set up simulation parameters. """
            # self.rrtstar.set_stepsize(ss)
            # self.rrtstar.set_max_expansion_iteraions(itr)
            # self.rrtstar.set_goal_sampling_rate(gsr)

            print(ss)
            # print(gsr)

            self.dist_itr = []

            for i in range(1):
                # print(i)
                t1 = time.time()

                loc_now = np.array([1000, -1000])
                loc_end = np.array([4000, 200])
                wp = self.rrtstar.get_next_waypoint(loc_now, loc_end)
                nodes = self.rrtstar.get_tree_nodes()
                traj = self.rrtstar.get_trajectory()
                self.dist_itr.append(self.get_distance_along_trajectory(traj))

                self.tp.update_trees(nodes)
                plt.figure(figsize=(15, 12))

                # cv = self.cv.get_cost_field()
                # plotf_vector(self.grid[:, 1], self.grid[:, 0], cv, xlabel='x', ylabel='y', title='RRTCV', cbar_title="Cost",
                #              cmap=get_cmap("RdBu", 10), vmin=0, vmax=4, alpha=.3)
                self.tp.plot_tree()

                plt.plot(traj[:, 1], traj[:, 0], 'k-', linewidth=10)
                plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'r-.')
                plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'r-.')
                plt.plot(loc_now[1], loc_now[0], 'r.', markersize=20, label="Starting location")
                plt.plot(loc_end[1], loc_end[0], 'r*', markersize=20, label="Target location")
                plt.xlabel("x")
                plt.ylabel("y")
                plt.title("Stepsize: " + str(ss))
                # plt.title("Iterations: " + str(itr))
                # plt.title("Goal sampling rate: " + str(gsr))
                plt.legend()

                # plt.savefig(self.figpath + "rrtstar/P_goal_sampling_rate/P_{:03d}.png".format(cnt))
                plt.show()
                plt.close("all")
                cnt += 1

                t2 = time.time()
                print("Time consumed per iteration: ", t2 - t1)

            self.distance_traj.append(self.dist_itr)

    def get_distance_along_trajectory(self, traj: np.ndarray) -> float:
        dist = .0
        for i in range(traj.shape[0]-1):
            dist += np.sqrt((traj[i, 0] - traj[i+1, 0])**2 + (traj[i, 1] - traj[i+1, 1])**2)
        return dist


if __name__ == "__main__":
    r = RRTStarCase()
    # r.test_updating_parameters()
    r.check_convergence_for_rrtstar()

#%%
d = np.array(r.distance_traj)



