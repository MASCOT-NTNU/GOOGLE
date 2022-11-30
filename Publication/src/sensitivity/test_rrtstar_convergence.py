"""
This script tests the convergence rate of rrt star in different cost field.
"""
from Planner.RRTSCV.RRTStarCV import RRTStarCV
from Config import Config
from Visualiser.TreePlotter import TreePlotter
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap


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

    def test_get_new_location(self):
        loc_now = np.array([1000, -1000])
        # loc_now = self.config.get_loc_start()
        # loc_end = self.config.get_loc_end()
        loc_end = np.array([4000, 200])
        wp = self.rrtstar.get_next_waypoint(loc_now, loc_end)
        print(wp)
        nodes = self.rrtstar.get_tree_nodes()
        traj = self.rrtstar.get_trajectory()
        self.tp.update_trees(nodes)
        plt.figure(figsize=(15, 12))
        cv = self.cv.get_cost_field()
        plotf_vector(self.grid[:, 1], self.grid[:, 0], cv, xlabel='x', ylabel='y', title='RRTCV', cbar_title="Cost",
                     cmap=get_cmap("RdBu", 10), vmin=0, vmax=4, alpha=.3)
        self.tp.plot_tree()
        plt.plot(traj[:, 1], traj[:, 0], 'k-', linewidth=10)
        # plt.plot(wp[0], wp[1], 'b*', markersize=20)
        # plt.plot(loc_now[0], loc_now[1], 'c.', markersize=10)
        plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'r-.')
        plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'r-.')

        plt.plot(loc_now[1], loc_now[0], 'r.', markersize=20)
        plt.plot(loc_end[1], loc_end[0], 'k*', markersize=20)
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.savefig(os.getcwd() + "/../../fig/trees/rrtcv.png")
        plt.show()


if __name__ == "__main__":
    r = RRTStarCase()

