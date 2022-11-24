from unittest import TestCase
from Planner.RRTSCV.RRTStarCV import RRTStarCV
from Config import Config
import matplotlib.pyplot as plt
import numpy as np
from Visualiser.TreePlotter import TreePlotter
from usr_func.set_resume_state import set_resume_state
# from Visualiser.Visualiser import plotf_vector
from matplotlib.cm import get_cmap
import os


def plotf_vector(x, y, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
                 cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
                 stepsize=None, threshold=None, polygon_border=None,
                 polygon_obstacle=None, xlabel=None, ylabel=None):
    """
    Remember x, y is plotting x, y, thus x along horizonal and y along vertical.
    """
    plt.scatter(x, y, c=values, s=500, cmap=get_cmap("BrBG", 10), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlim([np.amin(x), np.amax(x)])
    plt.ylim([np.amin(y), np.amax(y)])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if np.any(polygon_border):
        plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'k-.', lw=2)
        if np.any(polygon_obstacle):
            for i in range(len(polygon_obstacle)):
                plt.plot(polygon_obstacle[i][:, 1], polygon_obstacle[i][:, 0], 'k-.', lw=2)
    return plt.gca()


class TestRRTStar(TestCase):

    def setUp(self) -> None:
        set_resume_state(False)
        self.config = Config()
        self.rrtstar = RRTStarCV()
        self.tp = TreePlotter()
        self.cv = self.rrtstar.get_CostValley()
        self.field = self.cv.get_field()
        self.grid = self.cv.get_grid()
        self.polygon_border = self.field.get_wgs_polygon_border()

    def test_get_new_location(self):
        loc_now = np.array([6000, 8000])
        # loc_now = self.config.get_loc_start()
        loc_end = self.config.get_loc_end()
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

        plt.plot(loc_now[1], loc_now[0], 'r.', markersize=20)
        plt.plot(loc_end[1], loc_end[0], 'k*', markersize=20)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig(os.getcwd() + "/../../fig/trees/rrtcv.png")
        plt.show()

