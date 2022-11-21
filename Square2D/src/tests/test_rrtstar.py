from unittest import TestCase
from RRTStar.RRTStar import RRTStar
from CostValley.CostValley import CostValley
import matplotlib.pyplot as plt
import numpy as np
from Visualiser.TreePlotter import TreePlotter
from Visualiser.Visualiser import plotf_vector
from matplotlib.cm import get_cmap


class TestRRTStar(TestCase):

    def setUp(self) -> None:
        self.rrtstar = RRTStar()
        self.tp = TreePlotter()
        self.cv = CostValley()
        self.field = self.cv.get_field()
        self.grid = self.cv.get_grid()
        self.polygon_border = self.field.get_polygon_border()
        self.polygon_border = np.append(self.polygon_border, self.polygon_border[0, :].reshape(1, -1), axis=0)
        self.polygon_obstacle = self.field.get_polygon_obstacles()[0]
        self.polygon_obstacle = np.append(self.polygon_obstacle, self.polygon_obstacle[0, :].reshape(1, -1), axis=0)

    def test_get_new_location(self):
        loc_now = np.array([.01, .01])
        loc_end = np.array([.01, .99])
        wp = self.rrtstar.get_next_waypoint(loc_now, loc_end, cost_valley=self.cv)
        print(wp)
        nodes = self.rrtstar.get_nodes()
        traj = self.rrtstar.get_trajectory()
        self.tp.update_trees(nodes)
        plt.figure(figsize=(15, 12))
        cv = self.cv.get_cost_valley()
        plotf_vector(self.grid[:, 0], self.grid[:, 1], cv, xlabel='x', ylabel='y', title='RRTCV', cbar_title="Cost",
                     cmap=get_cmap("RdBu", 10), alpha=.3)
        self.tp.plot_tree()
        plt.plot(traj[:, 0], traj[:, 1], 'k-', linewidth=10)
        # plt.plot(wp[0], wp[1], 'b*', markersize=20)
        # plt.plot(loc_now[0], loc_now[1], 'c.', markersize=10)
        plt.plot(self.polygon_border[:, 0], self.polygon_border[:, 1], 'r-.')
        plt.plot(self.polygon_obstacle[:, 0], self.polygon_obstacle[:, 1], 'r-.')

        plt.plot(0, 0, 'r.', markersize=20)
        plt.plot(0, 1, 'k*', markersize=20)
        plt.xlabel("x")
        plt.ylabel("y")
        plt.savefig("/Users/yaoling/Downloads/trees/rrtcv.png")
        plt.show()

