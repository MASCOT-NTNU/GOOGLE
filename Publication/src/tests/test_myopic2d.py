"""
Unittest for Myopic2D path planner
"""
from Config import Config
from Field import Field
from unittest import TestCase
from Planner.Myopic2D.Myopic2D import Myopic2D
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import get_cmap


class TestMyopic2D(TestCase):
    def setUp(self) -> None:
        self.c = Config()
        loc = np.array([2000, -1500])
        self.myopic = Myopic2D(loc)
        self.cv = self.myopic.getCostValley()
        self.polygon_border = self.c.get_polygon_border()

    def test_get_next_waypoint(self) -> None:
        ctd = np.array([[2000, -1500, 25]])
        wp_next = self.myopic.update_next_waypoint(ctd)
        wp_curr = self.myopic.get_current_waypoint()
        wp_prev = self.myopic.get_previous_waypoint()
        loc_cand = self.myopic.get_loc_cand()
        traj = self.myopic.get_trajectory()
        grid = self.myopic.getCostValley().get_grf_model().grid
        plt.scatter(grid[:, 1], grid[:, 0], c=self.cv.get_cost_field(),
                    cmap=get_cmap("BrBG", 10), vmin=.0, vmax=2., s=100, alpha=.4)
        plt.colorbar()
        plt.plot(wp_next[1], wp_next[0], 'r.', label="Next waypoint", markersize=20)
        plt.plot(wp_curr[1], wp_curr[0], 'g.', label="Curr waypoint", markersize=20)
        plt.plot(wp_prev[1], wp_prev[0], 'b.', label="Prev waypoint", markersize=20)
        plt.plot(traj[:, 1], traj[:, 0], 'y.-')
        plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'k-.')
        plt.gca().set_aspect("equal")
        plt.show()

        for i in range(3):
            ctd = np.array([[wp_curr[0], wp_curr[1], 25 + np.random.rand()]])
            wp_next = self.myopic.update_next_waypoint(ctd)
            wp_curr = self.myopic.get_current_waypoint()
            wp_prev = self.myopic.get_previous_waypoint()
            loc_cand = self.myopic.get_loc_cand()
            traj = self.myopic.get_trajectory()
            plt.scatter(grid[:, 1], grid[:, 0], c=self.cv.get_cost_field(),
                        cmap=get_cmap("BrBG", 10), vmin=.0, vmax=2., s=100, alpha=.4)
            plt.colorbar()
            plt.plot(wp_next[1], wp_next[0], 'r.', label="Next waypoint", markersize=20)
            plt.plot(wp_curr[1], wp_curr[0], 'g.', label="Curr waypoint", markersize=20)
            plt.plot(wp_prev[1], wp_prev[0], 'b.', label="Prev waypoint", markersize=20)
            plt.plot(traj[:, 1], traj[:, 0], 'y.-')
            plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'k-.')
            plt.gca().set_aspect("equal")
            plt.show()

