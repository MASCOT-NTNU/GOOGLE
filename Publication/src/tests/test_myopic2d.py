"""
Unittest for Myopic2D path planner

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-09-02
"""
from Config import Config
from Field import Field
from unittest import TestCase
from Planner.Myopic2D.Myopic2D import Myopic2D
import matplotlib.pyplot as plt
import numpy as np
# from matplotlib.cm import get_cmap
from matplotlib.pyplot import get_cmap


class TestMyopic2D(TestCase):
    def setUp(self) -> None:
        self.c = Config()
        self.myopic = Myopic2D(weight_eibv=1., weight_ivr=1.)
        self.cv = self.myopic.getCostValley()
        self.field = Field()
        self.polygon_border = self.c.get_polygon_border()

    def test_get_next_waypoint(self) -> None:
        figpath = "/Users/yaolin/Downloads/fig/"

        wp_s, wp_n = self.myopic.get_candidates_waypoints()
        wp_curr = self.myopic.get_current_waypoint()
        wp_prev = self.myopic.get_previous_waypoint()
        wp_next = self.myopic.get_next_waypoint()

        traj = self.myopic.get_trajectory()
        grid = self.myopic.getCostValley().get_grf_model().grid
        plt.figure(figsize=(15, 15))
        plt.scatter(grid[:, 1], grid[:, 0], c=self.cv.get_cost_field(),
                    cmap=get_cmap("BrBG", 10), vmin=.0, vmax=2., s=250)
        plt.colorbar()

        for wp in wp_n:
            plt.plot(wp[1], wp[0], 'b.', label="Neighbour locations", markersize=35)
        for wp in wp_s:
            plt.plot(wp[1], wp[0], 'y.', label="Candidate locations", markersize=30)

        plt.plot(wp_next[1], wp_next[0], 'r.', label="Next waypoint", markersize=25)
        plt.plot(wp_curr[1], wp_curr[0], 'g.', label="Curr waypoint", markersize=20)
        plt.plot(wp_prev[1], wp_prev[0], 'c.', label="Prev waypoint", markersize=20)

        plt.plot(traj[:, 1], traj[:, 0], 'k.-')
        plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'k-.')
        plt.gca().set_aspect("equal")
        plt.savefig(figpath + "P_000.png")
        plt.close("all")
        # plt.show()

        for i in range(120):
            print(i)
            ctd = np.array([[i * 600 + 1623450000, wp_curr[0], wp_curr[1], 25 + np.random.rand()]])
            wp_s, wp_n = self.myopic.get_candidates_waypoints()
            wp_next = self.myopic.update_next_waypoint(ctd)
            wp_curr = self.myopic.get_current_waypoint()
            wp_prev = self.myopic.get_previous_waypoint()

            traj = self.myopic.get_trajectory()
            plt.figure(figsize=(15, 15))
            plt.scatter(grid[:, 1], grid[:, 0], c=self.cv.get_cost_field(),
                        cmap=get_cmap("BrBG", 10), vmin=.0, vmax=2., s=250)
            plt.colorbar()

            for wp in wp_n:
                plt.plot(wp[1], wp[0], 'b.', label="Neighbour locations", markersize=35)
            for wp in wp_s:
                plt.plot(wp[1], wp[0], 'y.', label="Candidate locations", markersize=30)

            plt.plot(wp_next[1], wp_next[0], 'r.', label="Next waypoint", markersize=25)
            plt.plot(wp_curr[1], wp_curr[0], 'g.', label="Curr waypoint", markersize=20)
            plt.plot(wp_prev[1], wp_prev[0], 'c.', label="Prev waypoint", markersize=20)

            plt.plot(traj[:, 1], traj[:, 0], 'y.-')
            plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'k-.')
            plt.gca().set_aspect("equal")
            plt.savefig(figpath + "P_{:03d}.png".format(i+1))
            plt.close("all")
            # plt.show()

