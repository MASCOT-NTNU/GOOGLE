"""
Unittest for Myopic2D path planner
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
        loc = np.array([2000, -1500])
        self.myopic = Myopic2D(loc, neighbour_distance=240,
                               weight_eibv=1., weight_ivr=1., sigma=1., nugget=.1,
                               approximate_eibv=False, fast_eibv=True)
        self.cv = self.myopic.getCostValley()
        self.field = self.myopic.get_field()
        self.polygon_border = self.c.get_polygon_border()

    def test_get_next_waypoint(self) -> None:
        figpath = "/Users/yaolin/Downloads/fig/"

        ctd = np.array([[2000, -1500, 25]])
        id_s, id_n = self.myopic.get_candidates_indices()
        wp_next = self.myopic.update_next_waypoint(ctd)
        wp_curr = self.myopic.get_current_waypoint()
        wp_prev = self.myopic.get_previous_waypoint()
        loc_cand = self.myopic.get_loc_cand()
        traj = self.myopic.get_trajectory()
        grid = self.myopic.getCostValley().get_grf_model().grid
        plt.figure(figsize=(15, 15))
        plt.scatter(grid[:, 1], grid[:, 0], c=self.cv.get_cost_field(),
                    cmap=get_cmap("BrBG", 10), vmin=.0, vmax=2., s=250)
        plt.colorbar()

        for iid in id_n:
            wp = self.field.get_location_from_ind(iid)
            plt.plot(wp[1], wp[0], 'b.', label="Neighbour locations", markersize=35)
        for iid in id_s:
            wp = self.field.get_location_from_ind(iid)
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

        for i in range(30):
            print(i)
            ctd = np.array([[wp_curr[0], wp_curr[1], 25 + np.random.rand()]])
            id_s, id_n = self.myopic.get_candidates_indices()
            wp_next = self.myopic.update_next_waypoint(ctd)
            wp_curr = self.myopic.get_current_waypoint()
            wp_prev = self.myopic.get_previous_waypoint()
            loc_cand = self.myopic.get_loc_cand()
            traj = self.myopic.get_trajectory()
            plt.figure(figsize=(15, 15))
            plt.scatter(grid[:, 1], grid[:, 0], c=self.cv.get_cost_field(),
                        cmap=get_cmap("BrBG", 10), vmin=.0, vmax=2., s=250)
            plt.colorbar()

            for iid in id_n:
                wp = self.field.get_location_from_ind(iid)
                plt.plot(wp[1], wp[0], 'b.', label="Neighbour locations", markersize=35)
            for iid in id_s:
                wp = self.field.get_location_from_ind(iid)
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

