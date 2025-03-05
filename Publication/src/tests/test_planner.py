""" Unit test for planner

This module tests the planner object with temporal focus

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-08-24
"""
from unittest import TestCase
from Planner.Planner import Planner
from Field import Field
from Config import Config
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap
from matplotlib.pyplot import get_cmap


class TestPlanner(TestCase):
    """ Common test class for the waypoint graph module
    """

    def setUp(self) -> None:
        loc_start = np.array([1000, -1000])
        sigma = 1.5
        nugget = .4
        # self.planner = Planner(loc_start, sigma=sigma, nugget=nugget)
        self.planner = Planner(loc_start, neighhour_distance=120, weight_eibv=1., weight_ivr=1.,
                               sigma=sigma, nugget=nugget, budget_mode=False,
                               approximate_eibv=False, fast_eibv=True)
        # self.planner = Planner(loc_start, weight_eibv=.0, weight_ivr=2., sigma=sigma, nugget=nugget)
        self.rrtstarcv = self.planner.get_rrtstarcv()
        self.cv = self.rrtstarcv.get_CostValley()
        self.stepsize = self.rrtstarcv.get_stepsize()
        self.field = self.cv.get_field()
        self.config = Config()
        self.grid = self.field.get_grid()
        self.plg_border = self.config.get_polygon_border()
        self.plg_obs = self.config.get_polygon_obstacle()

    # def test_initial_waypoints(self):
    #     """ Test initial indices to be 0. """
    #     wp_start = self.planner.get_starting_waypoint()
    #     wp_min_cv = self.cv.get_minimum_cost_location()
    #     angle = np.math.atan2(wp_min_cv[0] - wp_start[0],
    #                           wp_min_cv[1] - wp_start[1])
    #     xn = wp_start[0] + self.stepsize * np.sin(angle)
    #     yn = wp_start[1] + self.stepsize * np.cos(angle)
    #     wp_next = np.array([xn, yn])
    #     self.assertIsNone(testing.assert_array_equal(wp_next, self.planner.get_next_waypoint()))
    #
    #     xp = xn + self.stepsize * np.sin(angle)
    #     yp = yn + self.stepsize * np.cos(angle)
    #     wp_pion = np.array([xp, yp])
    #     self.assertIsNone(testing.assert_array_equal(wp_pion, self.planner.get_pioneer_waypoint()))
    #
    #     plt.plot(yp, xp, 'g.')
    #     plt.plot(yn, xn, 'b.')
    #     plt.plot(wp_start[1], wp_start[0], 'r.')
    #     plt.plot(self.plg[:, 1], self.plg[:, 0], 'r-.')
    #     plt.show()

    def test_sense_act_plan(self):
        """ Test update planner method. """
        xn, yn = self.planner.get_current_waypoint()
        xnn, ynn = self.planner.get_next_waypoint()
        xp, yp = self.planner.get_pioneer_waypoint()

        plt.plot(yp, xp, 'g.')
        plt.plot(ynn, xnn, 'b.')
        plt.plot(yn, xn, 'r.')
        plt.plot(self.plg_border[:, 1], self.plg_border[:, 0], 'r-.')
        plt.scatter(self.grid[:, 1], self.grid[:, 0], c=self.cv.get_cost_field(), s=300,
                    cmap=get_cmap("BrBG", 10), vmin=0, vmax=2, alpha=.1)
        plt.colorbar()
        plt.show()

        # s0: update planning trackers
        self.planner.update_planning_trackers()

        # s1: move auv to current location
        xn, yn = self.planner.get_current_waypoint()
        xnn, ynn = self.planner.get_next_waypoint()

        # s2: on the way to the current location, update field.
        ctd_data = np.array([[0, 1500, -1200, 20],
                             [1200, 1800, -1300, 23],
                             [2400, 2300, -1500, 30]])

        self.planner.update_pioneer_waypoint(ctd_data)
        xp, yp = self.planner.get_pioneer_waypoint()

        plt.plot(yp, xp, 'g.')
        plt.plot(ynn, xnn, 'b.')
        plt.plot(yn, xn, 'r.')
        plt.scatter(self.grid[:, 1], self.grid[:, 0], c=self.cv.get_cost_field(), s=300,
                    cmap=get_cmap("BrBG", 10), vmin=0, vmax=2, alpha=.1)
        plt.colorbar()
        plt.plot(self.plg_border[:, 1], self.plg_border[:, 0], 'r-.')
        plt.show()

        # s0: update planning trackers
        self.planner.update_planning_trackers()

        # s1: move auv to current location
        xn, yn = self.planner.get_current_waypoint()
        xnn, ynn = self.planner.get_next_waypoint()

        # s2: on the way to the current location, update field.
        ctd_data = np.array([[2400, 2500, -2000, 25],
                             [3600, 3000, -800, 30],
                             [4800, 3300, -900, 28]])

        self.planner.update_pioneer_waypoint(ctd_data)
        xp, yp = self.planner.get_pioneer_waypoint()

        plt.plot(yp, xp, 'g.')
        plt.plot(ynn, xnn, 'b.')
        plt.plot(yn, xn, 'r.')
        plt.scatter(self.grid[:, 1], self.grid[:, 0], c=self.cv.get_cost_field(), s=300,
                    cmap=get_cmap("BrBG", 10), vmin=0, vmax=2, alpha=.1)
        plt.colorbar()
        plt.plot(self.plg_border[:, 1], self.plg_border[:, 0], 'r-.')
        plt.show()

        # c3, one more step
        ctd_data = np.array([[4800, 2800, -2000, 25],
                             [6000, 3000, -800, 30],
                             [7200, 3300, -900, 28]])

        self.planner.update_pioneer_waypoint(ctd_data)
        xp, yp = self.planner.get_pioneer_waypoint()

        plt.plot(yp, xp, 'g.')
        plt.plot(ynn, xnn, 'b.')
        plt.plot(yn, xn, 'r.')
        plt.scatter(self.grid[:, 1], self.grid[:, 0], c=self.cv.get_cost_field(), s=300,
                    cmap=get_cmap("BrBG", 10), vmin=0, vmax=2, alpha=.1)
        plt.colorbar()
        plt.plot(self.plg_border[:, 1], self.plg_border[:, 0], 'r-.')
        plt.show()