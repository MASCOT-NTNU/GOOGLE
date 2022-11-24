from unittest import TestCase
from Planner.StraightLinePathPlanner import StraightLinePathPlanner
from Field import Field
import numpy as np
import matplotlib.pyplot as plt


class TestStraightLinePlanner(TestCase):

    def setUp(self) -> None:
        self.sl = StraightLinePathPlanner()
        self.f = Field()

    def test_get_next_waypoint(self):
        # c1: start location
        loc_now = np.array([4000, 4000])
        loc_home = np.array([12000, 10000])
        wp = self.sl.get_waypoint_from_straight_line(loc_now, loc_home)
        plg = self.f.get_wgs_polygon_border()
        plt.plot(plg[:, 1], plg[:, 0], 'r-.')
        plt.plot(loc_now[1], loc_now[0], 'k.')
        plt.plot(loc_home[1], loc_home[0], 'b.')
        plt.plot(wp[1], wp[0], 'g.')
        plt.show()

        # c2: move a bit further
        loc_now = np.array([8000, 8000])
        loc_home = np.array([12000, 10000])
        wp = self.sl.get_waypoint_from_straight_line(loc_now, loc_home)
        plg = self.f.get_wgs_polygon_border()
        plt.plot(plg[:, 1], plg[:, 0], 'r-.')
        plt.plot(loc_now[1], loc_now[0], 'k.')
        plt.plot(loc_home[1], loc_home[0], 'b.')
        plt.plot([loc_now[1], loc_home[1]], [loc_now[0], loc_home[0]], 'y-', alpha=.5)
        plt.plot(wp[1], wp[0], 'g.')
        plt.show()
        pass

