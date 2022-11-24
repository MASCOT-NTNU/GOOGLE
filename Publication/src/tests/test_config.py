from unittest import TestCase
from Config import Config
from WGS import WGS
import matplotlib.pyplot as plt
from numpy import testing
import numpy as np


class TestConfig(TestCase):

    def setUp(self) -> None:
        self.c = Config()

    def test_starting_home_location(self):
        loc_end = self.c.get_loc_end()
        loc_start = self.c.get_loc_start()
        plg_border = self.c.get_polygon_border()
        plg_obs = self.c.get_polygon_obstacle()
        plt.plot(plg_border[:, 1], plg_border[:, 0], 'r-.')
        plt.plot(plg_obs[:, 1], plg_obs[:, 0], 'r-.')
        plt.plot(loc_start[1], loc_start[0], 'k.')
        plt.plot(loc_end[1], loc_end[0], 'b.')
        plt.show()

    def test_wgs_starting_home_location(self):
        loc_end = self.c.get_wgs_loc_end()
        loc_start = self.c.get_wgs_loc_start()
        plg_border = self.c.get_wgs_polygon_border()
        plg_obs = self.c.get_wgs_polygon_obstacle()
        plt.plot(plg_border[:, 1], plg_border[:, 0], 'r-.')
        plt.plot(plg_obs[:, 1], plg_obs[:, 0], 'r-.')
        plt.plot(loc_start[1], loc_start[0], 'k.')
        plt.plot(loc_end[1], loc_end[0], 'b.')
        plt.show()



