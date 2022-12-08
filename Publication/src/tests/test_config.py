""" Unittest for configuration. """
from unittest import TestCase
from Config import Config
import matplotlib.pyplot as plt


class TestConfig(TestCase):

    def setUp(self) -> None:
        self.c = Config()

    def test_starting_home_location(self):
        loc_start = self.c.get_loc_start()
        plg_border = self.c.get_polygon_border()
        plg_obs = self.c.get_polygon_obstacle()
        plt.plot(plg_border[:, 1], plg_border[:, 0], 'r-.')
        plt.plot(plg_obs[:, 1], plg_obs[:, 0], 'r-.')
        plt.plot(loc_start[1], loc_start[0], 'k.')
        plt.show()

    def test_wgs_starting_home_location(self):
        loc_start = self.c.get_wgs_loc_start()
        plg_border = self.c.get_wgs_polygon_border()
        plg_obs = self.c.get_wgs_polygon_obstacle()
        plt.plot(plg_border[:, 1], plg_border[:, 0], 'r-.')
        plt.plot(plg_obs[:, 1], plg_obs[:, 0], 'r-.')
        plt.plot(loc_start[1], loc_start[0], 'k.')
        plt.show()

