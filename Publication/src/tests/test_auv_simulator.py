""" Unit test for AUV Simulator

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-08-24
"""
import os
from unittest import TestCase
from AUVSimulator.AUVSimulator import AUVSimulator
from WGS import WGS
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
import numpy as np
import pandas as pd
from numpy import testing
from usr_func.is_list_empty import is_list_empty


def value(x, y):
    return 2 * x + 3 * y


class TestAUVSimulator(TestCase):

    def setUp(self) -> None:
        self.auv = AUVSimulator(random_seed=0, sigma=1., loc_start=np.array([0, 0]), temporal_truth=True)
        self.polygon_border_wgs = pd.read_csv(os.getcwd() + "/csv/polygon_border.csv").to_numpy()
        x, y = WGS.latlon2xy(self.polygon_border_wgs[:, 0], self.polygon_border_wgs[:, 1])
        self.polygon_border = np.stack((x, y), axis=1)

    def test_get_ctd_data_at_dt_loc(self) -> None:
        def plot_data_on_path(loc, counter=0):
            self.auv.move_to_location(loc)
            df = self.auv.get_ctd_data()
            plt.plot(df[:, -1], label=str(counter))

        plt.figure()
        loc1 = np.array([2750, 0])
        self.auv.move_to_location(loc1)
        # plot_data_on_path(loc1, counter=1)

        loc2 = np.array([2500, 1000])
        plot_data_on_path(loc2, counter=1)

        plot_data_on_path(loc1, counter=2)
        plot_data_on_path(loc2, counter=3)
        plot_data_on_path(loc1, counter=4)
        plot_data_on_path(loc2, counter=5)
        plot_data_on_path(loc1, counter=6)
        plot_data_on_path(loc2, counter=7)
        plot_data_on_path(loc1, counter=8)

        plt.legend()
        plt.show()

    def test_move_to_location(self):
        """
        Test if the AUV moves according to the given direction.
        """
        # c1: starting location
        self.assertIsNone(testing.assert_array_equal(self.auv.get_location(), np.array([0, 0])))
        self.assertIsNone(testing.assert_array_equal(self.auv.get_previous_location(), np.array([0, 0])))

        # c2: move to another location
        loc_new = np.array([10, 10])
        self.auv.move_to_location(loc_new)
        self.assertIsNone(testing.assert_array_equal(self.auv.get_location(), loc_new))
        self.assertIsNone(testing.assert_array_equal(self.auv.get_previous_location(), np.array([0, 0])))

        # c3: move to another location
        loc_new = np.array([20, 20])
        self.auv.move_to_location(loc_new)
        self.assertIsNone(testing.assert_array_equal(self.auv.get_location(), loc_new))
        self.assertIsNone(testing.assert_array_equal(self.auv.get_previous_location(), np.array([10, 10])))

    def test_arrived(self):
        # c1: not arrived
        self.assertFalse(self.auv.is_arrived())

        # c2: arrived
        self.auv.arrive()
        self.assertTrue(self.auv.is_arrived())

        # c3: move
        self.auv.move()
        self.assertFalse(self.auv.is_arrived())
