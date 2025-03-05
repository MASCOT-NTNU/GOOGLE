""" Unit test for CTD simulator

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-08-30
"""

from unittest import TestCase
from WGS import WGS
from AUVSimulator.CTDSimulator import CTDSimulator
from SINMOD import SINMOD
import numpy as np
from numpy import testing
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from datetime import datetime
import os


class TestCTDSimulator(TestCase):

    def setUp(self) -> None:
        filepath_sinmod = os.getcwd() + "/../sinmod/samples_2022.05.11.nc"
        self.ctd = CTDSimulator(random_seed=14, filepath=filepath_sinmod, sigma=1.)
        self.sinmod = SINMOD(filepath_sinmod)

    def test_get_salinity_at_time_loc(self) -> None:
        # c1, whole region
        x, y, depth = self.sinmod.get_coordinates()
        loc = np.vstack((x.flatten(), y.flatten())).T

        dt = 1200
        for i in range(10):
            salinity = self.ctd.get_salinity_at_dt_loc(dt, loc)
            plt.figure()
            plt.scatter(loc[:, 1], loc[:, 0], c=salinity, cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
            plt.colorbar()
            plt.title("Time: " + datetime.fromtimestamp(self.ctd.timestamp).strftime("%Y-%m-%d %H:%M:%S"))
            plt.show()

        # c2, regional locations
        xx = np.linspace(2000, 4000, 100)
        yy = np.linspace(0, 2000, 100)
        xv, yv = np.meshgrid(xx, yy)
        x = xv.flatten()
        y = yv.flatten()
        loc = np.vstack((x, y)).T

        dt = 1200
        for i in range(10):
            salinity = self.ctd.get_salinity_at_dt_loc(dt, loc)
            plt.figure()
            plt.scatter(loc[:, 1], loc[:, 0], c=salinity, cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
            plt.colorbar()
            plt.title("Time: " + datetime.fromtimestamp(self.ctd.timestamp).strftime("%Y-%m-%d %H:%M:%S"))
            plt.show()
