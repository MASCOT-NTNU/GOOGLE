""" Unit test for CTD simulator
"""

from unittest import TestCase
from AUVSimulator.CTDSimulator import CTDSimulator
from GRF.GRF import GRF
from Field import Field
from Config import Config
import numpy as np
from numpy import testing
import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from Visualiser.Visualiser import plotf_vector
from usr_func.normalize import normalize


class TestCTDSimulator(TestCase):

    def setUp(self) -> None:
        self.grf = GRF()
        self.f = Field()
        self.c = Config()
        self.ctd = CTDSimulator()

    def test_get_salinity_at_loc(self) -> None:
        """
        Test get salinity from location
        """
        np.random.seed(0)
        grid = self.grf.grid
        # c1: value at the corners
        loc = np.array([7000, 8000])
        self.ctd.get_salinity_at_loc(loc)
        truth = self.ctd.get_ground_truth()
        # value = normalize(truth, 16, 32)
        value = truth
        plg = self.c.get_polygon_border()
        plt.figure(figsize=(15, 12))
        # plt.scatter(grid[:, 1], grid[:, 0], c=truth, cmap=get_cmap("BrBG", 10), vmin=10, vmax=35)
        # plt.colorbar()
        plotf_vector(grid[:, 1], grid[:, 0], values=truth, cmap=get_cmap("BrBG", 10),
                     vmin=10, vmax=36, stepsize=1.5, threshold=27, cbar_title="Value",
                     title="Ground field", xlabel="East", ylabel="North", polygon_border=plg)
        # plt.plot(plg[:, 1], plg[:, 0], 'r-.')
        plt.gca().set_aspect('equal')
        # plt.scatter(grid[:, 0], grid[:, 1], c=truth, cmap=get_cmap("RdBu", 10), vmin=0, vmax=1.1)
        # plt.colorbar()
        plt.show()

        # # show prior
        # plt.figure(figsize=(15, 12))
        # plotf_vector(grid[:, 1], grid[:, 0], values=self.grf.get_mu(), cmap=get_cmap("BrBG", 10),
        #              vmin=10, vmax=36, stepsize=1.5, threshold=27, cbar_title="Value",
        #              title="Prior field", xlabel="East", ylabel="North", polygon_border=plg)
        # plt.gca().set_aspect('equal')
        # plt.show()



