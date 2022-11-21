""" Unit test for CTD simulator
"""

from unittest import TestCase
from AUVSimulator.CTDSimulator import CTDSimulator
from GRF import GRF
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
        self.ctd = CTDSimulator()

    def test_get_salinity_at_loc(self):
        """
        Test get salinity from location
        """
        np.random.seed(0)
        grid = self.grf.grid
        # c1: value at the corners
        loc = np.array([.0, .0])
        self.ctd.get_salinity_at_loc(loc)
        truth = self.ctd.get_ground_truth()
        # value = normalize(truth, 16, 32)
        value = truth
        plt.figure(figsize=(15, 12))
        plotf_vector(grid[:, 0], grid[:, 1], truth, cmap=get_cmap("RdBu", 10),
                     vmin=.0, vmax=1.4, stepsize=.1, threshold=.7, cbar_title="Value",
                     title="Ground field", xlabel="East", ylabel="North")
        plt.gca().set_aspect('equal')
        # plt.scatter(grid[:, 0], grid[:, 1], c=truth, cmap=get_cmap("RdBu", 10), vmin=0, vmax=1.1)
        # plt.colorbar()
        plt.show()



