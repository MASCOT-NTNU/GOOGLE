""" Unit test for CTD simulator
"""
from Simulators.CTD import CTD
from Visualiser.Visualiser import plotf_vector
from GRF.GRF import GRF
from Field import Field
from Config import Config

from unittest import TestCase
import numpy as np
import matplotlib.pyplot as plt
# from matplotlib.cm import get_cmap
from matplotlib.pyplot import get_cmap


class TestCTDSimulator(TestCase):

    def setUp(self) -> None:
        self.grf = GRF()
        self.f = Field()
        self.c = Config()
        self.ctd = CTD()

    def test_rmse(self) -> None:
        self.mu_truth = self.ctd.get_ground_truth()
        ind = np.random.randint(0, len(self.mu_truth), 100)
        wp = self.f.get_location_from_ind(ind)

        rmse = []
        from sklearn.metrics import mean_squared_error
        for i in range(len(wp)):
            ctd = self.ctd.get_ctd_data_1hz(wp[i])
            # print(ctd)
            self.grf.assimilate_data(ctd)
            rmse.append(mean_squared_error(self.mu_truth, self.grf.get_mu(), squared=False))
        import matplotlib.pyplot as plt
        plt.plot(rmse)
        plt.show()

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
        value = truth
        plg = self.c.get_polygon_border()
        plt.figure(figsize=(15, 12))
        plotf_vector(grid[:, 1], grid[:, 0], values=truth, cmap=get_cmap("BrBG", 10),
                     vmin=10, vmax=36, stepsize=1.5, threshold=27, cbar_title="Value",
                     title="Ground field", xlabel="East", ylabel="North", polygon_border=plg)
        plt.gca().set_aspect('equal')
        plt.show()

        # show prior
        plt.figure(figsize=(15, 12))
        plotf_vector(grid[:, 1], grid[:, 0], values=self.grf.get_mu(), cmap=get_cmap("BrBG", 10),
                     vmin=10, vmax=36, stepsize=1.5, threshold=27, cbar_title="Value",
                     title="Prior field", xlabel="East", ylabel="North", polygon_border=plg)
        plt.gca().set_aspect('equal')
        plt.show()

    def test_get_data_along_path(self) -> None:
        # c1: move to one step
        data = self.ctd.get_ctd_data(np.array([3000, -1000]))
        grid = self.grf.grid
        truth = self.ctd.get_ground_truth()
        plt.figure()
        plt.scatter(grid[:, 1], grid[:, 0], c=truth, cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
        plt.scatter(data[:, 1], data[:, 0], c=data[:, -1], cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
        plt.colorbar()
        plt.show()

        # c2: move to another direction
        data = self.ctd.get_ctd_data(np.array([2000, -2000]))
        grid = self.grf.grid
        truth = self.ctd.get_ground_truth()
        plt.figure()
        plt.scatter(grid[:, 1], grid[:, 0], c=truth, cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
        plt.scatter(data[:, 1], data[:, 0], c=data[:, -1], cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
        plt.colorbar()
        plt.show()

        # c2: move to another direction
        data = self.ctd.get_ctd_data(np.array([1000, -1500]))
        grid = self.grf.grid
        truth = self.ctd.get_ground_truth()
        plt.figure()
        plt.scatter(grid[:, 1], grid[:, 0], c=truth, cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
        plt.scatter(data[:, 1], data[:, 0], c=data[:, -1], cmap=get_cmap("BrBG", 10), vmin=10, vmax=33)
        plt.colorbar()
        plt.show()

