""" Unit test for GRF
This module tests the GRF object.
"""

from Config import Config
from unittest import TestCase
from GRF.GRF import GRF
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from Visualiser.Visualiser import plotf_vector
from matplotlib.cm import get_cmap


def plotf(self, v1, v2, title1="mean", title2="cov", cbar1="salinity", cbar2="std", stepsize1=None, threshold1=None,
          stepsize2=None, threshold2=None, vmin1=None, vmax1=None, vmin2=None, vmax2=None):
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(nrows=1, ncols=2)
    ax = fig.add_subplot(gs[0])
    plotf_vector(self.grid[:, 1], self.grid[:, 0], v1, title=title1, cmap=get_cmap("BrBG", 10),
                 vmin=vmin1, vmax=vmax1, cbar_title=cbar1, stepsize=stepsize1, threshold=threshold1,
                 polygon_border=self.c.get_polygon_border(), polygon_obstacle=self.c.get_polygon_obstacle())
    plt.title(title1)

    ax = fig.add_subplot(gs[1])
    plotf_vector(self.grid[:, 1], self.grid[:, 0], v2, title=title1, cmap=get_cmap("RdBu", 10),
                 vmin=vmin2, vmax=vmax2, cbar_title=cbar2, stepsize=stepsize2, threshold=threshold2,
                 polygon_border=self.c.get_polygon_border(), polygon_obstacle=self.c.get_polygon_obstacle())
    plt.title(title2)
    plt.show()


class TestGRF(TestCase):

    def setUp(self) -> None:
        self.c = Config()
        self.g = GRF()
        self.f = self.g.field
        self.grid = self.f.get_grid()
        self.cov = self.g.get_covariance_matrix()
        self.mu = self.g.get_mu()
        self.sigma = self.g.get_sigma()

    def test_prior_matern_covariance(self):
        print("S1")
        plotf(self, v1=self.g.get_mu(), v2=np.sqrt(np.diag(self.g.get_covariance_matrix())),
              vmin1=10, vmax1=30, vmin2=0, vmax2=self.sigma, cbar1="salinity", cbar2="std", stepsize1=1.5, threshold1=27)
        print("END S1")

    def test_assimilate(self):
        # c2: one
        print("S2")
        dataset = np.array([[3000, 1000, 0, 10]])
        self.g.assimilate_data(dataset)
        plotf(self, v1=self.g.get_mu(), v2=np.sqrt(np.diag(self.g.get_covariance_matrix())),
              vmin1=10, vmax1=30, vmin2=0, vmax2=self.sigma, cbar1="salinity", cbar2="std", stepsize1=1.5, threshold1=27)

        # c3: multiple
        dataset = np.array([[2000, -1000,  0, 15],
                            [1500, -1500, 0, 10],
                            [1400, -1800, 0, 25],
                            [2500, -1400, 0, 20]])
        self.g.assimilate_data(dataset)
        plotf(self, v1=self.g.get_mu(), v2=np.sqrt(np.diag(self.g.get_covariance_matrix())),
              vmin1=10, vmax1=30, vmin2=0, vmax2=self.sigma, cbar1="salinity", cbar2="std", stepsize1=1.5, threshold1=27)
        print("End S2")

    def test_get_ei_field(self):
        # c1: no data assimilation
        print("S3")
        """ For now, it takes too much time to compute the entire EI field. """
        eibv, ivr = self.g.get_ei_field()
        plotf(self, v1=eibv, v2=ivr, vmin1=0, vmax1=1, vmin2 =0, vmax2=1, cbar1="cost", cbar2="cost",
              title1="EIBV", title2="IVR")

        # c2: with data assimilation
        dataset = np.array([[2000, -1000,  0, 15],
                            [1500, -1500, 0, 10],
                            [1400, -1800, 0, 25],
                            [2500, -1400, 0, 20]])
        self.g.assimilate_data(dataset)
        eibv, ivr = self.g.get_ei_field()

        plotf(self, v1=eibv, v2=ivr, vmin1=0, vmax1=1, vmin2=0, vmax2=1, cbar1="cost", cbar2="cost")
        plotf(self, v1=self.g.get_mu(), v2=np.diag(self.g.get_covariance_matrix()),
              vmin1=10, vmax1=30, vmin2=0, vmax2=self.sigma,
              cbar1="salinity", cbar2="std", stepsize1=1.5, threshold1=27)
        print("End S3")

