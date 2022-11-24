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
from numpy import testing
from matplotlib import tri


# def is_masked(x, y) -> bool:
#     point = Point(x, y)
#     # loc = np.array([x, y])
#     masked = False
#     if not plg_wgs_sh.contains(point):
#     # if not field.border_contains(loc):
#         masked = True
#     return masked

def plotf_vector2(x, y, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
                 cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
                 stepsize=None, threshold=None, polygon_border=None,
                 polygon_obstacle=None, xlabel=None, ylabel=None):
    """
    Remember x, y is plotting x, y, thus x along horizonal and y along vertical.
    """
    plt.scatter(x, y, c=values, cmap=get_cmap("BrBG", 10), vmin=vmin, vmax=vmax)
    plt.colorbar()
    plt.xlim([np.amin(x), np.amax(x)])
    plt.ylim([np.amin(y), np.amax(y)])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if np.any(polygon_border):
        plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'k-.', lw=2)
        if np.any(polygon_obstacle):
            for i in range(len(polygon_obstacle)):
                plt.plot(polygon_obstacle[i][:, 1], polygon_obstacle[i][:, 0], 'k-.', lw=2)
    return plt.gca()


def plotf(self, v1, v2, title1="mean", title2="cov", vmin1=None, vmax1=None, vmin2=None, vmax2=None):
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(nrows=1, ncols=2)
    ax = fig.add_subplot(gs[0])
    # plotf_vector(self.grid[:, 1], self.grid[:, 0], v1,
    #              polygon_border=self.c.get_polygon_border(), vmin=vmin1, vmax=vmax1)
    plotf_vector(self.grid[:, 1], self.grid[:, 0], v1, title=title1, cmap=get_cmap("BrBG", 10),
                 vmin=10, vmax=33, cbar_title="Salinity", stepsize=1.5, threshold=27,
                 polygon_border=self.c.get_polygon_border(), polygon_obstacle=self.c.get_polygon_obstacle())
    # plt.title(title1)

    ax = fig.add_subplot(gs[1])
    # plotf_vector2(self.grid[:, 1], self.grid[:, 0], v2, cmap="RdBu",
    #              polygon_border=self.c.get_polygon_border(), vmin=vmin2, vmax=vmax2)
    plotf_vector(self.grid[:, 1], self.grid[:, 0], v2, title=title1, cmap=get_cmap("RdBu", 10),
                 vmin=vmin2, vmax=vmax2, cbar_title="std",
                 polygon_border=self.c.get_polygon_border(), polygon_obstacle=self.c.get_polygon_obstacle())
    plt.title(title2)
    plt.show()


class TestGRF(TestCase):

    def setUp(self) -> None:
        self.c = Config()
        self.g = GRF()
        self.grid = self.g.field.get_grid()
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        self.f = self.g.field
        self.cov = self.g.get_Sigma()
        self.mu = self.g.get_mu()
        self.sigma = self.g.get_sigma()

        # plt.imshow(self.cov)
        # plt.colorbar()
        # plt.show()
        # plt.show()

    def test_prior_matern_covariance(self):
        print("S1")
        plotf(self, v1=self.g.get_mu(), v2=np.sqrt(np.diag(self.g.get_Sigma())), vmin1=10, vmax1=30, vmin2=0, vmax2=self.sigma)
        print("END S1")

    def test_assimilate(self):
        # c2: one
        print("S2")
        dataset = np.array([[3000, 1000, 0, 10]])
        self.g.assimilate_data(dataset)
        plotf(self, v1=self.g.get_mu(), v2=np.sqrt(np.diag(self.g.get_Sigma())), vmin1=10, vmax1=30, vmin2=0, vmax2=self.sigma)

        # c3: multiple
        dataset = np.array([[2000, -1000,  0, 15],
                            [1500, -1500, 0, 10],
                            [1400, -1800, 0, 25],
                            [2500, -1400, 0, 20]])
        self.g.assimilate_data(dataset)
        plotf(self, v1=self.g.get_mu(), v2=np.sqrt(np.diag(self.g.get_Sigma())), vmin1=10, vmax1=30, vmin2=0, vmax2=self.sigma)
        print("End S2")

    def test_get_ei_field_total(self):
        # c1: no data assimilation
        print("S3")
        """ For now, it takes too much time to compute the entire EI field. """
        eibv, ivr = self.g.get_ei_field_total()
        plotf(self, v1=eibv, v2=ivr)

        # eibv, ivr = self.g.get_ei_field_para()
        # plotf(self, v1=eibv, v2=ivr)

        # c2: with data assimilation
        dataset = np.array([[8000, 8000, 0, 10],
                            [9200, 9000, 0, 15],
                            [7000, 8000, 0, 13],
                            [8000, 7000, 0, 33],
                            [6000, 8000, 0, 26],
                            [5000, 9000, 0, 24]])
        self.g.assimilate_data(dataset)
        eibv, ivr = self.g.get_ei_field_total()
        plotf(self, v1=eibv, v2=ivr)
        plotf(self, v1=self.g.get_mu(), v2=np.diag(self.g.get_Sigma()))
        print("End S3")



