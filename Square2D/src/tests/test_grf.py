""" Unit test for GRF
This module tests the GRF object.
"""


from unittest import TestCase
from GRF import GRF
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec
from Visualiser.Visualiser import plotf_vector
from matplotlib.cm import get_cmap


def plotf(self, v1, v2, title1="mean", title2="cov"):
    fig = plt.figure(figsize=(15, 5))
    gs = GridSpec(nrows=1, ncols=2)
    ax = fig.add_subplot(gs[0])
    plotf_vector(self.grid[:, 0], self.grid[:, 1], v1,
                 polygon_border=self.g.field.get_polygon_border(),
                 polygon_obstacle=self.g.field.get_polygon_obstacles())
    plt.title(title1)

    ax = fig.add_subplot(gs[1])
    plotf_vector(self.grid[:, 0], self.grid[:, 1], v2,
                 polygon_border=self.g.field.get_polygon_border(),
                 polygon_obstacle=self.g.field.get_polygon_obstacles())
    plt.title(title2)
    plt.show()


class TestGRF(TestCase):

    def setUp(self) -> None:
        self.g = GRF()
        self.grid = self.g.field.get_grid()
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        # mu_prior = 1. - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .07)
        # mu_prior = (.5 * (1 - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .07)) +
        #             .5 * (1 - np.exp(- ((x - .0) ** 2 + (y - .5) ** 2) / .07)))
        # mu_prior = mu_prior.reshape(-1, 1)
        # self.g.set_mu(mu_prior)
        self.cov = self.g.get_Sigma()
        self.mu = self.g.get_mu()

    def test_prior_matern_covariance(self):
        plotf(self, v1=self.g.get_mu(), v2 = np.diag(self.g.get_Sigma()))

    # def test_assimilate(self):
    #     # c2: one
    #     dataset = np.array([[.2, .2, 10]])
    #     self.g.assimilate_data(dataset)
    #     plotf(self, v1=self.g.get_mu(), v2=np.diag(self.g.get_Sigma()))
    #
    #     # c3: multiple
    #     dataset = np.array([[.6, .4,  1],
    #                         [.2, .8, .5],
    #                         [.8, .2, .1],
    #                         [.9, .9, .7]])
    #     self.g.assimilate_data(dataset)
    #     plotf(self, v1=self.g.get_mu(), v2=np.diag(self.g.get_Sigma()))

    def test_get_ei_field(self):
        # c1: no data assimilation
        eibv, ivr = self.g.get_ei_field()
        plotf(self, v1=eibv, v2=ivr)

        # # c2: with data assimilation
        # dataset = np.array([[.1, .8, .1],
        #                     [.2, .8, .1],
        #                     [.3, .8, .0],
        #                     [.1, .7, .6],
        #                     [.2, .9, .7],
        #                     [.05, .9, .7]])
        # self.g.assimilate_data(dataset)
        # eibv, ivr = self.g.get_ei_field()
        # plotf(self, v1=eibv, v2=ivr)
        # plotf(self, v1=self.g.get_mu(), v2=np.diag(self.g.get_Sigma()))




