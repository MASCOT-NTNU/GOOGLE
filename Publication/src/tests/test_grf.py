""" Unit test for GRF
This module tests the GRF object.
"""

from Config import Config
from unittest import TestCase
from GRF.GRF import GRF
import matplotlib.pyplot as plt
import numpy as np
from usr_func.set_resume_state import set_resume_state
from matplotlib.gridspec import GridSpec
# from Visualiser.Visualiser import plotf_vector
from matplotlib.cm import get_cmap
from numpy import testing


def plotf_vector(x, y, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
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
    plotf_vector(self.grid[:, 1], self.grid[:, 0], v1,
                 polygon_border=self.g.field.get_polygon_border(), vmin=vmin1, vmax=vmax1)
    plt.title(title1)

    ax = fig.add_subplot(gs[1])
    plotf_vector(self.grid[:, 1], self.grid[:, 0], v2,
                 polygon_border=self.g.field.get_polygon_border(), vmin=vmin2, vmax=vmax2)
    plt.title(title2)
    plt.show()


class TestGRF(TestCase):

    def setUp(self) -> None:
        set_resume_state(False)
        self.g = GRF()
        self.grid = self.g.field.get_grid()
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        self.f = self.g.field
        self.cov = self.g.get_Sigma()
        self.mu = self.g.get_mu()

    def test_prior_matern_covariance(self):
        print("S1")
        plotf(self, v1=self.g.get_mu(), v2 = np.diag(self.g.get_Sigma()), vmin1=10, vmax1=36, vmin2=0, vmax2=1)
        print("END S1")

    def test_assimilate(self):
        # c2: one
        print("S2")
        dataset = np.array([[6000, 8000, 0, 30]])
        self.g.assimilate_data(dataset)
        plotf(self, v1=self.g.get_mu(), v2=np.diag(self.g.get_Sigma()), vmin1=10, vmax1=36, vmin2=0, vmax2=1)

        # c3: multiple
        dataset = np.array([[6500, 7000,  0, 35],
                            [7000, 8000, 0, 20],
                            [7200, 8500, 0, 15],
                            [7600, 8800, 0, 20]])
        self.g.assimilate_data(dataset)
        plotf(self, v1=self.g.get_mu(), v2=np.diag(self.g.get_Sigma()), vmin1=10, vmax1=36, vmin2=0, vmax2=1)
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

    def test_resuming_features(self):
        # s1: start assimilating multiple steps before resuming.
        print("Start S4")
        set_resume_state(False)
        g1 = GRF()
        dataset = np.array([[6500, 7000,  0, 35],
                            [7000, 8000, 0, 20],
                            [7200, 8500, 0, 15],
                            [7600, 8800, 0, 20]])
        g1.assimilate_data(dataset)
        plotf(self, v1=g1.get_mu(), v2=np.diag(g1.get_Sigma()), vmin1=10, vmax1=36, vmin2=0, vmax2=1)

        dataset = np.array([[7500, 7000,  0, 35],
                            [8000, 8000, 0, 20],
                            [8200, 8500, 0, 15],
                            [8600, 8800, 0, 20]])
        g1.assimilate_data(dataset)
        plotf(self, v1=g1.get_mu(), v2=np.diag(g1.get_Sigma()), vmin1=10, vmax1=36, vmin2=0, vmax2=1)

        dataset = np.array([[8500, 7000,  0, 35],
                            [9000, 8000, 0, 20],
                            [9200, 8500, 0, 15],
                            [9600, 8800, 0, 20]])
        g1.assimilate_data(dataset)
        plotf(self, v1=g1.get_mu(), v2=np.diag(g1.get_Sigma()), vmin1=10, vmax1=36, vmin2=0, vmax2=1)

        set_resume_state(False)
        g2 = GRF()
        plotf(self, v1=g2.get_mu(), v2=np.diag(g2.get_Sigma()), vmin1=10, vmax1=36, vmin2=0, vmax2=1)

        set_resume_state(True)
        g3 = GRF()
        plotf(self, v1=g3.get_mu(), v2=np.diag(g3.get_Sigma()), vmin1=10, vmax1=36, vmin2=0, vmax2=1)

        set_resume_state(False)
        g4 = GRF()
        plotf(self, v1=g4.get_mu(), v2=np.diag(g4.get_Sigma()), vmin1=10, vmax1=36, vmin2=0, vmax2=1)
        print("End S4")

    # def test_get_ei_field_partial(self):
    #     print("S4")
    #     loc = np.array([6000, 8000])
    #     ind_now = self.f.get_ind_from_location(loc)
    #     loc_now = self.f.get_location_from_ind(ind_now)
    #     ind_neighbours_layer1 = self.f.get_neighbour_indices(ind_now)
    #     ind_neighbours_layer2 = self.f.get_neighbour_indices(ind_neighbours_layer1)
    #     ind_neighbours_layer3 = self.f.get_neighbour_indices(ind_neighbours_layer2)
    #     ind_neighbours_layer4 = self.f.get_neighbour_indices(ind_neighbours_layer3)
    #     ind_neighbours_layer5 = self.f.get_neighbour_indices(ind_neighbours_layer4)
    #
    #     eibv, ivr = self.g.get_ei_field_partial(ind_neighbours_layer5)
    #     plotf(self, v1=eibv, v2=ivr, vmin1=0, vmax1=1, vmin2=0, vmax2=1)
    #
    #     eibv, ivr = self.g.get_ei_field_total()
    #     plotf(self, v1=eibv, v2=ivr, vmin1=0, vmax1=1, vmin2=0, vmax2=1)
    #
    #     # c2: with data assimilation
    #     dataset = np.array([[10000, 9000, 0, 10],
    #                         [12000, 8000, 0, 15],
    #                         [8000, 10000, 0, 13],
    #                         [2000, 2000, 0, 33],
    #                         [8000, 8000, 0, 26],
    #                         [4000, 8000, 0, 24]])
    #     self.g.assimilate_data(dataset)
    #     eibv, ivr = self.g.get_ei_field_partial(ind_neighbours_layer5)
    #     plotf(self, v1=eibv, v2=ivr, vmin1=0, vmax1=1, vmin2=0, vmax2=1)
    #     plotf(self, v1=self.g.get_mu(), v2=np.diag(self.g.get_Sigma()), vmin1=10, vmax1=36, vmin2=0, vmax2=1)
    #
    #     eibv, ivr = self.g.get_ei_field_total()
    #     plotf(self, v1=eibv, v2=ivr, vmin1=0, vmax1=1, vmin2=0, vmax2=1)
    #     plotf(self, v1=self.g.get_mu(), v2=np.diag(self.g.get_Sigma()), vmin1=10, vmax1=36, vmin2=0, vmax2=1)
    #     print("End S4")

    # def test_para_ei_field(self) -> None:
    #     eibv, ivr = self.g.get_ei_field_total()
    #     eibvp, ivrp = self.g.get_ei_field_para()
    #     testing.assert_array_equal(eibv, eibvp)
    #     testing.assert_array_equal(ivr, ivrp)
    #     pass

