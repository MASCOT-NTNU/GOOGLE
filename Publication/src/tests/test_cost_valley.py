"""
Unittest for cost valley.
"""
from unittest import TestCase
from CostValley.CostValley import CostValley
from Config import Config
# from Visualiser.Visualiser import plotf_vector
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import numpy as np
import math
from numpy import testing
from matplotlib.cm import get_cmap


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


class TestCostValley(TestCase):

    def setUp(self) -> None:
        self.c = Config()
        self.cv = CostValley()
        self.grf = self.cv.get_grf_model()
        self.field = self.grf.field
        self.polygon_border = self.c.get_polygon_border()
        self.polygon_obstacle = self.c.get_polygon_obstacle()
        self.xlim, self.ylim = self.field.get_border_limits()

    def test_weights(self):
        loc_now = np.array([1000, -1000])
        # c1: equal weights
        print("weight_EIBV: ", self.cv.get_eibv_weight(), " weight IVR: ", self.cv.get_ivr_weight())
        self.cv.update_cost_valley(loc_now)
        self.plot_cost_valley()

        # c2: more EIBV
        self.cv.set_weight_eibv(1.9)
        self.cv.set_weight_ivr(.1)
        print("weight_EIBV: ", self.cv.get_eibv_weight(), " weight IVR: ", self.cv.get_ivr_weight())
        self.cv.update_cost_valley(loc_now)
        self.plot_cost_valley()

        # c3: more IVR
        self.cv.set_weight_eibv(.1)
        self.cv.set_weight_ivr(1.9)
        print("weight_EIBV: ", self.cv.get_eibv_weight(), " weight IVR: ", self.cv.get_ivr_weight())
        self.cv.update_cost_valley(loc_now)
        self.plot_cost_valley()

    # def test_minimum_cost_location(self):
    #     print("S1")
    #     loc_m = self.cv.get_minimum_cost_location()
    #     cv = self.cv.get_cost_field()
    #     id = np.argmin(cv)
    #     loc = self.grf.grid[id]
    #     self.assertIsNone(testing.assert_array_equal(loc, loc_m))
    #     print("End S1")

    # def test_get_cost_at_location(self):
    #     print("S2")
    #     loc = np.array([2000, -2000])
    #     cost = self.cv.get_cost_at_location(loc)
    #     print("End S2")

    # def test_get_cost_along_path(self):
    #     print("S3")
    #     l1 = np.array([1000, -1000])
    #     l2 = np.array([2000, -200])
    #     c = self.cv.get_cost_along_path(l1, l2)
    #     print("End S3")

    def plot_cost_valley(self):
        grid = self.cv.get_grid()
        cv = self.cv.get_cost_field()
        eibv = self.cv.get_eibv_field()
        ivr = self.cv.get_ivr_field()
        budget = self.cv.get_budget_field()
        Bu = self.cv.get_Budget()
        angle = Bu.get_ellipse_rotation_angle()
        mid = Bu.get_ellipse_middle_location()
        a = Bu.get_ellipse_a()
        b = Bu.get_ellipse_b()
        c = Bu.get_ellipse_c()
        e = Ellipse(xy=(mid[1], mid[0]), width=2*a, height=2*np.sqrt(a**2-c**2),
                    angle=math.degrees(angle), edgecolor='r', fc='None', lw=2)

        # azimuth = self.cv.get_direction_field()

        fig = plt.figure(figsize=(30, 5))
        gs = GridSpec(nrows=1, ncols=6)
        ax = fig.add_subplot(gs[0])
        plotf_vector(grid[:, 1], grid[:, 0], cv, vmin=0, vmax=4)
        plt.title("Cost Valley")

        ax = fig.add_subplot(gs[1])
        plotf_vector(grid[:, 1], grid[:, 0], eibv, vmin=0, vmax=1)
        plt.title("EIBV")

        ax = fig.add_subplot(gs[2])
        plotf_vector(grid[:, 1], grid[:, 0], ivr, vmin=0, vmax=1)
        plt.title("IVR")

        ax = fig.add_subplot(gs[3])
        plotf_vector(grid[:, 1], grid[:, 0], budget, vmin=0, vmax=1)
        plt.title("Budget")
        plt.gca().add_patch(e)

        # ax = fig.add_subplot(gs[4])
        # plotf_vector(grid[:, 1], grid[:, 0], azimuth, vmin=0, vmax=1)
        # plt.title("Direction")

        ax = fig.add_subplot(gs[5])
        plotf_vector(grid[:, 1], grid[:, 0], self.grf.get_mu(), vmin=10, vmax=35)
        plt.title("mean")
        plt.show()

    # def test_update_cost_valley(self):
    #     print("S4")
    #     self.plot_cost_valley()
    #
    #     # s1: move and sample
    #     dataset = np.array([[1000, -1000, 0, 20]])
    #     self.grf.assimilate_data(dataset)
    #     self.cv.update_cost_valley(dataset[0, :2])
    #     self.plot_cost_valley()
    #
    #     # s2: move more and sample
    #     dataset = np.array([[1500, -1500, 0, 15]])
    #     self.grf.assimilate_data(dataset)
    #     self.cv.update_cost_valley(dataset[0, :2])
    #     self.plot_cost_valley()
    #
    #     # s3: move more and sample
    #     dataset = np.array([[2000, -2000, 0, 20]])
    #     self.grf.assimilate_data(dataset)
    #     self.cv.update_cost_valley(dataset[0, :2])
    #     self.plot_cost_valley()
    #
    #     # s4: move more and sample
    #     dataset = np.array([[2500, -1500, 0, 22]])
    #     self.grf.assimilate_data(dataset)
    #     self.cv.update_cost_valley(dataset[0, :2])
    #     self.plot_cost_valley()
    #
    #     # # s5: move more and sample
    #     # dataset = np.array([[8600, 9000, 0, 25]])
    #     # self.grf.assimilate_data(dataset)
    #     # self.cv.update_cost_valley(dataset[0, :2])
    #     # self.plot_cost_valley()
    #     #
    #     # # s6: move final steps and sample
    #     # dataset = np.array([[9000, 9200, 0, 25]])
    #     # self.grf.assimilate_data(dataset)
    #     # self.cv.update_cost_valley(dataset[0, :2])
    #     # self.plot_cost_valley()
    #     #
    #     # # s6: move final steps and sample
    #     # dataset = np.array([[9200, 9500, 0, 25]])
    #     # self.grf.assimilate_data(dataset)
    #     # self.cv.update_cost_valley(dataset[0, :2])
    #     # self.plot_cost_valley()
    #     #
    #     # # s6: move final steps and sample
    #     # dataset = np.array([[9500, 9800, 0, 10]])
    #     # self.grf.assimilate_data(dataset)
    #     # self.cv.update_cost_valley(dataset[0, :2])
    #     # self.plot_cost_valley()
    #     print("End S4")



