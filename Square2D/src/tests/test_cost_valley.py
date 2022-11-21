from unittest import TestCase
from CostValley.CostValley import CostValley
from Visualiser.Visualiser import plotf_vector
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse
import numpy as np
import math
from numpy import testing
from matplotlib.cm import get_cmap


class TestCostValley(TestCase):

    def setUp(self) -> None:
        self.cv = CostValley()
        self.grf = self.cv.get_grf_model()
        self.field = self.grf.field
        self.polygon_border = self.field.get_polygon_border()
        self.polygon_border = np.append(self.polygon_border, self.polygon_border[0, :].reshape(1, -1), axis=0)
        self.polygon_obstacle = self.field.get_polygon_obstacles()[0]
        self.polygon_obstacle = np.append(self.polygon_obstacle, self.polygon_obstacle[0, :].reshape(1, -1), axis=0)

        self.xlim, self.ylim = self.field.get_border_limits()
    # def test_minimum_cost_location(self):
    #     loc_m = self.cv.get_minimum_cost_location()
    #     cv = self.cv.get_cost_valley()
    #     id = np.argmin(cv)
    #     loc = self.grf.grid[id]
    #     self.assertIsNone(testing.assert_array_equal(loc, loc_m))
    #
    # def test_get_cost_at_location(self):
    #     loc = np.array([.1, .2])
    #     cost = self.cv.get_cost_at_location(loc)
    #
    # def test_get_cost_along_path(self):
    #     l1 = np.array([1.0, .0])
    #     l2 = np.array([.0, .0])
    #     c = self.cv.get_cost_along_path(l1, l2)

    def test_presentation(self):
        # dataset = np.array([[.0, .0, .0]])
        # self.grf.assimilate_data(dataset)
        # self.grf.get_ei_field()
        # self.cv.update_cost_valley(dataset[0, :2])

        grid = self.cv.get_grid()
        cv = self.cv.get_cost_valley()
        eibv = self.cv.get_eibv_field()
        ivr = self.cv.get_ivr_field()
        budget = self.cv.get_budget_field()
        Bu = self.cv.get_Budget()
        angle = Bu.get_ellipse_rotation_angle()
        mid = Bu.get_ellipse_middle_location()
        a = Bu.get_ellipse_a()
        b = Bu.get_ellipse_b()
        c = Bu.get_ellipse_c()
        e = Ellipse(xy=(mid[0], mid[1]), width=2*a, height=2*np.sqrt(a**2-c**2),
                    angle=math.degrees(angle), edgecolor='r', fc='None', lw=2)

        azimuth = self.cv.get_direction_field()
        obs = self.cv.get_obstacle_field()

        plt.figure(figsize=(15, 12))
        plotf_vector(grid[:, 0], grid[:, 1], cv, xlabel='x', ylabel='y', title='Direction', cbar_title="Cost",
                     cmap=get_cmap("RdBu", 10), vmin=-1, vmax=1.1, stepsize=.4)
        # plt.plot(self.polygon_border[:, 0], self.polygon_border[:, 1], 'r-.')
        plt.plot(self.polygon_obstacle[:, 0], self.polygon_obstacle[:, 1], 'r-.')
        # plt.plot(0, 0, 'r.', markersize=20)
        # plt.plot(0, 1, 'k*', markersize=20)
        # plt.xlim(self.xlim)
        # plt.ylim(self.ylim)
        plt.savefig("/Users/yaoling/Downloads/trees/dir.png")
        plt.show()

        # fig = plt.figure(figsize=(60, 10))
        # gs = GridSpec(nrows=1, ncols=6)
        # ax = fig.add_subplot(gs[0])
        # plotf_vector(grid[:, 0], grid[:, 1], cv)
        #              # polygon_border=self.cv.field.get_polygon_border(),
        #              # polygon_obstacle=self.g.field.get_polygon_obstacles())
        # plt.title("Cost Valley")
        #
        # ax = fig.add_subplot(gs[1])
        # plotf_vector(grid[:, 0], grid[:, 1], eibv)
        #              # polygon_border=self.g.field.get_polygon_border(),
        #              # polygon_obstacle=self.g.field.get_polygon_obstacles())
        # plt.title("EIBV")
        #
        # ax = fig.add_subplot(gs[2])
        # plotf_vector(grid[:, 0], grid[:, 1], ivr)
        #              # polygon_border=self.g.field.get_polygon_border(),
        #              # polygon_obstacle=self.g.field.get_polygon_obstacles())
        # plt.title("IVR")
        #
        # ax = fig.add_subplot(gs[3])
        # plotf_vector(grid[:, 0], grid[:, 1], budget)
        # # polygon_border=self.g.field.get_polygon_border(),
        # # polygon_obstacle=self.g.field.get_polygon_obstacles())
        # plt.title("Budget")
        # plt.gca().add_patch(e)
        #
        # ax = fig.add_subplot(gs[4])
        # plotf_vector(grid[:, 0], grid[:, 1], azimuth)
        # # polygon_border=self.g.field.get_polygon_border(),
        # # polygon_obstacle=self.g.field.get_polygon_obstacles())
        # plt.title("Direction")
        #
        # ax = fig.add_subplot(gs[5])
        # plotf_vector(grid[:, 0], grid[:, 1], self.grf.get_mu())
        # # polygon_border=self.g.field.get_polygon_border(),
        # # polygon_obstacle=self.g.field.get_polygon_obstacles())
        # plt.title("mean")
        # plt.show()

    # def plot_cost_valley(self):
    #     grid = self.cv.get_grid()
    #     cv = self.cv.get_cost_valley()
    #     eibv = self.cv.get_eibv_field()
    #     ivr = self.cv.get_ivr_field()
    #     budget = self.cv.get_budget_field()
    #     Bu = self.cv.get_Budget()
    #     angle = Bu.get_ellipse_rotation_angle()
    #     mid = Bu.get_ellipse_middle_location()
    #     a = Bu.get_ellipse_a()
    #     b = Bu.get_ellipse_b()
    #     c = Bu.get_ellipse_c()
    #     e = Ellipse(xy=(mid[0], mid[1]), width=2*a, height=2*np.sqrt(a**2-c**2),
    #                 angle=math.degrees(angle), edgecolor='r', fc='None', lw=2)
    #
    #     azimuth = self.cv.get_direction_field()
    #     obs = self.cv.get_obstacle_field()
    #
    #     fig = plt.figure(figsize=(60, 10))
    #     gs = GridSpec(nrows=1, ncols=6)
    #     ax = fig.add_subplot(gs[0])
    #     plotf_vector(grid[:, 0], grid[:, 1], cv)
    #                  # polygon_border=self.cv.field.get_polygon_border(),
    #                  # polygon_obstacle=self.g.field.get_polygon_obstacles())
    #     plt.title("Cost Valley")
    #
    #     ax = fig.add_subplot(gs[1])
    #     plotf_vector(grid[:, 0], grid[:, 1], eibv)
    #                  # polygon_border=self.g.field.get_polygon_border(),
    #                  # polygon_obstacle=self.g.field.get_polygon_obstacles())
    #     plt.title("EIBV")
    #
    #     ax = fig.add_subplot(gs[2])
    #     plotf_vector(grid[:, 0], grid[:, 1], ivr)
    #                  # polygon_border=self.g.field.get_polygon_border(),
    #                  # polygon_obstacle=self.g.field.get_polygon_obstacles())
    #     plt.title("IVR")
    #
    #     ax = fig.add_subplot(gs[3])
    #     plotf_vector(grid[:, 0], grid[:, 1], budget)
    #     # polygon_border=self.g.field.get_polygon_border(),
    #     # polygon_obstacle=self.g.field.get_polygon_obstacles())
    #     plt.title("Budget")
    #     plt.gca().add_patch(e)
    #
    #     ax = fig.add_subplot(gs[4])
    #     plotf_vector(grid[:, 0], grid[:, 1], azimuth)
    #     # polygon_border=self.g.field.get_polygon_border(),
    #     # polygon_obstacle=self.g.field.get_polygon_obstacles())
    #     plt.title("Direction")
    #
    #     ax = fig.add_subplot(gs[5])
    #     plotf_vector(grid[:, 0], grid[:, 1], self.grf.get_mu())
    #     # polygon_border=self.g.field.get_polygon_border(),
    #     # polygon_obstacle=self.g.field.get_polygon_obstacles())
    #     plt.title("mean")
    #     plt.show()
    #
    # def test_update_cost_valley(self):
    #     self.plot_cost_valley()
    #
    #     # s1: move and sample
    #     dataset = np.array([[.0, .0, .0]])
    #     self.grf.assimilate_data(dataset)
    #     self.grf.get_ei_field()
    #     self.cv.update_cost_valley(dataset[0, :2])
    #     self.plot_cost_valley()
    #
    #     # s2: move more and sample
    #     dataset = np.array([[.0, 1., .0]])
    #     self.grf.assimilate_data(dataset)
    #     self.grf.get_ei_field()
    #     self.cv.update_cost_valley(dataset[0, :2])
    #     self.plot_cost_valley()
    #
    #     # s3: move more and sample
    #     dataset = np.array([[1., 1., .2]])
    #     self.grf.assimilate_data(dataset)
    #     self.grf.get_ei_field()
    #     self.cv.update_cost_valley(dataset[0, :2])
    #     self.plot_cost_valley()
    #
    #     # s4: move more and sample
    #     dataset = np.array([[.4, .6, .1]])
    #     self.grf.assimilate_data(dataset)
    #     self.grf.get_ei_field()
    #     self.cv.update_cost_valley(dataset[0, :2])
    #     self.plot_cost_valley()
    #
    #     # s5: move more and sample
    #     dataset = np.array([[.0, .0, .0]])
    #     self.grf.assimilate_data(dataset)
    #     self.grf.get_ei_field()
    #     self.cv.update_cost_valley(dataset[0, :2])
    #     self.plot_cost_valley()
    #
    #     # s6: move final steps and sample
    #     dataset = np.array([[.2, .9, .0]])
    #     self.grf.assimilate_data(dataset)
    #     self.grf.get_ei_field()
    #     self.cv.update_cost_valley(dataset[0, :2])
    #     self.plot_cost_valley()
    #
    #     # s6: move final steps and sample
    #     # dataset = np.array([[.2, .7, .0]])
    #     # self.grf.assimilate_data(dataset)
    #     # self.grf.get_ei_field()
    #     # self.cv.update_cost_valley(dataset[0, :2])
    #     # self.plot_cost_valley()

