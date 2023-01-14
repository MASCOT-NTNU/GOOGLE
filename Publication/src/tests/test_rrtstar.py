from unittest import TestCase
from Planner.RRTSCV.RRTStarCV import RRTStarCV
from Config import Config
import matplotlib.pyplot as plt
import numpy as np
from Visualiser.TreePlotter import TreePlotter
from Visualiser.Visualiser import plotf_vector
# from matplotlib.cm import get_cmap
from matplotlib.pyplot import get_cmap
from matplotlib.patches import Ellipse
import math


class TestRRTStar(TestCase):

    def setUp(self) -> None:
        self.config = Config()
        sigma = .1
        nugget = .01
        # self.rrtstar = RRTStarCV(sigma=sigma, nugget=nugget)
        self.rrtstar = RRTStarCV(weight_eibv=2., weight_ivr=.0, sigma=sigma, nugget=nugget, budget_mode=True)
        # self.rrtstar = RRTStarCV(weight_eibv=.0, weight_ivr=2., sigma=sigma, nugget=nugget)
        self.tp = TreePlotter()
        self.cv = self.rrtstar.get_CostValley()
        self.field = self.cv.get_field()
        self.grid = self.field.get_grid()
        self.polygon_border = self.config.get_polygon_border()
        self.polygon_obstacle = self.config.get_polygon_obstacle()

    def test_weights_for_cost_on_trees(self) -> None:
        print("weight_EIBV: ", self.cv.get_eibv_weight(), " weight IVR: ", self.cv.get_ivr_weight())

        loc_now = np.array([2000, -2000])
        # loc_end = np.array([3000, 1000])
        self.cv.update_cost_valley(loc_now=loc_now)
        loc_end = self.cv.get_minimum_cost_location()

        wp = self.rrtstar.get_next_waypoint(loc_now, loc_end)
        nodes = self.rrtstar.get_tree_nodes()
        traj = self.rrtstar.get_trajectory()
        self.tp.update_trees(nodes)

        """ Get polygon for budget constraint. """
        Bu = self.cv.get_Budget()
        angle = Bu.get_ellipse_rotation_angle()
        mid = Bu.get_ellipse_middle_location()
        a = Bu.get_ellipse_a()
        b = Bu.get_ellipse_b()
        c = Bu.get_ellipse_c()
        e = Ellipse(xy=(mid[1], mid[0]), width=2 * a, height=2 * np.sqrt(a ** 2 - c ** 2),
                    angle=math.degrees(angle), edgecolor='r', fc='None', lw=2)

        plt.figure(figsize=(15, 12))
        cv = self.cv.get_cost_field()
        plotf_vector(self.grid[:, 1], self.grid[:, 0], cv, xlabel='East', ylabel='North', title='RRTCV',
                     cbar_title="Cost", cmap=get_cmap("RdBu", 10), vmin=0, vmax=1)
        self.tp.plot_tree()
        plt.plot(traj[:, 1], traj[:, 0], 'k-', linewidth=10)
        plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'r-.')
        plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'r-.')

        plt.plot(loc_now[1], loc_now[0], 'r.', markersize=20)
        plt.plot(loc_end[1], loc_end[0], 'k*', markersize=20)
        plt.gca().add_patch(e)
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.savefig(os.getcwd() + "/../../fig/trees/rrtcv.png")
        plt.show()

