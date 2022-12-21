from unittest import TestCase
from Planner.RRTSCV.RRTStarCV import RRTStarCV
from Config import Config
import matplotlib.pyplot as plt
import numpy as np
from Visualiser.TreePlotter import TreePlotter
from usr_func.set_resume_state import set_resume_state
from Visualiser.Visualiser import plotf_vector
# from matplotlib.cm import get_cmap
from matplotlib.pyplot import get_cmap


class TestRRTStar(TestCase):

    def setUp(self) -> None:
        set_resume_state(False)
        self.config = Config()
        self.rrtstar = RRTStarCV()
        self.tp = TreePlotter()
        self.cv = self.rrtstar.get_CostValley()
        self.field = self.cv.get_field()
        self.grid = self.field.get_grid()
        self.polygon_border = self.config.get_polygon_border()
        self.polygon_obstacle = self.config.get_polygon_obstacle()

    def test_weights_for_cost_on_trees(self) -> None:
        # self.cv.set_weight_eibv(.1)
        # self.cv.set_weight_ivr(1.9)

        self.cv.set_weight_eibv(1.9)
        self.cv.set_weight_ivr(.1)

        # self.cv.set_weight_eibv(1.)
        # self.cv.set_weight_ivr(1.)

        print("weight_EIBV: ", self.cv.get_eibv_weight(), " weight IVR: ", self.cv.get_ivr_weight())

        loc_now = np.array([1500, -2000])
        # loc_end = np.array([3000, 1000])
        self.cv.update_cost_valley()
        loc_end = self.cv.get_minimum_cost_location()

        wp = self.rrtstar.get_next_waypoint(loc_now, loc_end)
        nodes = self.rrtstar.get_tree_nodes()
        traj = self.rrtstar.get_trajectory()
        self.tp.update_trees(nodes)

        plt.figure(figsize=(15, 12))
        cv = self.cv.get_cost_field()
        plotf_vector(self.grid[:, 1], self.grid[:, 0], cv, xlabel='East', ylabel='North', title='RRTCV',
                     cbar_title="Cost", cmap=get_cmap("RdBu", 10))
        self.tp.plot_tree()
        plt.plot(traj[:, 1], traj[:, 0], 'k-', linewidth=10)
        plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'r-.')
        plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'r-.')

        plt.plot(loc_now[1], loc_now[0], 'r.', markersize=20)
        plt.plot(loc_end[1], loc_end[0], 'k*', markersize=20)
        plt.xlabel("x")
        plt.ylabel("y")
        # plt.savefig(os.getcwd() + "/../../fig/trees/rrtcv.png")
        plt.show()

