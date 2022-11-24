from unittest import TestCase
from Config import Config
from CostValley.Budget import Budget
from Field import Field
import numpy as np
import matplotlib.pyplot as plt
import math
from numpy import testing
from matplotlib.cm import get_cmap
from matplotlib.patches import Ellipse


class TestBudget(TestCase):

    def setUp(self) -> None:
        self.c = Config()
        self.f = Field()
        self.grid = self.f.get_grid()
        self.b = Budget(self.grid)

    def test_locations(self):
        # c1: test goal location
        l = self.b.get_goal()
        lc = self.c.get_loc_end()
        self.assertEqual(lc[0], l[0])
        self.assertEqual(lc[1], l[1])

        # c2: test now location
        x, y = self.c.get_loc_start()
        l = self.b.get_loc_prev()
        self.assertEqual(x, l[0])
        self.assertEqual(y, l[1])

        # c3: test previous location
        l = self.b.get_loc_now()
        self.assertEqual(x, l[0])
        self.assertEqual(y, l[1])

    def run_update_budget(self, loc, ls, BU, bf):
        l = self.b.get_loc_now()
        self.assertEqual(loc[0], l[0])
        self.assertEqual(loc[1], l[1])

        l = self.b.get_loc_prev()
        self.assertEqual(loc[0], l[0])
        self.assertEqual(loc[1], l[1])

        # s2: test remaining budget
        b = self.b.get_budget()
        br = BU - np.sqrt((loc[0] - ls[0]) ** 2 +
                          (loc[1] - ls[1]) ** 2)
        self.assertEqual(b, br)

        # s3: test ellipse
        goal = self.b.get_goal()
        alpha = self.b.get_ellipse_rotation_angle()
        mid = self.b.get_ellipse_middle_location()
        a = self.b.get_ellipse_a()
        b = self.b.get_ellipse_b()
        c = self.b.get_ellipse_c()
        self.assertEqual(a * 2, br)
        dx = goal[0] - loc[0]
        dy = goal[1] - loc[1]
        ct = np.sqrt(dx ** 2 + dy ** 2)
        self.assertEqual(c * 2, ct)
        if not self.b.get_go_home_alert():
            self.assertIsNone(testing.assert_almost_equal(b ** 2, (br / 2) ** 2 - (ct / 2) ** 2))
        angle = math.atan2(dx, dy)
        self.assertEqual(alpha, angle)
        self.assertEqual(mid[0], (goal[0] + loc[0]) / 2)
        self.assertEqual(mid[1], (goal[1] + loc[1]) / 2)

        e = Ellipse(xy=(mid[1], mid[0]), width=2*a, height=2*np.sqrt((br / 2) ** 2 - (ct / 2) ** 2),
                    angle=math.degrees(angle), edgecolor='r', fc='None', lw=2)
        if not self.b.get_go_home_alert():
            plt.scatter(self.grid[:, 1], self.grid[:, 0], c=bf, cmap=get_cmap("RdBu", 10), vmin=-1, vmax=20)
            plt.colorbar()
        else:
            plt.plot(self.grid[:, 1], self.grid[:, 0], 'k.')
        plt.plot(loc[1], loc[0], 'k.', markersize=20)
        plt.plot(goal[1], goal[0], 'b.', markersize=20)
        plt.gca().add_patch(e)
        plt.show()

    def test_get_budget_field(self):
        # c0: starting location
        BU = self.b.get_budget()
        self.b.set_loc_prev(self.c.get_loc_start())
        ls = self.b.get_loc_prev()
        loc = np.array([7000, 8000])
        bf = self.b.get_budget_field(loc[0], loc[1])
        self.run_update_budget(loc, ls, BU, bf)

        # c1: move to first location,
        BU = self.b.get_budget()
        ls = self.b.get_loc_prev()
        loc = np.array([8000, 8500])
        bf = self.b.get_budget_field(loc[0], loc[1])
        self.run_update_budget(loc, ls, BU, bf)

        # c2: consume a little budget
        loc = np.array([9500., 9000.])
        BU = self.b.get_budget()
        ls = self.b.get_loc_prev()
        bf = self.b.get_budget_field(loc[0], loc[1])
        self.run_update_budget(loc, ls, BU, bf)

        # c3: consume more budget
        loc = np.array([10000., 9500.])
        BU = self.b.get_budget()
        ls = self.b.get_loc_prev()
        bf = self.b.get_budget_field(loc[0], loc[1])
        self.run_update_budget(loc, ls, BU, bf)

        # # c4: consume more budget
        # loc = np.array([4500, 4500])
        # BU = self.b.get_budget()
        # ls = self.b.get_loc_prev()
        # bf = self.b.get_budget_field(loc[0], loc[1])
        # self.run_update_budget(loc, ls, BU, bf)

        # # c5: consume last budget
        # loc = np.array([5500, 4700])
        # BU = self.b.get_budget()
        # ls = self.b.get_loc_prev()
        # bf = self.b.get_budget_field(loc[0], loc[1])
        # self.run_update_budget(loc, ls, BU, bf)

        # c5: consume last budget
        # loc = np.array([11700, 10000])
        # BU = self.b.get_budget()
        # ls = self.b.get_loc_prev()
        # bf = self.b.get_budget_field(loc[0], loc[1])
        # self.run_update_budget(loc, ls, BU, bf)

        # c6: consume last budget
        # loc = np.array([12000, 10000])
        # BU = self.b.get_budget()
        # ls = self.b.get_loc_prev()
        # bf = self.b.get_budget_field(loc[0], loc[1])
        # self.run_update_budget(loc, ls, BU, bf)

        # c7: consume last budget
        # loc = np.array([0.1, 0.9])
        # BU = self.b.get_budget()
        # ls = self.b.get_loc_prev()
        # bf = self.b.get_budget_field(loc[0], loc[1])
        # self.run_update_budget(loc, ls, BU, bf)


