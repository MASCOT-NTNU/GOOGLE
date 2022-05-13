"""
This script builds the budget
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-26
"""

from Config.Config import FILEPATH, BUDGET, X_HOME, Y_HOME, CRASH
import numpy as np
import pandas as pd
import math
from matplotlib.patches import Ellipse
from usr_func import Polygon, LineString, Point
from numba import vectorize
import time


BUDGET_GOHOME_MARGIN = 150


@vectorize(['float32(float32, float32, float32, float32, float32, float32, float32)'])
def get_utility_ellipse(x, y, xm, ym, ellipse_a, ellipse_b, angle):
    xn = x - xm
    yn = y - ym
    xr = xn * np.cos(angle) - yn * np.sin(angle)
    yr = xn * np.sin(angle) + yn * np.cos(angle)
    u = (xr / ellipse_b)**2 + (yr / ellipse_a)**2
    return u


class Budget:

    def __init__(self):
        if not CRASH:
            self.budget_left = BUDGET
            self.gohome_alert = False
        else:
            self.budget_left = np.loadtxt(FILEPATH + "Backup/budget.txt")
            self.gohome_alert = np.loadtxt(FILEPATH + "Backup/gohome.txt")
        self.load_grf_grid()

    def load_grf_grid(self):
        self.grf_grid = pd.read_csv(FILEPATH+"Config/GRFGrid.csv").to_numpy()
        print("B1: GRF Grid is loaded successfully!")

    def update_budget(self, x_current, y_current, x_previous, y_previous):
        distance_travelled = np.sqrt((x_current - x_previous)**2 +
                                     (y_current - y_previous)**2)
        self.budget_left = self.budget_left - distance_travelled
        self.x_middle = (x_current + X_HOME) / 2
        self.y_middle = (y_current + Y_HOME) / 2
        dx = X_HOME - x_current
        dy = Y_HOME - y_current
        self.angle = np.math.atan2(dx, dy)
        self.ellipse_a = self.budget_left / 2
        self.ellipse_c = np.sqrt(dx**2 + dy**2) / 2
        if self.ellipse_a > self.ellipse_c + BUDGET_GOHOME_MARGIN:
            self.ellipse_b = np.sqrt(self.ellipse_a**2 - self.ellipse_c**2)
            self.ellipse = Ellipse(xy=(self.y_middle, self.x_middle), width=2 * self.ellipse_a,
                                   height=2 * self.ellipse_b, angle=math.degrees(self.angle))
            self.vertices = self.ellipse.get_verts() #TODO: different x, y from grf grid
            self.polygon_budget_ellipse = Polygon(np.fliplr(self.vertices))
            self.line_budget_ellipse = LineString(np.fliplr(self.vertices))
        else:
            self.ellipse_b = 0
            self.ellipse = Ellipse(xy=(self.y_middle, self.x_middle), width=2 * self.ellipse_a,
                                   height=2 * self.ellipse_b, angle=math.degrees(self.angle))
            self.vertices = self.ellipse.get_verts()
            self.polygon_budget_ellipse = Polygon([])
            self.line_budget_ellipse = LineString([])
            self.gohome_alert = True
        np.savetxt(FILEPATH + "Backup/budget.txt", np.array([self.budget_left]))
        np.savetxt(FILEPATH + "Backup/gohome.txt", np.array([self.gohome_alert]))

    def get_budget_field(self):
        t1 = time.time()
        self.budget_field = np.zeros_like(self.grf_grid[:, 0])
        xm = self.x_middle
        ym = self.y_middle
        ea = self.ellipse_a
        eb = self.ellipse_b
        angle = self.angle
        x, y, xm, ym, ea, eb, angle = map(np.float32, [self.grf_grid[:, 0], self.grf_grid[:, 1],
                                                       xm, ym, ea, eb, angle])
        if not self.gohome_alert:
            self.u = get_utility_ellipse(x, y, xm, ym, ea, eb, angle)
            ind_penalty = self.get_ind_penalty()
            ind_penalty = np.where(ind_penalty == True)[0]
            self.budget_field[ind_penalty] = self.u[ind_penalty] ** 2
        else:
            self.budget_field = np.ones_like(x) * np.inf
        t2 = time.time()
        print("Budget filed takes: ", t2 - t1)
        print("Budget remaining: ", self.budget_left)

    def get_ind_penalty(self):
        penalty = np.ones(len(self.grf_grid))
        for i in range(len(self.grf_grid)):
            point = Point(self.grf_grid[i, 0], self.grf_grid[i, 1])
            if self.polygon_budget_ellipse.contains(point):
                penalty[i] = 0
        return penalty

    def check_budget(self):
        x_prev = 1000
        y_prev = -2000
        x_now = 2000
        y_now = -300
        self.budget_left = self.budget_left - 4150
        print("Budget left: ", self.budget_left)
        self.update_budget(x_now, y_now, x_prev, y_prev)
        print("Budget left: ", self.budget_left)
        print("Distance between currnet, home: ", np.sqrt((X_HOME - x_now)**2 +
                                                          (Y_HOME - y_now)**2))
        self.get_budget_field()
        import matplotlib.pyplot as plt
        from matplotlib.patches import Ellipse
        import math
        # plt.plot(b.grf_grid[:, 1], b.grf_grid[:, 0], 'k.')
        from matplotlib.cm import get_cmap
        plt.scatter(self.grf_grid[:, 1], self.grf_grid[:, 0], c=self.budget_field, cmap=get_cmap("BrBG", 7),
                    vmin=1, vmax=10, alpha=.5)
        plt.plot(y_prev, x_prev, 'ro', ms=10, label='previous waypoint')
        plt.plot(y_now, x_now, 'cs', ms=10, label='current waypoint')
        plt.plot(Y_HOME, X_HOME, 'b^', ms=10, label='home')
        ellipse = Ellipse(xy=(self.y_middle, self.x_middle), width=2*self.ellipse_a,
                          height=2*self.ellipse_b, angle=math.degrees(self.angle),
                          edgecolor='r', fc='None', lw=2)
        plt.gca().add_patch(ellipse)
        plt.colorbar()
        plt.xlim([np.min(self.grf_grid[:, 1]), np.max(self.grf_grid[:, 1])])
        plt.ylim([np.min(self.grf_grid[:, 0]), np.max(self.grf_grid[:, 0])])
        plt.show()
        print("GOHOME: ", self.gohome_alert)
    # def get_budget_field(self, current_location, goal_location, budget):
    #     t1 = time.time()
    #     if budget >= BUDGET_MARGIN:
    #         self.budget_middle_location = self.get_middle_location(current_location, goal_location)
    #         self.budget_ellipse_angle = self.get_angle_between_locations(current_location, goal_location)
    #         self.budget_ellipse_a = budget / 2
    #         self.budget_ellipse_c = get_distance_between_xy_locations(current_location, goal_location) / 2
    #         self.budget_ellipse_b = np.sqrt(self.budget_ellipse_a ** 2 - self.budget_ellipse_c ** 2)
    #         print("a: ", self.budget_ellipse_a, "b: ", self.budget_ellipse_b, "c: ", self.budget_ellipse_c)
    #         if self.budget_ellipse_b > BUDGET_ELLIPSE_B_MARGIN:
    #             x_wgs = self.coordinates_xyz[:, 0] - self.budget_middle_location.x
    #             y_wgs = self.coordinates_xyz[:, 1] - self.budget_middle_location.y
    #             self.cost_budget = []
    #             for i in range(self.coordinates_xyz.shape[0]):
    #                 x_usr = (x_wgs[i] * np.cos(self.budget_ellipse_angle) -
    #                          y_wgs[i] * np.sin(self.budget_ellipse_angle))
    #                 y_usr = (x_wgs[i] * np.sin(self.budget_ellipse_angle) +
    #                          y_wgs[i] * np.cos(self.budget_ellipse_angle))
    #                 if (x_usr / self.budget_ellipse_b) ** 2 + (y_usr / self.budget_ellipse_a) ** 2 <= 1:
    #                     self.cost_budget.append(0)
    #                 else:
    #                     self.cost_budget.append(np.inf)
    #             self.cost_budget = np.array(self.cost_budget)
    #         else:
    #             self.knowledge.gohome = True
    #     else:
    #         self.knowledge.gohome = True
    #     t2 = time.time()
    #     print("budget field consumed: ", t2 - t1)


if __name__ == "__main__":
    b = Budget()
    b.check_budget()
    # b.update_budget()

# #%%
# import matplotlib.pyplot as plt
# from matplotlib.patches import Ellipse
# from matplotlib.cm import get_cmap
#
# xv = np.linspace(0, 1, 25)
# yv = np.linspace(0, 1, 25)
# xx, yy = np.meshgrid(xv, yv)
# x = xx.reshape(-1, 1).astype(np.float32)
# y = yy.reshape(-1, 1).astype(np.float32)
# u = get_utility_ellipse(x, y, .5, .5, .5, .1, 0)
# # xu = x - .5
# # yu = y - .5
# # u = (yu/.5)**2 + (xu/.1)**2
#
# ellipse = Ellipse(xy=(.5, .5), width=.5, height=.1, angle=0, edgecolor='r', fc='None', lw=2)
#
# plt.gca().add_patch(ellipse)
# plt.scatter(y, x, c=u, s=50, cmap=get_cmap('BrBG', 10), vmin=0, vmax=3)
# plt.colorbar()
# plt.show()
#
# plt.plot(u)
# plt.show()




