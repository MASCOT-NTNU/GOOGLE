"""
This script builds the RRTStarCV
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-26
"""
import matplotlib.pyplot as plt

from GOOGLE.Simulation_2DNidelva.Config.Config import FILEPATH
from usr_func import Polygon, Point, LineString
import pandas as pd
import numpy as np
import time

# == Setup
GOAL_SAMPLE_RATE = .01
MAX_ITERATION = 500
STEPSIZE = 200
NEIGHBOUR_RADIUS = 250
HOME_RADIUS = 100
FAIL_TO_GENERATE_PATH = 8888
# ==



class RRTStarCV:

    def __init__(self):
        self.load_grf_grid()
        self.load_random_locations()
        self.load_polygon_border_obstacle()
        self.trees = np.zeros([MAX_ITERATION, 4])
        self.counter_generations = 0
        pass

    def load_grf_grid(self):
        self.grf_grid = pd.read_csv(FILEPATH+"Config/GRFGrid.csv").to_numpy()
        print("R1: GRF Grid is loaded successfully!")

    def load_random_locations(self):
        self.random_locations = np.load(FILEPATH+"Config/RandomLocations.npy")
        self.goal_random_indices = np.load(FILEPATH+"Config/RandomIndices.npy")
        print("R2: Pre-generated random locations / indices are loaded successfully!")

    def load_polygon_border_obstacle(self):
        file_polygon_border = FILEPATH + "Config/polygon_border.csv"
        file_polygon_obstacle = FILEPATH + "Config/polygon_obstacle.csv"
        self.polygon_border = pd.read_csv(file_polygon_border).to_numpy()
        self.polygon_obstacle = pd.read_csv(file_polygon_obstacle).to_numpy()
        self.polygon_border_shapely = Polygon(self.polygon_border)
        self.polygon_obstacle_shapely = Polygon(self.polygon_obstacle)
        self.line_border_shapely = LineString(self.polygon_border)
        self.line_obstacle_shapely = LineString(self.polygon_obstacle)
        print("R3: Polygon border / obstacle are loaded successfully!")

    def search_path_from_trees(self, cost_valley, polygon_budget_ellipse, line_budget_ellipse, x_current, y_current):
        t1 = time.time()
        self.counter_generations += 1 # increase by 1 everytime it is called
        self.cost_valley = cost_valley
        self.polygon_budget_ellipse = polygon_budget_ellipse
        self.line_budget_ellipse = line_budget_ellipse
        self.x_current = x_current
        self.y_current = y_current

        ind_min = np.argmin(cost_valley)
        x_target, y_target = self.grf_grid[ind_min, :]

        self.tree_table = np.zeros([MAX_ITERATION, 4])
        ind_selected = np.arange(self.counter_generations * MAX_ITERATION, (self.counter_generations+1)*MAX_ITERATION)
        self.tree_table[:, :2] = self.random_locations[ind_selected, :]
        self.tree_table[0, :] = [x_current, y_current, 0, 0]
        self.tree_table[-1, :] = [x_target, y_target, np.inf, FAIL_TO_GENERATE_PATH]
        self.goal_indices = self.goal_random_indices[ind_selected]

        Ntest = MAX_ITERATION-1
        for i in range(1, Ntest):
        # plt.plot(self.tree_table[:Ntest, 1], self.tree_table[:Ntest, 0], 'bx')
        # for i in range(1, MAX_ITERATION-1):
            if self.goal_indices[i] < GOAL_SAMPLE_RATE:
                self.tree_table[i, :] = [x_target, y_target, 0, 0] # refresh table

            # get nearest neighbour
            dx = (self.tree_table[:i, 0] - self.tree_table[i, 0]) ** 2
            dy = (self.tree_table[:i, 1] - self.tree_table[i, 1]) ** 2
            dd = dx + dy
            ind_nearest = np.argmin(dd)
            self.tree_table[i, 3] = ind_nearest

            # steer location to be witin stepsize
            x0 = self.tree_table[ind_nearest, 0]
            y0 = self.tree_table[ind_nearest, 1]
            x1 = self.tree_table[i, 0]
            y1 = self.tree_table[i, 1]

            dx1 = (x1 - x0) ** 2
            dy1 = (y1 - y0) ** 2
            dd1 = np.sqrt(dx1 + dy1)
            if dd1 > STEPSIZE:
                angle = np.math.atan2(x1 - x0, y1 - y0)
                self.tree_table[i, 1] = y1 + STEPSIZE * np.cos(angle)
                self.tree_table[i, 0] = x1 + STEPSIZE * np.sin(angle)
            if self.is_location_legal(self.tree_table[i, 0], self.tree_table[i, 1]):
                self.tree_table[i, 2] = self.get_cost_along_path(x0, y0, x1, y1, self.tree_table[ind_nearest, 2])
            else:
                self.tree_table[i, 2] = np.inf

            # rewire te tree
            dx2 = (self.tree_table[:i, 0] - self.tree_table[i, 0]) ** 2
            dy2 = (self.tree_table[:i, 1] - self.tree_table[i, 1]) ** 2
            dd2 = np.sqrt(dx2 + dy2)
            ind_neighbours = np.where(dd2 <= NEIGHBOUR_RADIUS)[0]

            for ind_neighbour in ind_neighbours:
                ind_new_nearest = int(self.tree_table[i, 3])
                x1 = self.tree_table[ind_new_nearest, 0]
                y1 = self.tree_table[ind_new_nearest, 1]
                x2 = self.tree_table[i, 0]
                y2 = self.tree_table[i, 1]
                cost_path1 = self.get_cost_along_path(x1, y1, x2, y2, self.tree_table[ind_new_nearest, 2])

                x3 = self.tree_table[ind_neighbour, 0]
                y3 = self.tree_table[ind_neighbour, 1]
                x4 = self.tree_table[i, 0]
                y4 = self.tree_table[i, 1]
                cost_path2 = self.get_cost_along_path(x3, y3, x4, y4, self.tree_table[ind_neighbour, 2])

                if cost_path2 < cost_path1:
                    self.tree_table[i, 3] = ind_neighbour
                    self.tree_table[i, 2] = cost_path2

            for ind_neighbour in ind_neighbours:
                x5 = self.tree_table[i, 0]
                y5 = self.tree_table[i, 1]
                x6 = self.tree_table[ind_neighbour, 0]
                y6 = self.tree_table[ind_neighbour, 1]
                cost_path1 = self.get_cost_along_path(x5, y5, x6, y6, self.tree_table[i, 2])

                if cost_path1 < self.tree_table[ind_neighbour, 2]:

                    self.tree_table[ind_neighbour, 3] = i
                    self.tree_table[ind_neighbour, 2] = cost_path1

            dx3 = (self.tree_table[i, 0] - x_target)**2
            dy3 = (self.tree_table[i, 1] - y_target)**2
            dd3 = np.sqrt(dx3 + dy3)
            if dd3 <= HOME_RADIUS:
                self.tree_table[-1, 3] = i

            plt.figure()
            x11 = self.tree_table[i, 0]
            y11 = self.tree_table[i, 1]
            x22 = self.tree_table[int(self.tree_table[i, 3]), 0]
            y22 = self.tree_table[int(self.tree_table[i, 3]), 1]
            plt.plot([y11, y22], [x11, x22], 'g-')
            plt.plot(y_current, x_current, 'bs')
            plt.plot(y_target, x_target, 'r*')
            plt.xlim([np.min(self.polygon_border[:, 1]), np.max(self.polygon_border[:, 1])])
            plt.ylim([np.min(self.polygon_border[:, 0]), np.max(self.polygon_border[:, 0])])
            plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'r-.')
            plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'r-.')
            plt.grid()
            plt.savefig(FILEPATH + "fig/tree/P_{:03d}.jpg".format(i))
            plt.close("all")
            print(i)


        self.path_to_target = []
        self.path_to_target.append([x_target, y_target])
        if self.tree_table[-1, 3] != FAIL_TO_GENERATE_PATH:
            ind_pointer = int(self.tree_table[-1, 3])
            while self.tree_table[ind_pointer, 3] != 0:
                ind = int(self.tree_table[ind_pointer, 3])
                self.path_to_target.append([self.tree_table[ind, 0], self.tree_table[ind, 1]])
                ind_pointer = ind
        else:
            self.path_to_target.append([x_current, y_current])
        self.path_to_target = np.flipud(np.array(self.path_to_target))

        dx4 = (self.path_to_target[1, 0] - x_current)**2
        dy4 = (self.path_to_target[1, 1] - y_current)**2
        dd4 = np.sqrt(dx4 + dy4)

        if dd4 > STEPSIZE:
            angle = np.math.atan2(self.path_to_target[1, 0] - x_current,
                                  self.path_to_target[1, 1] - y_current)
            self.y_next = y_current + STEPSIZE * np.cos(angle)
            self.x_next = x_current + STEPSIZE * np.sin(angle)
        else:
            self.x_next = self.path_to_target[1, 0]
            self.y_next = self.path_to_target[1, 1]

        t2 = time.time()
        print("RRTStarCV takes: ", t2 - t1)
        for i in range(Ntest):
            x1 = self.tree_table[i, 0]
            y1 = self.tree_table[i, 1]
            x2 = self.tree_table[int(self.tree_table[i, 3]), 0]
            y2 = self.tree_table[int(self.tree_table[i, 3]), 1]
            plt.plot([y1, y2], [x1, x2], 'g-')

        plt.plot(self.path_to_target[:, 1], self.path_to_target[:, 0], 'r')
        # print(x_target, y_target)
        # plt.plot(y_target, x_target, 'ks', ms=20)
        from matplotlib.cm import get_cmap
        # plt.plot(self.tree_table[:Ntest, 1], self.tree_table[:Ntest, 0], 'gx', ms=2)
        plt.scatter(self.tree_table[:Ntest, 1], self.tree_table[:Ntest, 0], c=self.tree_table[:Ntest, 2],
                    cmap=get_cmap('RdBu', 2), vmin=0, vmax=5)
        plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'r-.')
        plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'r-.')
        plt.plot(y_target, x_target, 'r*', ms=10)
        plt.colorbar()
        # pass

    def is_location_legal(self, x, y):
        point = Point(x, y)
        islegal = True
        if self.polygon_obstacle_shapely.contains(point) or not self.polygon_budget_ellipse.contains(point):
            islegal = False
        return islegal

    def is_path_legal(self, x1, y1, x2, y2):
        line = LineString([(x1, y1), (x2, y2)])
        islegal = True
        if (self.line_border_shapely.intersects(line) or
                self.line_obstacle_shapely.intersects(line) or
                self.line_budget_ellipse.intersects(line)):
            islegal = False

        return islegal

    def get_cost_along_path(self, x1, y1, x2, y2, cost0):
        if self.is_path_legal(x1, y1, x2, y2):
            cost1 = self.get_cost_from_cost_valley(x1, y1)
            cost2 = self.get_cost_from_cost_valley(x2, y2)
            distance = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
            cost_path = (cost1 + cost2) / 2 * distance
            cost_total = cost_path + distance + cost0
        else:
            cost_total = np.inf
        return cost_total

    def get_cost_from_cost_valley(self, x, y):
        ind = self.get_ind_from_grid(x, y)
        cost = self.cost_valley[ind]
        return cost

    def get_ind_from_grid(self, x, y):
        dx = self.grf_grid[:, 0] - x
        dy = self.grf_grid[:, 1] - y
        dd = dx ** 2 + dy ** 2
        ind = np.argmin(dd)
        return ind

    def check(self):
        import matplotlib.pyplot as plt
        from scipy.spatial.distance import cdist
        from matplotlib.patches import Ellipse
        import math
        from matplotlib.cm import get_cmap
        from GOOGLE.Simulation_2DNidelva.CostValley import CostValley

        self.mu = pd.read_csv(FILEPATH + "Config/data_interpolated.csv")['salinity'].to_numpy()
        DM = cdist(self.grf_grid, self.grf_grid)
        eta = 4.5 / 1600
        Sigma = (1 + eta * DM) * np.exp(-eta * DM)

        cv = CostValley()

        xp = 2000
        yp = -2000
        xn = 1990
        yn = -1900
        cv.budget.budget_left = 4000
        cv.update_cost_valley(self.mu, Sigma, xn, yn, xp, yp)
        cv.get_cost_valley()
        self.search_path_from_trees(cv.cost_valley, cv.budget.polygon_budget_ellipse, cv.budget.line_budget_ellipse, xn, yn)

        # plt.scatter(self.grf_grid[:, 1], self.grf_grid[:, 0], c=self.cost_valley, cmap=get_cmap("BrBG", 10), vmin=0,
        #             vmax=4)
        plt.plot(yn, xn, 'bs', alpha=.3)
        plt.plot(yp, xp, 'ro')
        ellipse = Ellipse(xy=(cv.budget.y_middle, cv.budget.x_middle), width=2 * cv.budget.ellipse_a,
                          height=2 * cv.budget.ellipse_b, angle=math.degrees(cv.budget.angle),
                          edgecolor='r', fc='None', lw=2)
        self.vertices = ellipse.get_verts()

        plt.gca().add_patch(ellipse)
        plt.xlim([np.min(self.polygon_border[:, 1]), np.max(self.polygon_border[:, 1])])
        plt.ylim([np.min(self.polygon_border[:, 0]), np.max(self.polygon_border[:, 0])])
        plt.grid()
        plt.show()

        # plt.plot(self.random_locations[:, 1], self.random_locations[:, 0], 'r.', alpha=.05)
        # plt.plot(self.grf_grid[:, 1], self.grf_grid[:, 0], 'g.')
        # plt.show()

if __name__ == "__main__":
    r = RRTStarCV()
    r.check()

#%%
# x1 = 1000
# y1 = -2000
# x2 = 4000
# y2 = 0
# print(r.is_path_legal(x1, y1, x2, y2))




