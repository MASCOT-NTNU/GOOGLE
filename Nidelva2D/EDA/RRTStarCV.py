"""
This script builds the RRTStarCV
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-26
"""

from GOOGLE.Nidelva2D.Config.Config import FILEPATH, X_HOME, Y_HOME
from usr_func import Polygon, Point, LineString
import pandas as pd
import numpy as np
import time

# == Setup
GOAL_SAMPLE_RATE = .01
MAX_ITERATION = 1000
STEPSIZE = 120
NEIGHBOUR_RADIUS = 150
TARGET_RADIUS = 100
# ==


class TreeNode:

    def __init__(self, x, y, cost=0, parent=None):
        self.x = x
        self.y = y
        self.cost = cost
        self.parent = parent


class RRTStarCV:

    def __init__(self):
        self.load_grf_grid()
        self.load_random_locations()
        self.load_polygon_border_obstacle()

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
        N_random_locations = self.random_locations.shape[0]
        self.cost_valley = cost_valley
        self.polygon_budget_ellipse = polygon_budget_ellipse
        self.line_budget_ellipse = line_budget_ellipse
        self.x_current = x_current
        self.y_current = y_current

        ind_min = np.argmin(cost_valley)
        x_target, y_target = self.grf_grid[ind_min, :]

        start_node = TreeNode(x_current, y_current, 0, None)
        target_node = TreeNode(x_target, y_target, 0, None)

        ind_selected = np.random.randint(0, N_random_locations, MAX_ITERATION)
        x_random = self.random_locations[ind_selected, 0]
        y_random = self.random_locations[ind_selected, 1]
        goal_indices = self.goal_random_indices[ind_selected]

        self.tree_nodes = []
        self.tree_nodes.append(start_node)

        for i in range(MAX_ITERATION):
            # print(i)
            # get random location
            if goal_indices[i] <= GOAL_SAMPLE_RATE:
                x_new = x_target
                y_new = y_target
            else:
                x_new = x_random[i]
                y_new = y_random[i]

            # find nearest node from tree
            nearest_node, dist_nearest = self.get_nearest_node(x_new, y_new)
            # print("s1: found nearest node")

            # steer random location
            if dist_nearest > STEPSIZE:
                angle = np.math.atan2(x_new - nearest_node.x, y_new - nearest_node.y)
                y_new = nearest_node.y + STEPSIZE * np.cos(angle)
                x_new = nearest_node.x + STEPSIZE * np.sin(angle)
            # print("s2: finished steering")

            if not self.is_location_legal(x_new, y_new):
                continue

            # rewire tree
            new_node = TreeNode(x_new, y_new, 0, nearest_node)
            ind_neighbours = self.get_neighbour_node_ind()
            for ind_neighbour in ind_neighbours:
                neighbour_node = self.tree_nodes[ind_neighbour]
                cost1 = self.get_cost_along_path(nearest_node.x, nearest_node.y, new_node.x,
                                                 new_node.y, nearest_node.cost)
                cost2 = self.get_cost_along_path(neighbour_node.x, neighbour_node.y, new_node.x,
                                                 new_node.y, neighbour_node.cost)
                if cost2 < cost1:
                    nearest_node = neighbour_node
                new_node.cost = cost2
                new_node.parent = nearest_node

            for ind_neighbour in ind_neighbours:
                neighbour_node = self.tree_nodes[ind_neighbour]
                cost3 = self.get_cost_along_path(new_node.x, new_node.y, neighbour_node.x,
                                                 neighbour_node.y, new_node.cost)
                if cost3 < neighbour_node.cost:
                    neighbour_node.cost = cost3
                    neighbour_node.parent = new_node
            # print("s3: finished rewiring")

            if not self.is_path_legal(nearest_node.x, nearest_node.y, new_node.x, new_node.y):
                continue

            # check home criteria
            if np.sqrt((new_node.x - x_target)**2 + (new_node.y - y_target)**2) <= TARGET_RADIUS:
                target_node.parent = new_node
            else:
                self.tree_nodes.append(new_node)
            # print("s4: finished home checking")

        # produce trajectory
        self.path_to_target = []
        self.path_to_target.append([target_node.x, target_node.y])
        pointer_node = target_node
        checker = 0
        while pointer_node.parent is not None:
            checker += 1
            node = pointer_node.parent
            self.path_to_target.append([node.x, node.y])
            pointer_node = node
            if checker > MAX_ITERATION:
                break
        self.path_to_target = np.flipud(np.array(self.path_to_target))
        # print("s5: finished path generation")

        if len(self.path_to_target)>2:
            angle = np.math.atan2(self.path_to_target[1, 0] - x_current,
                                  self.path_to_target[1, 1] - y_current)
            self.y_next = y_current + STEPSIZE * np.cos(angle)
            self.x_next = x_current + STEPSIZE * np.sin(angle)
        else:
            angle = np.math.atan2(x_target - x_current,
                                  y_target - y_current)
            self.y_next = y_current + STEPSIZE * np.cos(angle)
            self.x_next = x_current + STEPSIZE * np.sin(angle)
        # print("finished waypoint generation")
        if not self.is_location_legal(self.x_next, self.y_next) or not self.is_path_legal(x_current, y_current,
                                                                                          self.x_next, self.y_next):
            # get legal location next
            self.x_next, self.y_next = self.get_legal_location(x_current, y_current)

        t2 = time.time()
        print("RRTStarCV takes: ", t2 - t1)
        # for node in self.tree_nodes:
        #     if node.parent is not None:
        #         plt.plot([node.y, node.parent.y],
        #                  [node.x, node.parent.x], "g-")
        #         # plt.plot(node.y, node.x, 'k.', alpha=.5)
        # plt.plot(self.path_to_target[:, 1], self.path_to_target[:, 0], 'r-')
        # from matplotlib.cm import get_cmap
        # plt.scatter(self.grf_grid[:, 1], self.grf_grid[:, 0], c=self.cost_valley, s=50, cmap=get_cmap("BrBG", 10), vmin=0, vmax=2, alpha=.5)
        # plt.colorbar()
        # plt.plot(y_target, x_target, 'g*')
        np.savetxt(FILEPATH + "Waypoint/waypoint.txt", np.array([self.x_next, self.y_next]), delimiter=', ')
        print("waypoint is saved!")
        # return self.x_next, self.y_next

    def get_nearest_node(self, x, y):
        self.distance_from_location_to_nodes = np.zeros(len(self.tree_nodes))
        tempNode = TreeNode(x, y)
        for i in range(len(self.tree_nodes)):
            self.distance_from_location_to_nodes[i] = self.get_distance_between_nodes(tempNode, self.tree_nodes[i])
        ind = np.argmin(self.distance_from_location_to_nodes)
        return self.tree_nodes[ind], self.distance_from_location_to_nodes[ind]

    def get_distance_between_nodes(self, node1, node2):
        dx = node1.x - node2.x
        dy = node1.y - node2.y
        dist = np.sqrt(dx**2 + dy**2)
        return dist

    def get_legal_location(self, x, y):
        angles = np.linspace(0, 2*np.pi, 60)
        for angle in angles:
            y_next = y + STEPSIZE * np.cos(angle)
            x_next = x + STEPSIZE * np.sin(angle)
            if self.is_location_legal(x_next, y_next) and self.is_path_legal(x, y, x_next, y_next):
                return x_next, y_next

    def get_route_home(self, x, y):
        distance = np.sqrt((X_HOME - x)**2 + (Y_HOME - y)**2)
        if distance > STEPSIZE:
            angle = np.math.atan2(X_HOME - x, Y_HOME - y)
            y_next = y + STEPSIZE * np.cos(angle)
            x_next = x + STEPSIZE * np.sin(angle)
        else:
            x_next = X_HOME
            y_next = Y_HOME
        return x_next, y_next

    def get_neighbour_node_ind(self):
        ind = np.where(self.distance_from_location_to_nodes <= NEIGHBOUR_RADIUS)[0]
        return ind

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
            cost_dist = distance
            cost_path = (cost1 + cost2) / 2 * distance
            cost_total = cost_path + cost_dist + cost0
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

        xp = 2900
        yp = -10
        xn = 3000
        yn = 0
        cv.budget.budget_left = 4000
        cv.update_cost_valley(self.mu, Sigma, xn, yn, xp, yp)
        cv.get_cost_valley()
        plt.figure()

        self.search_path_from_trees(cv.cost_valley, cv.budget.polygon_budget_ellipse, cv.budget.line_budget_ellipse, xn, yn)

        # plt.scatter(self.grf_grid[:, 1], self.grf_grid[:, 0], c=self.cost_valley, cmap=get_cmap("BrBG", 10), vmin=0,
        #             vmax=4)

        plt.plot(yn, xn, 'bs', alpha=.3)
        plt.plot(yp, xp, 'ro')
        plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'r-.')
        plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'r-.')
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




