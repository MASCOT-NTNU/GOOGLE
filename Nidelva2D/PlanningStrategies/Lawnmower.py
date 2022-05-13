"""
This script generates the lawnmower pattern
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-23
"""
from usr_func import *


class LawnMowerPlanning:

    def __init__(self, knowledge=None):
        self.knowledge = knowledge
        self.starting_location = self.knowledge.starting_location
        self.ending_location = self.knowledge.ending_location
        self.polygon_border_shapely = self.knowledge.polygon_border_shapely
        self.polygon_obstacles_shapely = self.knowledge.polygon_obstacles_shapely
        self.budget = self.knowledge.budget
        self.stepsize = self.knowledge.step_size_lawnmower
        print("End of initialisation! ")

    def get_lawnmower_path(self):
        self.get_bigger_box()
        self.discretise_the_grid()
        self.lawnmower_trajectory = []
        self.lawnmower_trajectory.append([self.starting_location.X_START, self.starting_location.Y_START])
        for j in range(len(self.y)):
            if isEven(j):
                for i in range(len(self.x)):
                    x_temp, y_temp = self.x[i], self.y[j]
                    point = Point(x_temp, y_temp)
                    if self.polygon_border_shapely.contains(point) and not self.is_within_obstacles(point):
                        self.lawnmower_trajectory.append([x_temp, y_temp])
            else:
                for i in range(len(self.x)-1, -1, -1):
                    x_temp, y_temp = self.x[i], self.y[j]
                    point = Point(x_temp, y_temp)
                    if self.polygon_border_shapely.contains(point) and not self.is_within_obstacles(point):
                        self.lawnmower_trajectory.append([x_temp, y_temp])
        self.lawnmower_trajectory.append([self.ending_location.X_START, self.ending_location.Y_START])

    def is_within_obstacles(self, point):
        within = False
        for i in range(len(self.polygon_obstacles_shapely)):
            if self.polygon_obstacles_shapely[i].contains(point):
                within = True
        return within

    def get_bigger_box(self):
        self.box_x_min, self.box_y_min = map(np.amin, [self.knowledge.polygon_border[:, 0],
                                                       self.knowledge.polygon_border[:, 1]])
        self.box_x_max, self.box_y_max = map(np.amax, [self.knowledge.polygon_border[:, 0],
                                                       self.knowledge.polygon_border[:, 1]])

    def discretise_the_grid(self):
        XRANGE = self.box_x_max - self.box_x_min
        YRANGE = self.box_y_max - self.box_y_min
        self.x, self.y = map(np.arange, [0, 0], [XRANGE, YRANGE], [self.stepsize, self.stepsize])

    def get_distance_of_trajectory(self):
        dist = 0
        path = np.array(self.lawnmower_trajectory)
        for i in range(len(path) - 1):
            dist_x = path[i, 0] - path[i+1, 0]
            dist_y = path[i, 1] - path[i+1, 1]
            dist += np.sqrt(dist_x ** 2 + dist_y ** 2)
        print("Distance of the trajectory: ", dist)
        return dist




