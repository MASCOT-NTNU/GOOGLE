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
        self.polygon_border = self.knowledge.polygon_border
        self.polygon_obstacle = self.knowledge.polygon_obstacle
        self.polygon_border_shapely = Polygon(self.polygon_border)
        self.polygon_obstacle_shapely = Polygon(self.polygon_obstacle)
        self.budget = self.knowledge.budget
        self.stepsize = self.knowledge.step_size_lawnmower
        print("End of initialisation! ")

    def get_lawnmower_path(self):
        self.get_bigger_box()
        self.discretise_the_grid()
        self.lawnmower_trajectory = []
        self.lawnmower_trajectory.append([self.starting_location.lat, self.starting_location.lon])
        for j in range(len(self.y)):
            if isEven(j):
                for i in range(len(self.x)):
                    lat_temp, lon_temp = xy2latlon(self.x[i], self.y[j], self.box_lat_min, self.box_lon_min)
                    point = Point(lat_temp, lon_temp)
                    if self.polygon_border_shapely.contains(point) and not self.polygon_obstacle_shapely.contains(point):
                        self.lawnmower_trajectory.append([lat_temp, lon_temp])
            else:
                for i in range(len(self.x)-1, -1, -1):
                    lat_temp, lon_temp = xy2latlon(self.x[i], self.y[j], self.box_lat_min, self.box_lon_min)
                    point = Point(lat_temp, lon_temp)
                    if self.polygon_border_shapely.contains(point) and not self.polygon_obstacle_shapely.contains(point):
                        self.lawnmower_trajectory.append([lat_temp, lon_temp])
        self.lawnmower_trajectory.append([self.ending_location.lat, self.ending_location.lon])

    def get_bigger_box(self):
        self.box_lat_min, self.box_lon_min = map(np.amin, [self.polygon_border[:, 0],
                                                           self.polygon_border[:, 1]])
        self.box_lat_max, self.box_lon_max = map(np.amax, [self.polygon_border[:, 0],
                                                           self.polygon_border[:, 1]])

    def discretise_the_grid(self):
        XRANGE, YRANGE = latlon2xy(self.box_lat_max, self.box_lon_max, self.box_lat_min, self.box_lon_min)
        self.x, self.y = map(np.arange, [0, 0], [XRANGE, YRANGE], [self.stepsize, self.stepsize])

    def get_distance_of_trajectory(self):
        dist = 0
        path = np.array(self.lawnmower_trajectory)
        for i in range(len(path) - 1):
            dist_x, dist_y = latlon2xy(path[i, 0], path[i, 1],
                                       path[i+1, 0], path[i+1, 1])
            dist += np.sqrt(dist_x ** 2 + dist_y ** 2)
        print("Distance of the trajectory: ", dist)
        return dist




