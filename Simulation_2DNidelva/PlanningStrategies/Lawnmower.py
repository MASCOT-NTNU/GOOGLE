"""
This script generates the lawnmower pattern
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-23
"""
from usr_func import *


class LawnMowerPlanning:

    def __init__(self, starting_location=None, ending_location=None, polygon_border=None, polygon_obstacle=None,
                 budget=None, width=None, stepsize=None):
        self.starting_location = starting_location
        self.ending_location = ending_location
        self.polygon_border = polygon_border
        self.polygon_obstacle = polygon_obstacle
        self.polygon_border_shapely = Polygon(self.polygon_border)
        self.polygon_obstacle_shapely = Polygon(self.polygon_obstacle)
        self.budget = budget
        self.width = width
        self.stepsize = stepsize
        print("End of initialisation! ")

    def get_lawnmower_path(self):
        self.get_bigger_box()
        self.discretise_the_grid()
        self.lawn_mower_path_2d = []
        for j in range(len(self.y)):
            if isEven(j):
                for i in range(len(self.x)):
                    lat_temp, lon_temp = xy2latlon(self.x[i], self.y[j], self.box_lat_min, self.box_lon_min)
                    point = Point(lat_temp, lon_temp)
                    if self.polygon_border_shapely.contains(point):
                        self.lawn_mower_path_2d.append([lat_temp, lon_temp])
            else:
                for i in range(len(self.x)-1, -1, -1):
                    lat_temp, lon_temp = xy2latlon(self.x[i], self.y[j], self.box_lat_min, self.box_lon_min)
                    point = Point(lat_temp, lon_temp)
                    if self.polygon_border_shapely.contains(point):
                        self.lawn_mower_path_2d.append([lat_temp, lon_temp])
        self.lawn_mower_path_2d = self.lawn_mower_path_2d[::-1] # change the starting location.

        pass

    def get_bigger_box(self):
        self.box_lat_min, self.box_lon_min = map(np.amin, [self.polygon_border[:, 0],
                                                           self.polygon_border[:, 1]])
        self.box_lat_max, self.box_lon_max = map(np.amax, [self.polygon_border[:, 0],
                                                           self.polygon_border[:, 1]])

    def discretise_the_grid(self):
        XRANGE, YRANGE = latlon2xy(self.box_lat_max, self.box_lon_max, self.box_lat_min, self.box_lon_min)
        self.x, self.y = map(np.arange, [0, 0], [XRANGE, YRANGE], [self.stepsize, self.stepsize])


    def build_3d_lawn_mower(self):
        self.lawn_mower_path_3d = []
        self.build_2d_lawn_mower()
        self.get_unique_depth_layer()
        self.get_yoyo_depth_waypoint()
        quotient = int(np.ceil(len(self.lawn_mower_path_2d) / len(self.depth_yoyo)))
        self.depth_yoyo_path_waypoint = np.tile(self.depth_yoyo, quotient)
        self.depth_yoyo_path_waypoint = self.depth_yoyo_path_waypoint[:len(self.lawn_mower_path_2d)]
        self.lawn_mower_path_3d = np.hstack((self.lawn_mower_path_2d, self.depth_yoyo_path_waypoint.reshape(-1, 1)))

    def build_2d_lawn_mower(self):
        self.lawn_mower_path_2d = []
        self.get_polygon_path()

        for j in range(len(self.y)):
            if isEven(j):
                for i in range(len(self.x)):
                    lat_temp, lon_temp = xy2latlon(self.x[i], self.y[j], self.box_lat_min, self.box_lon_min)
                    if self.polygon_path.contains_point((lat_temp, lon_temp)):
                        self.lawn_mower_path_2d.append([lat_temp, lon_temp])
            else:
                for i in range(len(self.x)-1, -1, -1):
                    lat_temp, lon_temp = xy2latlon(self.x[i], self.y[j], self.box_lat_min, self.box_lon_min)
                    if self.polygon_path.contains_point((lat_temp, lon_temp)):
                        self.lawn_mower_path_2d.append([lat_temp, lon_temp])
        self.lawn_mower_path_2d = self.lawn_mower_path_2d[::-1] # change the starting location.

    def get_polygon_path(self):
        self.polygon_path = mplPath.Path(self.knowledge.polygon)






