"""
This script generates the next waypoint based on pre_scripted lawnmower
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-08
"""

from usr_func import *
import warnings
import matplotlib.path as mplPath


class LawnMowerPlanning:

    def __init__(self, knowledge):
        # warnings.warn("Lawn mower distance between each leg is DISTANCE_LATERAL, it can be modified if needed")
        # warnings.warn("Lawn mower is set to be vertical as default, it can be horizontal as well")
        self.knowledge = knowledge
        # self.build_3d_lawn_mower()

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
        self.build_bigger_rectangular_box()
        self.discretise_the_grid()
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

    def build_bigger_rectangular_box(self):
        self.box_lat_min, self.box_lon_min = map(np.amin, [self.knowledge.polygon[:, 0], self.knowledge.polygon[:, 1]])
        self.box_lat_max, self.box_lon_max = map(np.amax, [self.knowledge.polygon[:, 0], self.knowledge.polygon[:, 1]])

    def discretise_the_grid(self):
        XRANGE, YRANGE = latlon2xy(self.box_lat_max, self.box_lon_max, self.box_lat_min, self.box_lon_min)
        self.x, self.y = map(np.arange, [0, 0], [XRANGE, YRANGE], [self.knowledge.distance_lateral, self.knowledge.distance_lateral])

    def get_unique_depth_layer(self):
        self.depth = np.unique(self.knowledge.coordinates[:, 2])

    def get_yoyo_depth_waypoint(self):
        self.depth_yoyo = np.append(self.depth, np.flip(self.depth))



