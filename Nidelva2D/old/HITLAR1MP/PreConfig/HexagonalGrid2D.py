"""
This script generates regular hexgonal grid points within certain boundary
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-25
"""

from usr_func import *
from GOOGLE.Nidelva2D.Config.Config import LATITUDE_ORIGIN, LONGITUDE_ORIGIN


class HexgonalGrid2DGenerator:

    def __init__(self, polygon_border=None, polygon_obstacle=None, distance_neighbour=0):
        self.polygon_border = polygon_border
        self.polygon_obstacle = polygon_obstacle
        self.neighbour_distance = distance_neighbour
        self.setup_polygons()
        self.get_bigger_box()
        self.get_grid_within_border()

    def setup_polygons(self):
        self.polygon_border_shapely = Polygon(self.polygon_border)
        self.polygon_obstacle_shapely = Polygon(self.polygon_obstacle)

    def get_bigger_box(self):
        self.box_x_min, self.box_y_min = map(np.amin, [self.polygon_border[:, 0], self.polygon_border[:, 1]])
        self.box_x_max, self.box_y_max = map(np.amax, [self.polygon_border[:, 0], self.polygon_border[:, 1]])

    def get_grid_within_border(self):
        self.get_step_size()
        self.grid_x = np.arange(self.box_x_min, self.box_x_max, self.stepsize_x)
        self.grid_y = np.arange(self.box_y_min, self.box_y_max, self.stepsize_y)
        self.grid_xy = []
        self.grid_wgs = []
        for i in range(len(self.grid_y)):
            for j in range(len(self.grid_x)):
                if isEven(j):
                    x = self.grid_x[j]
                    y = self.grid_y[i] + self.stepsize_y / 2
                else:
                    x = self.grid_x[j]
                    y = self.grid_y[i]
                lat, lon = xy2latlon(x, y, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
                point = Point(x, y)
                if self.is_location_within_border(point) and self.is_location_collide_with_obstacle(point):
                    self.grid_xy.append([x, y])
                    self.grid_wgs.append([lat, lon])

        self.grid_xy = np.array(self.grid_xy)
        self.grid_wgs = np.array(self.grid_wgs)

    def get_step_size(self):
        self.stepsize_y = self.neighbour_distance * np.cos(deg2rad(60)) * 2
        self.stepsize_x = self.neighbour_distance * np.sin(deg2rad(60))

    def is_location_within_border(self, location):
        return self.polygon_border_shapely.contains(location)

    def is_location_collide_with_obstacle(self, location):
        return not self.polygon_obstacle_shapely.contains(location)







