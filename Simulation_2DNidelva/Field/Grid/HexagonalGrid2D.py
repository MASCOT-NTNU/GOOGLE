"""
This script generates regular hexgonal grid points within certain boundary
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-25
"""

from usr_func import *


class HexgonalGrid2DGenerator:

    def __init__(self, polygon_border=None, polygon_obstacle=None, distance_neighbour=0):
        self.polygon_border = polygon_border
        self.polygon_obstacle = polygon_obstacle
        self.neighbour_distance = distance_neighbour
        self.setup_polygons()
        self.get_bigger_box()
        self.get_grid_within_border()

    def setup_polygons(self):
        self.polygon_border_path = mplPath.Path(self.polygon_border)
        self.polygon_obstacle_path = mplPath.Path(self.polygon_obstacle)

    def get_bigger_box(self):
        self.box_lat_min, self.box_lon_min = map(np.amin, [self.polygon_border[:, 0], self.polygon_border[:, 1]])
        self.box_lat_max, self.box_lon_max = map(np.amax, [self.polygon_border[:, 0], self.polygon_border[:, 1]])

    def get_grid_within_border(self):
        self.get_distance_coverage()
        self.get_lateral_gap()
        self.get_vertical_gap()

        self.grid_x = np.arange(0, self.box_vertical_range, self.vertical_distance)
        self.grid_y = np.arange(0, self.box_lateral_range, self.lateral_distance)
        self.grid_xy = []
        self.grid_wgs = []
        for i in range(len(self.grid_y)):
            for j in range(len(self.grid_x)):
                if isEven(j):
                    x = self.grid_x[j]
                    y = self.grid_y[i] + self.lateral_distance / 2
                else:
                    x = self.grid_x[j]
                    y = self.grid_y[i]
                lat, lon = xy2latlon(x, y, self.box_lat_min, self.box_lon_min)
                if self.is_location_within_border((lat, lon)) and self.is_location_collide_with_obstacle((lat, lon)):
                    self.grid_xy.append([x, y])
                    self.grid_wgs.append([lat, lon])

        self.grid_xy = np.array(self.grid_xy)
        self.grid_wgs = np.array(self.grid_wgs)
        self.coordinates2d = self.grid_wgs

    def get_distance_coverage(self):
        self.box_lateral_range, self.box_vertical_range = latlon2xy(self.box_lat_max, self.box_lon_max,
                                                                    self.box_lat_min, self.box_lon_min)

    def get_lateral_gap(self):
        self.lateral_distance = self.neighbour_distance * np.cos(deg2rad(60)) * 2

    def get_vertical_gap(self):
        self.vertical_distance = self.neighbour_distance * np.sin(deg2rad(60))

    def is_location_within_border(self, location):
        return self.polygon_border_path.contains_point(location)

    def is_location_collide_with_obstacle(self, location):
        return not self.polygon_obstacle_path.contains_point(location)







