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
        self.polygon_border_shapely = Polygon(self.polygon_border)
        self.polygon_obstacle_shapely = Polygon(self.polygon_obstacle)
        print("Polygons are setup successfully!")

    def get_bigger_box(self):
        self.box_x_min, self.box_y_min = map(np.amin, [self.polygon_border[:, 0], self.polygon_border[:, 1]])
        self.box_x_max, self.box_y_max = map(np.amax, [self.polygon_border[:, 0], self.polygon_border[:, 1]])
        print("bigger box is set up properly!")

    def get_grid_within_border(self):
        self.get_distance_coverage()
        self.get_lateral_gap()
        self.get_vertical_gap()

        self.grid_x = np.arange(0, self.box_lateral_range, self.lateral_gap_distance)
        self.grid_y = np.arange(0, self.box_vertical_range, self.vertical_gap_distance)
        self.grid_xy = []
        for i in range(len(self.grid_y)):
            for j in range(len(self.grid_x)):
                if isEven(i):
                    x = self.grid_x[j] + self.lateral_gap_distance / 2
                    y = self.grid_y[i]
                else:
                    x = self.grid_x[j]
                    y = self.grid_y[i]
                point = Point(x, y)
                if self.is_location_within_border(point) and not self.is_location_collide_with_obstacle(point):
                    self.grid_xy.append([x, y])
        self.grid_xy = np.array(self.grid_xy)

    def get_distance_coverage(self):
        self.box_lateral_range = self.box_x_max - self.box_x_min
        self.box_vertical_range = self.box_y_max - self.box_y_min
        print("Distance coverage is computed okay")
        print("Distance range - x: ", self.box_lateral_range)
        print("Distance range - y: ", self.box_vertical_range)

    def get_lateral_gap(self):
        self.lateral_gap_distance = self.neighbour_distance * np.cos(deg2rad(60)) * 2
        print("Lateral gap: ", self.lateral_gap_distance)

    def get_vertical_gap(self):
        self.vertical_gap_distance = self.neighbour_distance * np.sin(deg2rad(60))
        print("Vertical gap: ", self.vertical_gap_distance)

    def is_location_within_border(self, location):
        return self.polygon_border_shapely.contains(location)

    def is_location_collide_with_obstacle(self, location):
        return self.polygon_obstacle_shapely.contains(location)







