"""
This script only contains the knowledge node
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""

from shapely.geometry import Polygon, LineString


class Knowledge:

    def __init__(self, starting_location=None, ending_location=None, goal_location=None, goal_sample_rate=None,
                 polygon_border=None, polygon_obstacle=None, step_size=None, maximum_iteration=1000,
                 neighbour_radius=None, distance_tolerance=None, budget=None, kernel=None):
        self.starting_location = starting_location
        self.ending_location = ending_location
        self.goal_location = goal_location
        self.goal_sample_rate = goal_sample_rate

        self.polygon_border = polygon_border
        self.polygon_obstacle = polygon_obstacle
        self.polygon_border_path = Polygon(self.polygon_border)
        self.borderline_path = LineString(self.polygon_border)
        self.polygon_obstacle_path = Polygon(self.polygon_obstacle)

        self.step_size = step_size
        self.maximum_iteration = maximum_iteration
        self.neighbour_radius = neighbour_radius
        self.distance_tolerance = distance_tolerance

        self.kernel = kernel
        self.budget = budget



