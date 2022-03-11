"""
This script only contains the knowledge node
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""

from shapely.geometry import Polygon
import numpy as np


class Knowledge:

    def __init__(self, starting_location=None, ending_location=None, goal_location=None, goal_sample_rate=None,
                 polygon_border=None, polygon_obstacle=None, depth=None, step_size_lateral=None,
                 step_size_vertical=None, maximum_iteration=1000, neighbour_radius=None, distance_tolerance=None,
                 budget=None, kernel=None, mu=None, Sigma=None, F=None, EIBV=None):
        self.starting_location = starting_location
        self.ending_location = ending_location
        self.goal_location = goal_location
        self.goal_sample_rate = goal_sample_rate

        self.polygon_border = polygon_border
        self.polygon_obstacle = polygon_obstacle
        self.polygon_border_path = Polygon(self.polygon_border)
        self.polygon_obstacle_path = Polygon(self.polygon_obstacle)
        self.depth = depth

        self.step_size_lateral = step_size_lateral
        self.step_size_vertical = step_size_vertical
        self.step_size_total = np.sqrt(self.step_size_lateral ** 2 + self.step_size_vertical ** 2)
        self.maximum_iteration = maximum_iteration
        self.neighbour_radius = neighbour_radius
        self.distance_tolerance = distance_tolerance

        self.kernel = kernel
        self.budget = budget

        self.mu = mu
        self.Sigma = Sigma
        self.F = F
        self.EIBV = EIBV


