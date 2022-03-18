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
                 distance_neighbour_radar=None, distance_tolerance=None, budget=None, kernel=None):
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
        self.distance_neighbour_radar = distance_neighbour_radar
        self.distance_tolerance = distance_tolerance

        self.kernel = kernel
        self.budget = budget

        # == get values from kernel
        self.coordinates = self.kernel.coordinates

        self.excursion_prob = None
        self.excursion_set = None
        self.ind_prev = 0
        self.ind_now = 0

        # learned
        self.ind_cand = [] # save all potential candidate locations
        self.ind_cand_filtered = [] # save filtered candidate locations, [#1-No-PopUp-Dive, #2-No-Sharp-Turn]
        self.ind_next = []
        self.ind_visited = []
        self.trajectory = []
        self.step_no = 0

        # criteria
        self.integratedBernoulliVariance = []
        self.rootMeanSquaredError = []
        self.expectedVariance = []
        self.distance_travelled = [0]




