"""
This script only contains the knowledge node
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-23
"""
from shapely.geometry import Polygon, LineString


class Knowledge:

    def __init__(self, grid=None, starting_location=None, ending_location=None, goal_location=None,
                 goal_sample_rate=None, polygon_border=None, polygon_obstacles=None, step_size=None,
                 step_size_lawnmower=None, maximum_iteration=1000, distance_neighbour_radar=None,
                 distance_tolerance=None, budget=None, threshold=None):
        self.grid = grid
        self.starting_location = starting_location
        self.ending_location = ending_location
        self.goal_location = goal_location
        self.goal_sample_rate = goal_sample_rate
        self.polygon_border = polygon_border
        self.polygon_obstacles = polygon_obstacles
        self.step_size = step_size
        self.step_size_lawnmower = step_size_lawnmower
        self.maximum_iteration = maximum_iteration
        self.distance_neighbour_radar = distance_neighbour_radar
        self.distance_tolerance = distance_tolerance
        self.budget = budget
        self.threshold = threshold

        # computed
        self.polygon_border_shapely = Polygon(self.polygon_border)
        self.polygon_borderline_shapely = LineString(self.polygon_border)
        self.polygon_obstacles_shapely = []
        for i in range(len(self.polygon_obstacles)):
            self.polygon_obstacles_shapely.append(Polygon(list(map(tuple, self.polygon_obstacles[i]))))
        self.excursion_prob = None
        self.excursion_set = None
        self.ind_prev = 0
        self.ind_now = 0
        self.mu_prior = None
        self.mu_truth = None
        self.mu_cond = None
        self.R = None
        self.Sigma_prior = None
        self.Sigma_cond = None
        self.cost_valley = None
        self.cost_eibv = None
        self.cost_vr = None

        # computed budget ellipse
        self.budget_middle_location = None
        self.budget_ellipse_angle = None
        self.budget_ellipse_a = None
        self.budget_ellipse_b = None
        self.budget_ellipse_c = None

        # learned
        self.ind_cand = [] # save all potential candidate locations
        self.ind_cand_filtered = [] # save filtered candidate locations, [#1-No-PopUp-Dive, #2-No-Sharp-Turn]
        self.ind_next = []
        self.ind_visited = []
        self.trajectory = []
        self.step_no = 0

        # criteria
        self.integrated_bernoulli_variance = []
        self.root_mean_squared_error = []
        self.expected_variance = []
        self.distance_travelled = [0]

        # signal
        self.gohome = False

