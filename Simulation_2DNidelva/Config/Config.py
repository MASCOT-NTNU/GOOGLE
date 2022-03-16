"""
This config file contains all constants used for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""

# == random seed
# import numpy
# numpy.random.seed(0)

# == GP kernel
import os

import pandas as pd

SIGMA = 4
LATERAL_RANGE = 400
NUGGET = .03
THRESHOLD = 26
# ==

# == RRTStar
MAXITER_EASY = 300
MAXITER_HARD = 600
GOAL_SAMPLE_RATE = .01
STEPSIZE = 500
RADIUS_NEIGHBOUR = 600
DISTANCE_TOLERANCE = 500
# ==

# == Grid
NEIGHBOUR_DISTANCE = 50
LATITUDE_ORIGIN = 0
LONGITUDE_ORIGIN = 0
# ==

# == Budget
BUDGET_MARGIN = 500
BUDGET_ELLIPSE_B_MARGIN = 250
BUDGET_ELLIPSE_B_MARGIN_Tree = 500

# == Penalty
PENALTY = 10

# == Path planner
BUDGET = 6000
NUM_STEPS = 80
FIGPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/fig/Sim_2DNidelva/"
PATH_FILE = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Simulation_2DNidelva/"
PATH_BORDER = PATH_FILE + "Config/Polygon_border.csv"
PATH_OBSTACLE = PATH_FILE + "Config/Polygon_obstacle.csv"
PATH_DATA = PATH_FILE + "Field/Data/data_interpolated.csv"
PATH_GRID = PATH_FILE + "Field/Grid/Grid.csv"
# ==



