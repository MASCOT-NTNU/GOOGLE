"""
This config file contains all constants used for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""

# == GP kernel
SIGMA = .6
LATERAL_RANGE = 1600
NUGGET = .1 # .03
THRESHOLD = 27
# ==

# == RRTStar
MAXITER_EASY = 300
MAXITER_HARD = 600
GOAL_SAMPLE_RATE = .01
STEPSIZE = 400
RADIUS_NEIGHBOUR = 450
DISTANCE_TOLERANCE = 500
# ==

# == Grid
DISTANCE_NEIGHBOUR = 120
DISTANCE_NEIGHBOUR_RADAR = 150
LATITUDE_ORIGIN = 0
LONGITUDE_ORIGIN = 0
# ==

# == Budget
BUDGET_MARGIN = 1000
BUDGET_ELLIPSE_B_MARGIN = 1000
BUDGET_ELLIPSE_B_MARGIN_Tree = 1000

# == Penalty
PENALTY = 10

# == Path planner
BUDGET = 8000
NUM_STEPS = 80
FIGPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/fig/Sim_2DNidelva/"
PATH_FILE = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Simulation_2DNidelva/"
# PATH_BORDER = PATH_FILE + "Config/OpArea.csv"
# PATH_OBSTACLE = PATH_FILE + "Config/Munkholmen.csv"
PATH_BORDER = PATH_FILE + "Config/Polygon_border.csv"
PATH_OBSTACLE = PATH_FILE + "Config/Polygon_obstacle.csv"
PATH_DATA = PATH_FILE + "Field/Data/data_interpolated.csv"
PATH_GRID = PATH_FILE + "Field/Grid/Grid.csv"
# ==

# == PLotting
from matplotlib.cm import get_cmap
CMAP = get_cmap("BrBG", 10)
PATH_REPLICATES = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/fig/Sim_2DNidelva/Replicates/"
# ==


