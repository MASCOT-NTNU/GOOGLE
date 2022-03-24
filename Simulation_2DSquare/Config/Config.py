"""
This config file contains all constants used for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""

# == random seed
import os

# == GP kernel
SIGMA = .1
LATERAL_RANGE = .7
NUGGET = .03
THRESHOLD = .7
# ==

# == RRTStar
MAXITER_EASY = 600
MAXITER_HARD = 600
GOAL_SAMPLE_RATE = .01
STEPSIZE = .1
DISTANCE_NEIGHBOUR_RADAR = .12
DISTANCE_TOLERANCE = .05
# ==

# == Field
XLIM = [0, 1]
YLIM = [0, 1]
NX = 25
NY = 25
DISTANCE_NEIGHBOUR = .05
# ==

# == Budget
BUDGET_MARGIN = .2
BUDGET_ELLIPSE_B_MARGIN = .1
BUDGET_ELLIPSE_B_MARGIN_Tree = .5

# == Penalty
PENALTY = 10

# == Obstacles
# OBSTACLES = [[[.1, .1], [.2, .1], [.2, .2], [.1, .2]],
#              [[.4, .4], [.6, .5], [.5, .6], [.3, .4]],
#              [[.8, .8], [.95, .8], [.95, .95], [.8, .95]]]
# OBSTACLES = [[[.1, .0], [.2, .0], [.2, .5], [.1, .5]],
#              [[.0, .6], [.6, .6], [.6, 1.], [.0, 1.]],
#              [[.8, .0], [1., .0], [1., .9], [.8, .9]],
#              [[.3, .1], [.4, .1], [.4, .6], [.3, .6]],
#              [[.5, .0], [.6, .0], [.6, .4], [.5, .4]]]
# OBSTACLES = [[[1.2, 1.2], [1.4, 1.2], [1.4, 1.4], [1.2, 1.4]]]
OBSTACLES = [[[.4, .4], [.6, .5], [.5, .6], [.3, .4]]]
BORDER = [[.0, .0], [1., .0], [1., 1.], [.0, 1.]]
# OBSTACLES = [[]]
# ==

# == Path planner
BUDGET = 5
NUM_STEPS = 80
STEPSIZE_LAWNMOWER = .2
# FIGPATH = os.getcwd() + "/GOOGLE/fig/Sim_Square/rrt_star/"
# ==

# == Directories
FILEPATH = os.getcwd() + "/GOOGLE/Simulation_2DSquare/"
FIGPATH = os.getcwd() + "/GOOGLE/fig/"
PATH_REPLICATES = FIGPATH + "Sim_2DSquare/replicates/"
# ==

# == Plotting
from matplotlib.cm import get_cmap
CMAP = get_cmap("BrBG", 10)
# ==



