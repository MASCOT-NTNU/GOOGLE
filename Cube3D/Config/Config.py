"""
This config file contains all constants used for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-15
"""

# == random seed
import numpy
# numpy.random.seed(0)

# == GP kernel
SIGMA = .1
LATERAL_RANGE = .7
NUGGET = .01
THRESHOLD = .7
# ==

# == RRTStar
MAXITER_EASY = 600
MAXITER_HARD = 1000
GOAL_SAMPLE_RATE = .01
STEPSIZE = .2
RADIUS_NEIGHBOUR = .22
DISTANCE_TOLERANCE = .05
# ==

# == PreConfig
XLIM = [0, 1]
YLIM = [0, 1]
ZLIM = [0, 1]
NX = 25
NY = 25
NZ = 25
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
OBSTACLES = [[[.4, .4, .4], [.6, .5, ], [.5, .6], [.3, .4]]]
# OBSTACLES = [[]]
# ==

# == Path planner
BUDGET = 5
NUM_STEPS = 80
FIGPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/fig/Sim_Square/rrt_star/"
# ==



