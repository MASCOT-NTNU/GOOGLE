"""
This config file contains all constants used for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""


# == GP kernel
SIGMA = .15
LATERAL_RANGE = .7
NUGGET = .01
THRESHOLD = .6
# ==

# == RRTStar
MAXNUM = 300
GOAL_SAMPLE_RATE = .01
STEP = .15
RADIUS_NEIGHBOUR = .2
DISTANCE_TOLERANCE = .05
# ==

# == Field
XLIM = [0, 1]
YLIM = [0, 1]
NX = 25
NY = 25
# ==

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
OBSTACLES = [[]]
# ==

