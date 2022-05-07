"""
This config file contains all constants used for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""
import os
from usr_func import latlon2xy


# == sys
FILEPATH = os.getcwd() + "/"
# ==

# == WGS
LATITUDE_ORIGIN = 63.4269097
LONGITUDE_ORIGIN = 10.3969375
# ==

# == GP kernel
NUGGET = .1  # .03
# ==

# == Path planner
BUDGET = 8000 # [m]
LATITUDE_HOME = 63.440618
LONGITUDE_HOME = 10.355851
X_HOME, Y_HOME = latlon2xy(LATITUDE_HOME, LONGITUDE_HOME, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
# ==

# == GRF
GRF_DISTANCE_NEIGHBOUR = 120
DEPTH_LAYER = .5
# ==





