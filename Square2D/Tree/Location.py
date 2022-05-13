"""
This script only contains the location
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""

import numpy as np


class Location:

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y


def get_distance_between_locations(loc1, loc2):
    return np.sqrt((loc1.X_START - loc2.X_START) ** 2 + (loc1.Y_START - loc2.Y_START) ** 2)

