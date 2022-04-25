"""
This script only contains the location
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""

import numpy as np
from usr_func import latlon2xy, xy2latlon
from GOOGLE.Simulation_2DNidelva.Config.Config import LATITUDE_ORIGIN, LONGITUDE_ORIGIN


class LocationWGS:

    def __init__(self, lat=None, lon=None):
        self.lat = lat
        self.lon = lon


class LocationXY:

    def __init__(self, x=None, y=None):
        self.x = x
        self.y = y


def WGS2XY(loc_wgs):
    lat = loc_wgs.lat
    lon = loc_wgs.lon
    x, y = latlon2xy(lat, lon, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
    return LocationXY(x, y)


def XY2WGS(loc_xy):
    x = loc_xy.X_START
    y = loc_xy.Y_START
    lat, lon = xy2latlon(x, y, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
    return LocationWGS(lat, lon)


def get_distance_between_wgs_locations(loc1, loc2):
    dist_x, dist_y = latlon2xy(loc1.lat, loc1.lon, loc2.lat, loc2.lon)
    return np.sqrt(dist_x ** 2 + dist_y ** 2)


def get_distance_between_xy_locations(loc1, loc2):
    dist_x = loc1.X_START - loc2.X_START
    dist_y = loc1.Y_START - loc2.Y_START
    return np.sqrt(dist_x ** 2 + dist_y ** 2)




