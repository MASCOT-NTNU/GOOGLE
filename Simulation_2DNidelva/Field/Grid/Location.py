"""
This class contains location
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-23
"""


class Location:

    lat = None
    lon = None
    depth = None

    def __init__(self, lat=None, lon=None, depth=None):
        self.lat = lat
        self.lon = lon
        self.depth = depth