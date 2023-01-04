"""
WGS 84 coordinate system.
North-East-Down reference is employed.

It converts (lat, lon) in degrees to (x, y) in meters given a specific origin.
The selected origin is at Nidarosdomen in Trondheim.

Example:
    >>> wgs = WGS()
    >>> x, y = wgs.latlon2xy(63.42690974, 10.3969373)
    >>> print(x, y)
    >>> 0.0, 0.0
    >>> x, y = 1000, 2000
    >>> lat, lon = wgs.xy2latlon(x, y)
    >>> print(lat, lon)
    >>> 63.43589289658141 10.437112521492383
"""

import numpy as np
from math import degrees, radians
from numpy import vectorize


class WGS:
    __CIRCUMFERENCE = 40075000  # [m], circumference
    __LATITUDE_ORIGIN = 63.42690974
    __LONGITUDE_ORIGIN = 10.3969373

    @staticmethod
    @vectorize
    def latlon2xy(lat: float, lon: float) -> tuple:
        x = radians((lat - WGS.__LATITUDE_ORIGIN)) / 2 / np.pi * WGS.__CIRCUMFERENCE
        y = radians((lon - WGS.__LONGITUDE_ORIGIN)) / 2 / np.pi * WGS.__CIRCUMFERENCE * np.cos(radians(lat))
        return x, y

    @staticmethod
    @vectorize
    def xy2latlon(x: float, y: float) -> tuple:
        lat = WGS.__LATITUDE_ORIGIN + degrees(x * np.pi * 2.0 / WGS.__CIRCUMFERENCE)
        lon = WGS.__LONGITUDE_ORIGIN + degrees(y * np.pi * 2.0 / (WGS.__CIRCUMFERENCE * np.cos(radians(lat))))
        return lat, lon

    @staticmethod
    def set_origin(lat: float, lon: float) -> None:
        """ Update the origin for the coordinate system. """
        WGS.__LATITUDE_ORIGIN = lat
        WGS.__LONGITUDE_ORIGIN = lon

    @staticmethod
    def get_origin() -> tuple:
        """ Return origin lat, lon in degrees. """
        return WGS.__LATITUDE_ORIGIN, WGS.__LONGITUDE_ORIGIN

    @staticmethod
    def get_circumference() -> float:
        """ Return the circumference for the earth in meters. """
        return WGS.__CIRCUMFERENCE


if __name__ == "__main__":
    wgs = WGS()
    x, y = wgs.latlon2xy(63.42690974, 10.3969373)
    print(x, y)
    x, y = 1000, 2000
    lat, lon = wgs.xy2latlon(x, y)
    print(lat, lon)



