"""WGS coordinate system conversion module.

This module provides functions to convert latitude and longitude coordinates to and from x and y coordinates in the WGS coordinate system.

Attributes:
    __CIRCUMFERENCE (float): The circumference of the Earth in meters.
    __LATITUDE_ORIGIN (float): The origin latitude in degrees.
    __LONGITUDE_ORIGIN (float): The origin longitude in degrees.

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
        """
        Convert latitude and longitude coordinates to a two-dimensional Cartesian coordinate system.

        Args:
            lat (float): Latitude in decimal degrees.
            lon (float): Longitude in decimal degrees.

        Returns:
            tuple: A tuple of two floats representing x and y coordinates in meters.

        """
        x = radians((lat - WGS.__LATITUDE_ORIGIN)) / 2 / np.pi * WGS.__CIRCUMFERENCE
        y = radians((lon - WGS.__LONGITUDE_ORIGIN)) / 2 / np.pi * WGS.__CIRCUMFERENCE * np.cos(radians(lat))
        return x, y

    @staticmethod
    @vectorize
    def xy2latlon(x: float, y: float) -> tuple:
        """
        Convert a two-dimensional Cartesian coordinate system to latitude and longitude coordinates.

        Args:
            x (float): X-coordinate in meters.
            y (float): Y-coordinate in meters.

        Returns:
            tuple: A tuple of two floats representing latitude and longitude in decimal degrees.

        """
        lat = WGS.__LATITUDE_ORIGIN + degrees(x * np.pi * 2.0 / WGS.__CIRCUMFERENCE)
        lon = WGS.__LONGITUDE_ORIGIN + degrees(y * np.pi * 2.0 / (WGS.__CIRCUMFERENCE * np.cos(radians(lat))))
        return lat, lon

    @staticmethod
    def set_origin(lat: float, lon: float) -> None:
        """
        Set the origin for the coordinate system.

        Args:
            lat (float): Latitude in decimal degrees.
            lon (float): Longitude in decimal degrees.

        Returns:
            None

        """
        WGS.__LATITUDE_ORIGIN = lat
        WGS.__LONGITUDE_ORIGIN = lon

    @staticmethod
    def get_origin() -> tuple:
        """
        Get the origin of the coordinate system.

        Returns:
            tuple: A tuple of two floats representing the latitude and longitude of the origin in decimal degrees.

        """
        return WGS.__LATITUDE_ORIGIN, WGS.__LONGITUDE_ORIGIN

    @staticmethod
    def get_circumference() -> float:
        """
        Get the circumference of the earth.

        Returns:
            float: The circumference of the earth in meters.

        """
        return WGS.__CIRCUMFERENCE


if __name__ == "__main__":
    wgs = WGS()
    x, y = wgs.latlon2xy(63.42690974, 10.3969373)
    print(x, y)
    x, y = 1000, 2000
    lat, lon = wgs.xy2latlon(x, y)
    print(lat, lon)


