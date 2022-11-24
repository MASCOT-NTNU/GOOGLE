"""
Config has the most important parameter setting in the long horizon operation in MASCOT Nidelva mission 2022.
- polygon_operational_area: the polygon used to define the safe operational area.
- polygon_operational_area_shapely: shapely object to detect collision or border.
- polygon_obstalce: polygons used for identifying collisions.
- polygon_obstalce_shapely: shapely object to detect collision with obstacles.

- starting location: (lat, lon) used to define the starting location for the long horizon operation.
- home location: (lat, lon) used to define the end location for the home in the long horizon operation.
"""
from WGS import WGS
import numpy as np
import pandas as pd
from shapely.geometry import Polygon


class Config:
    """ Config contains essential setup for the simulation or experiment study. """

    """ Operational Area. """
    __polygon_operational_area = pd.read_csv("csv/polygon_border.csv").to_numpy()
    __polygon_operational_area_shapely = Polygon(__polygon_operational_area)

    """ Obstacles"""
    __polygon_obstacle = pd.read_csv("csv/polygon_obstacle.csv").to_numpy()
    __polygon_obstacle_shapely = Polygon(__polygon_obstacle)

    """ Starting and end locations. """
    # c4: start at home
    __lat_start = 63.45582
    __lon_start = 10.43287

    __lat_home = 63.440323
    __lon_home = 10.355410

    @staticmethod
    def set_polygon_operational_area(value: np.ndarray) -> None:
        """ Set operational area using polygon defined by lat lon coordinates.
        Example:
             value: np.ndarray([[lat1, lon1],
                                [lat2, lon2],
                                ...
                                [latn, lonn]])
        """
        Config.__polygon_operational_area = value
        Config.__polygon_operational_area_shapely = Polygon(Config.__polygon_operational_area)

    @staticmethod
    def set_loc_start(loc: np.ndarray) -> None:
        """ Set the starting location with (lat,lon). """
        Config.__lat_start, Config.__lon_start = loc

    @staticmethod
    def set_loc_home(loc: np.ndarray) -> None:
        """ Set the home location with (lat, lon). """
        Config.__lat_home, Config.__lon_home = loc

    @staticmethod
    def get_polygon_operational_area() -> np.ndarray:
        """ Return polygon for the oprational area. """
        return Config.__polygon_operational_area

    @staticmethod
    def get_polygon_operational_area_shapely() -> 'Polygon':
        """ Return shapelized polygon for the operational area. """
        return Config.__polygon_operational_area_shapely

    @staticmethod
    def get_loc_start() -> np.ndarray:
        """ Return starting location in (x, y). """
        x, y = WGS.latlon2xy(Config.__lat_start, Config.__lon_start)
        return np.array([x, y])

    @staticmethod
    def get_loc_home() -> np.ndarray:
        """ Return home location in (x, y). """
        x, y = WGS.latlon2xy(Config.__lat_home, Config.__lon_home)
        return np.array([x, y])


if __name__ == "__main__":
    s = Config()






