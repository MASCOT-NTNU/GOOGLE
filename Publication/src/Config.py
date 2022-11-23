"""
Config handles the most important parameter setting in the long horizon operation in MASCOT Porto mission 2022.
- misssion date: "2022-10-01_2022-10-02"
- wind direction: ["North", "East", "South", "West"]
- wind level: ["Mild", "Moderate", "Heavy"], the corresponding wind speed: [0, 2.5, 6] m/s
- clock_start: 10, when does the operation start: 10 o'clock.
- clock_end: 16, when does the operation end: 16 o'clock.

- polygon_operational_area: the polygon used to define the safe operational area.
- polygon_operational_area_shapely: shapely object to detect collision or border.

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
    __polygon_operational_area = pd.read_csv("OPA_GOOGLE_reduced.csv").to_numpy()
    __polygon_operational_area_shapely = Polygon(__polygon_operational_area)

    """ Obstacles"""
    __polygon_obstacles = [np.array([[.4, .4],
                                     [.6, .5],
                                     [.5, .6],
                                     [.3, .4]])]

    """ Starting and end locations. """
    # c4: start at home
    __lat_start = 41.12677
    __lon_start = -8.68574

    __lat_home = 41.12677
    __lon_home = -8.68574

    """ Create a resume state. """
    __resume = False

    @staticmethod
    def set_mission_date(value: str) -> None:
        """ Set mission date with a format 2022-10-01_2022-10-02. """
        Config.__mission_date = value

    @staticmethod
    def set_wind_direction(value: str) -> None:
        """ Set wind direction to be North, East, South, West. """
        Config.__wind_dir = value

    @staticmethod
    def set_wind_level(value: str) -> None:
        """ Set wind level to be Mild, Moderate, Heavy. """
        Config.__wind_level = value

    @staticmethod
    def set_clock_start(value: int) -> None:
        """ Set starting clock to be 0, 1, 2, 3, ..., 24. """
        Config.__clock_start = value

    @staticmethod
    def set_clock_end(value: int) -> None:
        """ Set starting clock to be 0, 1, 2, 3, ..., 24. Must be larger than MOHID.__clock_start. """
        Config.__clock_end = value

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






