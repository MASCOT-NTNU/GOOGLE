"""
Config has the most important parameter setting in the long horizon simulation study.
- polygon_operational_area: the polygon used to define the safe operational area.
- polygon_operational_area_shapely: shapely object to detect collision or border.
- polygon_obstalce: polygons used for identifying collisions.
- polygon_obstalce_shapely: shapely object to detect collision with obstacles.

- starting location: (lat, lon) used to define the starting location for the long horizon operation.
"""
from WGS import WGS
import numpy as np
import pandas as pd
import os
from shapely.geometry import Polygon, LineString


class Config:
    """ Config contains essential setup for the simulation study. """
    def __init__(self) -> None:
        """ Initializes the crucial parameters used later in the simulation. """

        """ Set up WGS polygons and starting and end locations. """
        self.__wgs_polygon_border = pd.read_csv(os.getcwd() + "/csv/polygon_border.csv").to_numpy()
        self.__wgs_polygon_obstacle = pd.read_csv(os.getcwd() + "/csv/polygon_obstacle.csv").to_numpy()
        self.__wgs_loc_start = np.array([63.45582, 10.43287])

        """ Convert them to cartesian polygons and starting and end locations. """
        self.__polygon_border = self.wgs2xy(self.__wgs_polygon_border)
        self.__polygon_border_shapely = Polygon(self.__polygon_border)
        self.__line_border_shapely = LineString(self.__polygon_border)

        self.__polygon_obstacle = self.wgs2xy(self.__wgs_polygon_obstacle)
        self.__polygon_obstacle_shapely = Polygon(self.__polygon_obstacle)
        self.__line_obstacle_shapely = LineString(self.__polygon_obstacle)

        x, y = WGS.latlon2xy(self.__wgs_loc_start[0], self.__wgs_loc_start[1])
        self.__loc_start = np.array([x, y])

        """ Default simulation parameter seteup. """
        self.__num_steps = 80  # number of steps.
        self.__num_replicates = 100  # number of replicates
        self.__num_cores = 40  # number of cores to use

    @staticmethod
    def wgs2xy(value: np.ndarray) -> np.ndarray:
        """ Convert polygon containing wgs coordinates to polygon containing xy coordinates. """
        x, y = WGS.latlon2xy(value[:, 0], value[:, 1])
        return np.stack((x, y), axis=1)

    def set_polygon_border(self, value: np.ndarray) -> None:
        """ Set operational area using polygon defined by lat lon coordinates.
        Example:
             value: np.ndarray([[lat1, lon1],
                                [lat2, lon2],
                                ...
                                [latn, lonn]])
        """
        self.__wgs_polygon_border = value
        self.__polygon_border = self.wgs2xy(self.__wgs_polygon_border)
        self.__polygon_border_shapely = Polygon(self.__polygon_border)
        self.__line_border_shapely = LineString(self.__polygon_border)

    def set_polygon_obstacle(self, value: np.ndarray) -> None:
        """ Set polygon obstacle using polygon defined by lat lon coordinates.
        Example:
             value: np.ndarray([[lat1, lon1],
                                [lat2, lon2],
                                ...
                                [latn, lonn]])
        """
        self.__wgs_polygon_obstacle = value
        self.__polygon_obstacle = self.wgs2xy(self.__wgs_polygon_obstacle)
        self.__polygon_obstacle_shapely = Polygon(self.__polygon_obstacle)
        self.__line_obstacle_shapely = LineString(self.__polygon_obstacle)

    def set_loc_start(self, loc: np.ndarray) -> None:
        """ Set the starting location with (lat,lon). """
        self.__wgs_loc_start = loc
        x, y = WGS.latlon2xy(self.__wgs_loc_start[0], self.__wgs_loc_start[1])
        self.__loc_start = np.array([x, y])

    def set_num_steps(self, value: int) -> None:
        """ Set the number of steps in the simulation to be an integer value. """
        self.__num_steps = value

    def set_num_replicates(self, value: int) -> None:
        """ Set the number of replicates in the simulation study. """
        self.__num_replicates = value

    def set_num_cores(self, value: int) -> None:
        """ Set the number of cores to use in the simulation study. """
        self.__num_cores = value

    def get_polygon_border(self) -> np.ndarray:
        """ Return polygon for opa in x y coordinates. """
        return self.__polygon_border

    def get_polygon_border_shapely(self) -> 'Polygon':
        """ Return shapelized polygon for opa in xy coordinates. """
        return self.__polygon_border_shapely

    def get_line_border_shapely(self) -> 'LineString':
        """ Return linestring of polygon border. """
        return self.__line_border_shapely

    def get_polygon_obstacle(self) -> np.ndarray:
        """ Return polygon for the obstacle. """
        return self.__polygon_obstacle

    def get_polygon_obstacle_shapely(self) -> 'Polygon':
        """ Return shapelized polygon for the obstacle. """
        return self.__polygon_obstacle_shapely

    def get_line_obstacle_shapely(self) -> 'LineString':
        """ Return linestring of polygon obstacle. """
        return self.__line_obstacle_shapely

    def get_loc_start(self) -> np.ndarray:
        """ Return starting location in (x, y). """
        return self.__loc_start

    def get_num_steps(self) -> int:
        """ Return the number of steps in the simulation study. """
        return self.__num_steps

    def get_num_replicates(self) -> int:
        """ Return the number of replicates in the simulation study. """
        return self.__num_replicates

    def get_num_cores(self) -> int:
        """ Return the number of cores in the simulation study. """
        return self.__num_cores

    def get_wgs_polygon_border(self) -> np.ndarray:
        """ Return polygon for the oprational area in wgs coordinates. """
        return self.__wgs_polygon_border

    def get_wgs_polygon_obstacle(self) -> np.ndarray:
        """ Return polygon for the oprational area in wgs coordinates. """
        return self.__wgs_polygon_obstacle

    def get_wgs_loc_start(self) -> np.ndarray:
        """ Return starting location in (lat, lon). """
        return self.__wgs_loc_start


if __name__ == "__main__":
    s = Config()






