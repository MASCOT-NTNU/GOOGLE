""" Config contains essential setup for the simulation study.
"""
from WGS import WGS
import numpy as np
import pandas as pd
import os
from shapely.geometry import Polygon, LineString


class Config:
    """
    Attributes:
        __wgs_polygon_border (np.ndarray): the polygon used to define the border.
        __wgs_polygon_obstacle (np.ndarray): polygons used for identifying collisions.
        __wgs_loc_start (np.ndarray): starting location: (lat, lon) used to define the starting location for the long horizon operation.
        __wgs_loc_end (np.ndarray): end location: (lat, lon) used to define the end location for the long horizon operation.
        
        __polygon_border (np.ndarray): the polygon used to define the border in local Cartesian coordinate system.
        __polygon_border_shapely (shapely.geometry.polygon.Polygon): shapely object to detect collision or border.
        __line_border_shapely (shapely.geometry.linestring.LineString): shapely object to detect collision or border.
        __polygon_obstacle (np.ndarray): polygons used for identifying collisions in local Cartesian coordinate system.
        __polygon_obstacle_shapely (shapely.geometry.polygon.Polygon): shapely object to detect collision with obstacles.
        __line_obstacle_shapely (shapely.geometry.linestring.LineString): shapely object to detect collision with obstacles.
        
        __loc_start (np.ndarray): starting location: (x, y) used to define the starting location for the long horizon operation.
        __loc_end (np.ndarray): end location: (x, y) used to define the end location for the long horizon operation.

        __num_steps (int): number of steps in the simulation.
        __num_replicates (int): number of replicates in the simulation.
        __num_cores (int): number of cores used in the simulation.
    """
    def __init__(self) -> None:
        """ Initializes the crucial parameters used later in the simulation. Set up WGS polygons and starting and end locations.
        """
        self.__num_steps = 100  # number of steps.
        self.__num_replicates = 1  # number of replicates
        self.__num_cores = 1  # number of cores to use

        self.__wgs_polygon_border = pd.read_csv(os.getcwd() + "/csv/polygon_border.csv").to_numpy()
        self.__wgs_polygon_obstacle = pd.read_csv(os.getcwd() + "/csv/polygon_obstacle.csv").to_numpy()
        self.__wgs_loc_start = np.array([63.456232, 10.435198])  # loc used in experiment.
        self.__wgs_loc_end = np.array([63.4425493, 10.3572617])
        # self.__wgs_loc_start = np.array([63.44038447, 10.35675578])  # close to TBS.
        # self.__wgs_loc_start = np.array([63.438611, 10.374487])  # lower west.
        # self.__wgs_loc_start = np.array([63.439921, 10.389458])  # lower middle.
        # self.__wgs_loc_start = np.array([63.44912, 10.35067])  # upper west.
        # self.__wgs_loc_start = np.array([63.46236, 10.41938])  # middle east.
        # self.__wgs_loc_start = np.array([63.46674, 10.39385])  # upper middle above munkholm.
        # self.__wgs_loc_start = np.array([63.439385, 10.356280])  # far west close to margin of boundary.
        
        self.__polygon_border = self.wgs2xy(self.__wgs_polygon_border)
        self.__polygon_border_shapely = Polygon(self.__polygon_border)
        self.__line_border_shapely = LineString(self.__polygon_border)

        self.__polygon_obstacle = self.wgs2xy(self.__wgs_polygon_obstacle)
        self.__polygon_obstacle_shapely = Polygon(self.__polygon_obstacle)
        self.__line_obstacle_shapely = LineString(self.__polygon_obstacle)

        x, y = WGS.latlon2xy(self.__wgs_loc_start[0], self.__wgs_loc_start[1])
        self.__loc_start = np.array([x, y])
        x, y = WGS.latlon2xy(self.__wgs_loc_end[0], self.__wgs_loc_end[1])
        self.__loc_end = np.array([x, y])

    @staticmethod
    def wgs2xy(value: np.ndarray) -> np.ndarray:
        """ 
        Convert polygon containing wgs coordinates to polygon containing xy coordinates. 

        Args:
            value (np.ndarray): polygon containing wgs coordinates.
        
        Returns:
            np.ndarray: polygon containing xy coordinates.
        
        """
        x, y = WGS.latlon2xy(value[:, 0], value[:, 1])
        return np.stack((x, y), axis=1)

    def set_polygon_border(self, value: np.ndarray) -> None:
        """ 
        Set operational area using polygon defined by lat lon coordinates.

        Args:
            value (np.ndarray): polygon defined by lat lon coordinates.

            Example:
                value: np.ndarray([[lat1, lon1],
                                    [lat2, lon2],
                                    ...
                                    [latn, lonn]])

        Returns:
            None    

        """
        self.__wgs_polygon_border = value
        self.__polygon_border = self.wgs2xy(self.__wgs_polygon_border)
        self.__polygon_border_shapely = Polygon(self.__polygon_border)
        self.__line_border_shapely = LineString(self.__polygon_border)

    def set_polygon_obstacle(self, value: np.ndarray) -> None:
        """ 
        Set polygon obstacle using polygon defined by lat lon coordinates.

        Args:
            value (np.ndarray): polygon defined by lat lon coordinates.

            Example:
                value: np.ndarray([[lat1, lon1],
                                    [lat2, lon2],
                                    ...
                                    [latn, lonn]])

        Returns:    
            None    
        """
        self.__wgs_polygon_obstacle = value
        self.__polygon_obstacle = self.wgs2xy(self.__wgs_polygon_obstacle)
        self.__polygon_obstacle_shapely = Polygon(self.__polygon_obstacle)
        self.__line_obstacle_shapely = LineString(self.__polygon_obstacle)

    def set_loc_start(self, loc: np.ndarray) -> None:
        """ 
        Set the starting location with (lat,lon).

        Args:
            loc (np.ndarray): starting location with (lat,lon).

        Returns:
            None

        """
        self.__wgs_loc_start = loc
        x, y = WGS.latlon2xy(self.__wgs_loc_start[0], self.__wgs_loc_start[1])
        self.__loc_start = np.array([x, y])

    def set_loc_end(self, loc: np.ndarray) -> None:
        """ 
        Set the starting location with (lat,lon).

        Args:
            loc (np.ndarray): starting location with (lat,lon).
        
        Returns:
            None

        """
        self.__wgs_loc_end = loc
        x, y = WGS.latlon2xy(self.__wgs_loc_end[0], self.__wgs_loc_end[1])
        self.__loc_end = np.array([x, y])

    def set_num_steps(self, value: int) -> None:
        """ 
        Set the number of steps in the simulation to be an integer value. 

        Args:
            value (int): number of steps in the simulation.
        
        Returns:
            None

        """
        self.__num_steps = value

    def set_num_replicates(self, value: int) -> None:
        """ 
        Set the number of replicates in the simulation study.

        Args:
            value (int): number of replicates in the simulation study.
        
        Returns:
            None

        """
        self.__num_replicates = value

    def set_num_cores(self, value: int) -> None:
        """ 
        Set the number of cores to use in the simulation study. 

        Args:
            value (int): number of cores to use in the simulation study.

        Returns:
            None

        """
        self.__num_cores = value

    def get_polygon_border(self) -> np.ndarray:
        """ 
        Return polygon for the operational area in x y coordinates. 

        Args:
            None

        Returns:
            np.ndarray: polygon for the operational area in x y coordinates.

        """
        return self.__polygon_border

    def get_polygon_border_shapely(self) -> 'Polygon':
        """ 
        Return shapelized polygon for the operational area in xy coordinates. 

        Args:
            None

        Returns:
            Polygon: shapelized polygon for the operational area in xy coordinates.

        """
        return self.__polygon_border_shapely

    def get_line_border_shapely(self) -> 'LineString':
        """ 
        Return linestring of polygon border. 

        Args:
            None

        Returns:
            LineString: linestring of polygon border.

        """
        return self.__line_border_shapely

    def get_polygon_obstacle(self) -> np.ndarray:
        """ 
        Return polygon for the obstacle. 

        Args:
            None

        Returns:
            np.ndarray: polygon for the obstacle.

        """
        return self.__polygon_obstacle

    def get_polygon_obstacle_shapely(self) -> 'Polygon':
        """ 
        Return shapelized polygon for the obstacle.

        Args:
            None
        
        Returns: 
            Polygon: shapelized polygon for the obstacle.

        """
        return self.__polygon_obstacle_shapely

    def get_line_obstacle_shapely(self) -> 'LineString':
        """ 
        Return linestring of polygon obstacle. 

        Args:
            None

        Returns:
            LineString: linestring of polygon obstacle.

        """
        return self.__line_obstacle_shapely

    def get_loc_start(self) -> np.ndarray:
        """ 
        Return starting location in (x, y). 

        Args:
            None
        
        Returns:
            np.ndarray: starting location in (x, y).

        """
        return self.__loc_start

    def get_loc_end(self) -> np.ndarray:
        """ 
        Return starting location in (x, y). 

        Args:
            None
        
        Returns:
            np.ndarray: starting location in (x, y).

        """
        return self.__loc_end

    def get_num_steps(self) -> int:
        """ 
        Return the number of steps in the simulation study. 

        Args:
            None

        Returns:
            int: number of steps in the simulation study.

        """
        return self.__num_steps

    def get_num_replicates(self) -> int:
        """ 
        Return the number of replicates in the simulation study. 
        
        Args:
            None
        
        Returns:
            int: number of replicates in the simulation study.

        """
        return self.__num_replicates

    def get_num_cores(self) -> int:
        """
        Return the number of cores in the simulation study. 

        Args:
            None
        
        Returns:
            int: number of cores in the simulation study.

        """
        return self.__num_cores

    def get_wgs_polygon_border(self) -> np.ndarray:
        """ 
        Return polygon for the oprational area in wgs coordinates. 

        Args:
            None

        Returns:
            np.ndarray: polygon for the oprational area in wgs coordinates.

        """
        return self.__wgs_polygon_border

    def get_wgs_polygon_obstacle(self) -> np.ndarray:
        """ 
        Return polygon for the oprational area in wgs coordinates. 

        Args:
            None
        
        Returns:
            np.ndarray: polygon for the oprational area in wgs coordinates.

        """
        return self.__wgs_polygon_obstacle

    def get_wgs_loc_start(self) -> np.ndarray:
        """ 
        Return starting location in (lat, lon). 

        Args:
            None
        
        Returns:
            np.ndarray: starting location in (lat, lon).

        """
        return self.__wgs_loc_start

    def get_wgs_loc_end(self) -> np.ndarray:
        """ 
        Return starting location in (lat, lon). 

        Args:
            None
        
        Returns:
            np.ndarray: starting location in (lat, lon).

        """
        return self.__wgs_loc_end


if __name__ == "__main__":
    s = Config()


