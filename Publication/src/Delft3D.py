"""
Delft3D takes the input from the Config class and then extract the data.
It mainly uses the wind_dir and wind_level to prepare the prior mean field.
"""
from WGS import WGS
from Config import Config
import pandas as pd
import numpy as np
import os
from shapely.geometry import Point


class Delft3D:
    """ Load setup. """
    __config = Config()
    __wind_dir = __config.get_wind_direction()
    __wind_level = __config.get_wind_level()
    __polygon_operational_area_shapely = __config.get_polygon_operational_area_shapely()

    """ Delft3D data manipulation. """
    __data = pd.read_csv(os.getcwd() + "/../prior/Nov/" + __wind_dir + "/" + __wind_level + ".csv").to_numpy()
    __lat = __data[:, 0]
    __lon = __data[:, 1]
    __salinity = __data[:, 2]
    xd, yd = WGS.latlon2xy(__lat, __lon)
    __dataset = np.stack((xd, yd, __salinity), axis=1)

    @staticmethod
    def get_dataset() -> np.ndarray:
        """ Return dataset of Delft3D.
        Example:
             dataset = np.array([[lat, lon, salinity]])
        """
        return Delft3D.__dataset


if __name__ == "__main__":
    m = Delft3D()

