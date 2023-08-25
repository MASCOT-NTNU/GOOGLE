"""
SINMOD module handles the data interpolation for a given set of coordinates.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-05-26

Methodology:
    1. Read SINMOD data from netCDF file.
    2. Construct KDTree for the SINMOD grid.
    3. For a given set of coordinates, find the nearest SINMOD grid point.
    4. Interpolate the data using the nearest SINMOD grid point.
"""
from WGS import WGS
from pykdtree.kdtree import KDTree
import xarray as xr
import re
import numpy as np
import netCDF4
from datetime import datetime
import time


class SINMOD:
    """
    SINMOD class handles the data interpolation for a given set of coordinates.
    """
    def __init__(self, filepath: str = None) -> None:
        if filepath is None:
            raise ValueError("Please provide the filepath to SINMOD data.")
        else:
            self.__filepath = filepath
            self.__dataset = netCDF4.Dataset(self.__filepath)
            ind_before = re.search("samples_", self.__filepath)
            ind_after = re.search(".nc", self.__filepath)
            date_string = self.__filepath[ind_before.end():ind_after.start()]
            ref_timestamp = datetime.strptime(date_string, "%Y.%m.%d").timestamp()
            self.__timestamp = np.array(self.__dataset["time"]) * 24 * 3600 + ref_timestamp  # change ref timestamp

            self.__lat = np.array(self.__dataset['gridLats'])
            self.__lon = np.array(self.__dataset['gridLons'])
            self.__x, self.__y = WGS.latlon2xy(self.__lat, self.__lon)
            self.__depth = np.array(self.__dataset['zc'])
            self.__salinity = np.array(self.__dataset['salinity'])
            salinity_sinmod_time_ave = np.mean(self.__salinity[:, :, :, :], axis=0)
            t1 = time.time()
            self.__sorted_data = []
            for i in range(self.__lat.shape[0]):
                for j in range(self.__lat.shape[1]):
                    for k in range(len(self.__depth)):
                        self.__sorted_data.append([self.__x[i, j], self.__y[i, j],
                                                   self.__depth[k], salinity_sinmod_time_ave[k, i, j]])
            self.__sorted_data = np.array(self.__sorted_data)
            self.sinmod_grid_tree = KDTree(self.__sorted_data[:, :3])
            t2 = time.time()
            print("KDTree construction time: ", t2 - t1)

    def get_data_at_locations(self, locations: np.array) -> np.ndarray:
        """
        Get SINMOD data values at given locations.

        Args:
            location: x, y, depth coordinates
            Example: np.array([[x1, y1, depth1],
                               [x2, y2, depth2],
                               ...
                               [xn, yn, depthn]])
        Returns:
            SINMOD data values at given locations.
        """
        ts = time.time()
        dist, ind = self.sinmod_grid_tree.query(locations.astype(np.float32))
        sal_interpolated = self.__sorted_data[ind, -1].reshape(-1, 1)
        df_interpolated = np.hstack((locations, sal_interpolated))
        te = time.time()
        print("Data is interpolated successfully! Time consumed: ", te - ts)
        return df_interpolated

    def get_data(self) -> np.ndarray:
        """
        Return the dataset of SINMOD data.
        """
        return self.__sorted_data

    def get_salinity(self) -> np.ndarray:
        """
        Return the salinity of SINMOD data.
        """
        return self.__salinity

    def get_timestamp(self) -> np.ndarray:
        """
        Return the timestamp of SINMOD data.
        """
        return self.__timestamp

    def get_coordinates(self) -> tuple:
        """
        Return the coordinates of SINMOD data.
        """
        return self.__x, self.__y, self.__depth


if __name__ == "__main__":
    s = SINMOD()
