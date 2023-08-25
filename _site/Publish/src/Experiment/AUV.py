"""
AUV module simulates an autonomous underwater vehicle (AUV) to collect data.
"""
from WGS import WGS
import numpy as np
import pandas as pd


class AUV:
    def __init__(self) -> None:
        """
        Constructor of AUV class.
        """
        self.__raw_dataset = None
        self.__dataset = None
        self.__load_auv_data()

    def __load_auv_data(self) -> None:
        """
        Load AUV data from synchronized csv file. Those data are collected from the AUV in the field experiment.

        Methodology:
            1. Read csv file using pandas.
            2. Convert pandas dataframe to numpy array.
            3. Extract timestamp, latitude, longitude, depth, salinity, and temperature from the numpy array.
            4. Convert latitude and longitude to UTM coordinates.
            5. Filter values when the AUV is at surface or too deep (0.25m < depth < 1.0 m).
            6. Concatenate timestamp, x, y, and salinity together to form the dataset.

        Examples:
            >>> auv = AUV()
            >>> auv.get_dataset()
            array([[ 1.60000000e+01,  1.00000000e+00,  1.00000000e+00,
                     3.00000000e+00,  3.00000000e+00,  3.00000000e+00,
                     3.00000000e+00,  3.00000000e+00,  3.00000000e+00]])

        Returns:
            None

        """
        self.__raw_dataset = pd.read_csv("./../auv/data_sync.csv").to_numpy()
        timestamp = self.__raw_dataset[:, 0]
        lat = self.__raw_dataset[:, 1]
        lon = self.__raw_dataset[:, 2]
        depth = self.__raw_dataset[:, 3]
        salinity = self.__raw_dataset[:, 4]
        temperature = self.__raw_dataset[:, 5]
        x, y = WGS.latlon2xy(lat, lon)
        # Filter depth noise before concatenate them together
        ind_filtered = np.where((depth >= .25) * (depth <= 1.))[0]
        self.__dataset = np.stack((timestamp[ind_filtered], x[ind_filtered],
                                   y[ind_filtered], salinity[ind_filtered]), axis=1)

    def get_dataset(self) -> np.ndarray:
        """
        Get dataset of AUV.

        Examples:
            >>> auv = AUV()
            >>> auv.get_dataset()
            array([[ 1.60000000e+01,  1.00000000e+00,  1.00000000e+00,
                        3.00000000e+00,  3.00000000e+00,  3.00000000e+00,
                        3.00000000e+00,  3.00000000e+00,  3.00000000e+00]])

        Returns:
            np.ndarray: Dataset of AUV.

        """
        return self.__dataset


if __name__ == "__main__":
    a = AUV()
