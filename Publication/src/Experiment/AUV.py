"""
AUV replicates the trajectory and data collected from the field experiment.
"""
from WGS import WGS
import numpy as np
import pandas as pd


class AUV:
    """ AUV class contains essential information to handle EDA. """
    def __init__(self):
        self.__raw_dataset = None
        self.__dataset = None
        self.__load_auv_data()

    def __load_auv_data(self) -> None:
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
        """ Return AUV dataset of interest. """
        return self.__dataset


if __name__ == "__main__":
    a = AUV()
