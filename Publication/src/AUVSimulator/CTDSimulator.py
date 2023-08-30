"""
CTDSimulator simulates the CTD sensor sampling in the ground truth field.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-08-28
"""
from SINMOD import SINMOD
from GRF.GRF import GRF
import os
from datetime import datetime
import numpy as np
from typing import Union
from pykdtree.kdtree import KDTree
from scipy.spatial.distance import cdist
from time import time


class CTDSimulator:
    """
    CTD module handles the simulated truth value at each specific location.
    """
    def __init__(self, random_seed: int = 0,
                 filepath: str = os.getcwd() + "/../sinmod/samples_2022.05.11.nc", sigma: float = 1.) -> None:
        """
        Set up the CTD simulated truth field.
        """
        # Load SINMOD data from a specific file
        np.random.seed(random_seed)
        filepath_sinmod = filepath
        datestring = filepath_sinmod.split("/")[-1].split("_")[-1][:-3].replace('.', '-') + " 10:00:00"
        self.timestamp = datetime.strptime(datestring, "%Y-%m-%d %H:%M:%S").timestamp()
        self.sinmod = SINMOD(filepath_sinmod)

        # Sort out the timestamped salinity
        self.timestamp_sinmod = self.sinmod.get_timestamp()
        self.timestamp_sinmod_tree = KDTree(self.timestamp_sinmod.reshape(-1, 1))
        self.salinity_sinmod = self.sinmod.get_salinity()[:, 0, :, :]

        # Get SINMOD grid points
        self.grid_sinmod = self.sinmod.get_data()[:, :3]
        ind_surface = np.where(self.grid_sinmod[:, -1] == 0.5)[0]
        self.grid_sinmod = self.grid_sinmod[ind_surface, :2]
        self.grid_sinmod_tree = KDTree(self.grid_sinmod)

        # Set up essential parameters
        self.ar1_corr = .965
        self.sigma = sigma
        l_range = 700
        eta = 4.5 / l_range
        t0 = time()
        dm = cdist(self.grid_sinmod, self.grid_sinmod)
        cov = self.sigma ** 2 * ((1 + eta * dm) * np.exp(-eta * dm))
        self.L = np.linalg.cholesky(cov)
        print("Cholesky decomposition takes: ", time() - t0)

        self.mu_truth = np.empty([0, len(self.grid_sinmod)])
        self.construct_ground_truth_field()

    def construct_ground_truth_field(self) -> None:
        """
        This function constructs the ground truth field given all the timestamps and locations.
        """
        x0 = (self.salinity_sinmod[0, :, :].flatten() +
             (self.L @ np.random.randn(len(self.L)).reshape(-1, 1)).flatten())
        xt_1 = x0
        xt = x0
        self.mu_truth = np.vstack((self.mu_truth, xt))

        t0 = time()
        for i in range(1, len(self.timestamp_sinmod)):
            xt = self.salinity_sinmod[i, :, :].flatten() + \
                 self.ar1_corr * (xt_1 - self.salinity_sinmod[i-1, :, :].flatten()) + \
                 (np.sqrt(1 - self.ar1_corr ** 2) * (self.L @ np.random.randn(len(self.L)).reshape(-1, 1))).flatten()
            xt_1 = xt
            self.mu_truth = np.vstack((self.mu_truth, xt))
        print("Constructing ground truth field takes: ", time() - t0)

    def get_salinity_at_dt_loc(self, dt: float, loc: np.ndarray) -> Union[np.ndarray, None]:
        """
        Get CTD measurement at a given time and location.

        Args:
            dt: time difference in seconds
            loc: np.array([x, y])
        """
        self.timestamp += dt
        print("Current datetime: ", datetime.fromtimestamp(self.timestamp).strftime("%Y-%m-%d %H:%M:%S"))
        ts = np.array([self.timestamp])
        dist, ind_time = self.timestamp_sinmod_tree.query(ts)
        t1 = time()
        sorted_salinity = self.mu_truth[ind_time, :].flatten()
        # sorted_salinity = self.salinity_sinmod[ind_time, :, :].flatten()
        dist, ind_loc = self.grid_sinmod_tree.query(loc)
        print("Query salinity at timestamp and location takes: ", time() - t1)
        return sorted_salinity[ind_loc]


if __name__ == "__main__":
    c = CTDSimulator()

