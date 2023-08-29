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

        self.construct_ground_truth_field()

    def construct_ground_truth_field(self) -> None:
        """
        This function constructs the ground truth field given all the timestamps and locations.
        """
        print("Start constructing ground truth field!")
        x0 = (self.salinity_sinmod[0, :, :].flatten() +
             (self.L @ np.random.randn(len(self.L)).reshape(-1, 1)).flatten())
        xt_1 = x0
        xt = x0
        self.mu_truth = np.empty([0, len(xt)])
        self.mu_truth = np.vstack((self.mu_truth, xt))

        import matplotlib.pyplot as plt
        from matplotlib.pyplot import get_cmap
        from matplotlib.gridspec import GridSpec
        figpath = os.getcwd() + "/../../../../OneDrive - NTNU/MASCOT_PhD/Projects" \
                                "/GOOGLE/Docs/fig/Sim_2DNidelva/Simulator/groundtruth/"
        t0 = time()
        for i in range(1, len(self.timestamp_sinmod)):
            print("Time string: ", datetime.fromtimestamp(self.timestamp_sinmod[i]).strftime("%Y-%m-%d %H:%M:%S"))
            fig = plt.figure(figsize=(35, 10))
            gs = GridSpec(1, 3, figure=fig)
            ax = fig.add_subplot(gs[0])
            im = ax.scatter(self.grid_sinmod[:, 1], self.grid_sinmod[:, 0], c=xt, cmap=get_cmap("BrBG", 10),
                        vmin=10, vmax=30)
            plt.colorbar(im)
            ax.set_title("Ground truth at time: " +
                         datetime.fromtimestamp(self.timestamp_sinmod[i]).strftime("%Y-%m-%d %H:%M:%S"))
            ax.set_xlabel("East (m)")
            ax.set_ylabel("North (m)")

            ax = fig.add_subplot(gs[1])
            im = ax.scatter(self.grid_sinmod[:, 1], self.grid_sinmod[:, 0], c=self.salinity_sinmod[i, :, :].flatten(),
                        cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
            plt.colorbar(im)
            ax.set_title("SINMOD at time: " +
                         datetime.fromtimestamp(self.timestamp_sinmod[i]).strftime("%Y-%m-%d %H:%M:%S"))
            ax.set_xlabel("East (m)")
            ax.set_ylabel("North (m)")

            ax = fig.add_subplot(gs[2])
            im = ax.scatter(self.grid_sinmod[:, 1], self.grid_sinmod[:, 0],
                            c=xt - self.salinity_sinmod[i, :, :].flatten(),
                            cmap=get_cmap("RdBu", 10), vmin=-5, vmax=5)
            plt.colorbar(im)
            ax.set_title("Difference at time: " +
                            datetime.fromtimestamp(self.timestamp_sinmod[i]).strftime("%Y-%m-%d %H:%M:%S"))
            ax.set_xlabel("East (m)")
            ax.set_ylabel("North (m)")

            plt.savefig(figpath + "P_{:03d}.png".format(i))
            plt.close("all")

            xt = self.salinity_sinmod[i, :, :].flatten() + \
                 self.ar1_corr * (xt_1 - self.salinity_sinmod[i-1, :, :].flatten()) + \
                 (np.sqrt(1 - self.ar1_corr ** 2) * (self.L @ np.random.randn(len(self.L)).reshape(-1, 1))).flatten()
            xt_1 = xt
            self.mu_truth = np.vstack((self.mu_truth, xt))

        self.mu_truth
        print("Time consumed: ", time() - t0)
        xt_1

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
        sorted_salinity = self.salinity_sinmod[ind_time, :, :].flatten()
        dist, ind_loc = self.grid_sinmod_tree.query(loc)
        print("Query salinity at timestamp and location takes: ", time() - t1)


        return sorted_salinity[ind_loc] + (L @ np.random.randn(len(L)).reshape(-1, 1)).flatten()


if __name__ == "__main__":
    c = CTDSimulator()

