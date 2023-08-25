"""
CTD module simulates the true field.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-08-24
"""
from GRF.GRF import GRF
import numpy as np
from typing import Union


class CTD:
    """
    CTD module handles the simulated truth value at each specific location.
    """
    def __init__(self, loc_start: np.ndarray = np.array([0, 0]), random_seed: int = 0,
                 sigma: float = 1., nugget: float = .1):
        # np.random.seed(0)
        """
        Set up the CTD simulated truth field.
        TODO: Check the starting location, it can induce serious problems.
        """
        np.random.seed(random_seed)

        self.grf = GRF(sigma=sigma, nugget=nugget)
        self.field = self.grf.field
        mu_prior = self.grf.get_mu()
        Sigma_prior = self.grf.get_covariance_matrix()
        self.mu_truth = mu_prior + np.linalg.cholesky(Sigma_prior) @ np.random.randn(len(mu_prior)).reshape(-1, 1)

        """
        Set up CTD data gathering
        """
        self.loc_now = loc_start
        self.loc_prev = np.array([0, 0])
        self.ctd_data = np.empty([0, 3])
        self.speed = 1.5  # m/s

    def get_ctd_data_1hz(self, loc: np.ndarray) -> np.ndarray:
        """
        Simulate CTD data gathering at 1hz.
        It means that it returns only one sample.
        Args:
            loc: np.array([x, y])
        Return:
            data: np.array([x, y, sal])
        """
        sal = self.get_salinity_at_loc(loc)
        self.ctd_data = np.stack((loc[0], loc[1], sal.flatten()[0])).reshape(1, -1)
        return self.ctd_data

    def get_ctd_data(self, loc: np.ndarray) -> np.ndarray:
        """
        Simulate CTD data gathering.
        Args:
            loc: np.array([x, y])
        Return:
            dataset: np.array([x, y, sal])
        """
        self.loc_prev = self.loc_now
        self.loc_now = loc
        x_start, y_start = self.loc_prev
        x_end, y_end = self.loc_now
        dx = x_end - x_start
        dy = y_end - y_start
        dist = np.sqrt(dx ** 2 + dy ** 2)
        N = 20
        # N = int(np.ceil(dist / self.speed) * 2)
        if N != 0:
            x_path = np.linspace(x_start, x_end, N)
            y_path = np.linspace(y_start, y_end, N)
            loc = np.stack((x_path, y_path), axis=1)
            sal = self.get_salinity_at_loc(loc)
            self.ctd_data = np.stack((x_path, y_path, sal.flatten()), axis=1)
        return self.ctd_data

    def get_salinity_at_loc(self, loc: np.ndarray) -> Union[np.ndarray, None]:
        """
        Get CTD measurement at a specific location.

        Args:
            loc: np.array([[x, y]])

        Returns:
            salinity value at loc
        """
        ind = self.field.get_ind_from_location(loc)
        if ind is not None:
            return self.mu_truth[ind]
        else:
            return None

    def get_ground_truth(self) -> np.ndarray:
        """ Return ground truth. """
        return self.mu_truth