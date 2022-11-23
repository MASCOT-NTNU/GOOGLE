"""
CTDSimulator module simulates CTD sensor.
"""

from GRF.GRF import GRF
import numpy as np
from typing import Union
from scipy.spatial.distance import cdist


class CTDSimulator:
    """
    CTD module handles the simulated truth value at each specific location.
    """
    __grf = None
    __field = None
    __truth = None

    def __init__(self):
        # np.random.seed(0)
        """
        Set up the CTD simulated truth field.
        """
        self.__grf = GRF()
        self.__field = self.__grf.field
        mu_prior = self.__grf.get_mu()
        Sigma_prior = self.__grf.get_Sigma()
        self.__truth = mu_prior + np.linalg.cholesky(Sigma_prior) @ np.random.randn(len(mu_prior)).reshape(-1, 1)


    def get_salinity_at_loc(self, loc: np.ndarray) -> Union[np.ndarray, None]:
        """
        Get CTD measurement at a specific location.

        Args:
            loc: np.array([[x, y]])

        Returns:
            salinity value at loc
        """
        ind = self.__field.get_ind_from_location(loc)
        if ind is not None:
            return self.__truth[ind]
        else:
            return None

    def get_ground_truth(self):
        return self.__truth

