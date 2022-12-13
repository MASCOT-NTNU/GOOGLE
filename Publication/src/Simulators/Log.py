"""
Log object logs the data generated during the simulation process.
"""

from Config import Config
from Simulators.CTD import CTD
import numpy as np
from scipy.stats import norm


class Log:
    """ Log """
    def __init__(self) -> None:
        self.rmse = np.empty([0, 1])
        self.ibv = np.empty([0, 1])

        self.config = Config()
        self.ctd = CTD()

        self.mu_truth = self.ctd.get_ground_truth()
        self.mu_truth

    def append_log(self, grf) -> None:
        mu = grf.get_mu()
        threshold = grf.get_threshold()
        sigma_diag = np.diag(grf.get_covariance_matrix())

        ibv = self.get_ibv(mu, sigma_diag, threshold)
        self.ibv

        pass

    def get_ibv(self, mu: np.ndarray, sigma_diag: np.ndarray, threshold: float) -> np.ndarray:
        """ !!! Be careful with dimensions, it can lead to serious problems.
        !!! Be careful with standard deviation is not variance, so it does not cause significant issues tho.
        :param mu: n x 1 dimension
        :param sigma_diag: n x 1 dimension
        :return:
        """
        p = norm.cdf(threshold, mu, np.sqrt(sigma_diag))
        bv = p * (1 - p)
        ibv = np.sum(bv)
        return ibv


if __name__ == "__main__":
    l = Log()
