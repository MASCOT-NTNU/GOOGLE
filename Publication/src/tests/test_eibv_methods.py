"""
This test checks if two methods produce the same result and time analysis
"""
from unittest import TestCase
from GRF.GRF import GRF
from usr_func.EIBV import EIBV_mvn, EIBV_norm
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm


class TestEIBV(TestCase):
    def setUp(self) -> None:
        """ Set the parameters. """
        self.sigma = 1
        self.nugget = .4
        self.grf = GRF(self.sigma, self.nugget)
        self.mu = self.grf.get_mu()
        self.cov = self.grf.get_covariance_matrix()
        self.threshold = self.grf.get_threshold()

    def test_eibv_calculations(self) -> None:
        """
        Compare the results from two different implementations.
        """
        eibv_mvn = np.zeros_like(self.mu)
        eibv_norm = np.zeros_like(self.mu)

        t_mvn = []
        t_norm = []
        for i in tqdm(range(len(self.mu))):
            H = np.zeros_like(self.mu).T
            H[0, i] = True

            t1 = time()
            eibv_mvn[i] = EIBV_mvn(self.threshold, self.mu, self.cov, H, self.nugget)
            t2 = time()
            t_mvn.append(t2 - t1)

            t1 = time()
            eibv_norm[i] = EIBV_norm(self.threshold, self.mu, self.cov, H, self.nugget)
            t2 = time()
            t_norm.append(t2 - t1)

        print("MVN takes: ", np.sum(t_mvn) / len(t_mvn))
        print("Norm takes: ", np.sum(t_norm) / len(t_norm))
        print("Result absolute discrepancy: ", np.sum(np.abs(eibv_mvn - eibv_norm)))


