"""
This test checks if two methods produce the same result and time analysis
"""
from unittest import TestCase
from GRF.GRF import GRF
# from usr_func.EIBV import EIBV_mvn, EIBV_norm
from scipy.stats import multivariate_normal, norm
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm


class TestEIBV(TestCase):

    def setUp(self) -> None:
        """ Set the parameters. """
        self.sigma = 1
        self.nugget = .4
        # self.grf = GRF(self.sigma, self.nugget)
        # self.mu = self.grf.get_mu()
        # self.cov = self.grf.get_covariance_matrix()
        # self.threshold = self.grf.get_threshold()

    def test_eibv(self) -> None:

        """ Section I: check correlation plot. """
        # mu = np.zeros(2)
        # Cov = np.array([[1, .84],
        #                 [.84, 1]])
        # CovN = np.array([[1, -.84],
        #                  [-.84, 1]])
        # x1 = np.arange(-3, 3, .1)
        # x2 = np.arange(-3, 3, .1)
        #
        # xx1, xx2 = np.meshgrid(x1, x2)
        #
        # Fx = np.zeros_like(xx1)
        # px = np.zeros_like(Fx)
        # qx = np.zeros_like(px)
        #
        # for i in range(xx1.shape[0]):
        #     for j in range(xx2.shape[1]):
        #         Fx[i, j] = multivariate_normal.cdf(np.array([xx1[i, j], xx2[i, j]]), mu, Cov)
        #         px[i, j] = multivariate_normal.pdf(np.array([xx1[i, j], xx2[i, j]]), mu, Cov)
        #         qx[i, j] = multivariate_normal.pdf(np.array([xx1[i, j], xx2[i, j]]), mu, CovN)
        #
        # " Here comes the plotting section. "
        # def plotf_contour(x, y, v, title):
        #     plt.figure()
        #     c = plt.contour(x, y, v)
        #     plt.gca().clabel(c, inline=True, fontsize=10)
        #     plt.title(title)
        #     plt.xlabel(r"$Z_a$")
        #     plt.ylabel(r"$Z_b$")
        #     plt.gca().set_aspect("equal")
        #     plt.show()

        # plotf_contour(xx1, xx2, Fx, "Cumulative density function")
        # plotf_contour(xx1, xx2, px, "Probability density function, positive correlation")
        # plotf_contour(xx1, xx2, qx, "Probability density function, negative correlation")


        """ Section II: check EIBV calculation. """
        m = 8
        T = 8.1
        s = .8
        tt = .5
        corr = .9
        sig2 = s**2 + tt**2
        sig = np.sqrt(sig2)
        ''' Prior ibv '''
        # ibv = norm.cdf((T - m)/s) * norm.cdf((m - T)/s)

        k = corr * s ** 2
        sn2 = s**2 - k / sig2 * k
        sn = np.sqrt(sn2)


        """ TOCHECK: """
        # a = (m - T) / sn
        # b = 1 / sn @ np.linalg.solve(sig2, k.T)
        # c = 1 + b.T * sig2 * b
        #
        # ''' MC '''
        # B = 1000000
        # u = sig * np.random.randn(B)
        # IntMC = np.mean(norm.cdf(a + b * u) * norm.cdf(-a - b * u))
        #
        # ''' Analytical - version 1 (Chevalier et al., 2014) '''
        # IntA = multivariate_normal.cdf(np.array([a, a]).squeeze(), [0, 0], np.array([[c, 1-c],
        #                                                                              [1-c, c]]).squeeze())

        ''' Analytical - version 2 (should give same answer, centered differently) '''
        mur = (T - m) / sn
        sig2r = (1 / sn2) * k / sig2 * k
        sig2r_1 = 1 + sig2r
        IntA2 = multivariate_normal.cdf(np.array([0, 0]), np.array([-mur, mur]),
                                        np.array([[sig2r_1, -sig2r],
                                                  [-sig2r, sig2r_1]]))

        ''' Analytical - version 3 (should give same answer, centered differently) '''
        sig2r = k / sig2 * k
        sig2r_1 = sn2 + sig2r
        IntA3 = multivariate_normal.cdf(np.array([T, -T]), np.array([m, -m]),
                                        np.array([[sig2r_1, -sig2r],
                                                  [-sig2r, sig2r_1]]))

        px

        pass

    # def test_eibv_calculations(self) -> None:
    #     """
    #     Compare the results from two different implementations.
    #     """
    #     eibv_mvn = np.zeros_like(self.mu)
    #     eibv_norm = np.zeros_like(self.mu)
    #
    #     t_mvn = []
    #     t_norm = []
    #     for i in tqdm(range(len(self.mu))):
    #         H = np.zeros_like(self.mu).T
    #         H[0, i] = True
    #
    #         t1 = time()
    #         eibv_mvn[i] = EIBV_mvn(self.threshold, self.mu, self.cov, H, self.nugget)
    #         t2 = time()
    #         t_mvn.append(t2 - t1)
    #
    #         t1 = time()
    #         eibv_norm[i] = EIBV_norm(self.threshold, self.mu, self.cov, H, self.nugget)
    #         t2 = time()
    #         t_norm.append(t2 - t1)
    #
    #     print("MVN takes: ", np.sum(t_mvn) / len(t_mvn))
    #     print("Norm takes: ", np.sum(t_norm) / len(t_norm))
    #     print("Result absolute discrepancy: ", np.sum(np.abs(eibv_mvn - eibv_norm)))


