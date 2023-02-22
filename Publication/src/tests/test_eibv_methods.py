"""
This test checks if two methods produce the same result and time analysis
"""
from unittest import TestCase
from GRF.GRF import GRF
# from usr_func.EIBV import EIBV_mvn, EIBV_norm
from scipy.stats import multivariate_normal, norm
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from time import time
from tqdm import tqdm

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20

class TestEIBV(TestCase):

    def setUp(self) -> None:
        """ Set the parameters. """
        self.sigma = 1
        self.nugget = .4
        # self.grf = GRF(self.sigma, self.nugget)
        # self.mu = self.grf.get_mu()
        # self.cov = self.grf.get_covariance_matrix()
        # self.threshold = self.grf.get_threshold()

    def test_3d_visualisation(self) -> None:
        sigma_ex = 2
        tau_ex = .55
        rho_ex = .0
        """ check EIBV calculation. """
        mu = 28
        threshold = 28.1
        sigma = np.linspace(0.01, 2, 6)
        tau = np.linspace(0.01, 1, 8)    # sqrt of nugget
        rho = np.linspace(0, 1, 15)    # correlation coefficient

        eibv_appr = np.zeros([len(rho), len(sigma), len(tau)])
        eibv_analy = np.zeros_like(eibv_appr)

        for i in range(len(rho)):
            for j in range(len(sigma)):
                for k in range(len(tau)):
                    rho_tmp = rho[i]
                    sigma_tmp = sigma[j]
                    tau_tmp = tau[k]
                    cov = rho_tmp * sigma_tmp ** 2
                    variance_reduction = cov / (sigma_tmp**2 + tau_tmp**2) * cov
                    sigma_post = sigma_tmp**2 - variance_reduction
                    # sigma_psqrt = np.sqrt(sigma_post)

                    def calc_eibv_approx(threshold, mu, sigma) -> float:
                        """ Calculate EIBV based on the numerical approximation.
                        """
                        ep = norm.cdf(threshold, mu, sigma)  # excursion probability
                        ibv = ep * (1 - ep)
                        return ibv

                    def calc_eibv_analy(threshold, mu, sigma_post, variance_reduction) -> float:
                        """ Calculate EIBV based on the analytical solution. """
                        eibv = multivariate_normal.cdf(np.array([threshold, -threshold]), np.array([mu, -mu]),
                                                        np.array([[sigma_post + variance_reduction, -variance_reduction],
                                                                  [-variance_reduction, sigma_post + variance_reduction]]))
                        return eibv

                    eibv_appr[i][j][k] = calc_eibv_approx(threshold, mu, sigma_post)
                    eibv_analy[i][j][k] = calc_eibv_analy(threshold, mu, sigma_post, variance_reduction)

        filepath = "/Users/yaolin/Downloads/fig/"
        np.save(filepath + "eibv_appr.npy", eibv_appr)
        np.save(filepath + "eibv_analy.npy", eibv_analy)

        # eibv_appr = np.load(filepath + "eibv_appr.npy")
        # eibv_analy = np.load(filepath + "eibv_analy.npy")

        def plotf(i_nugget: int = 0):
            fig = plt.figure(figsize=(30, 10))
            gs = GridSpec(nrows=1, ncols=3)
            ax = fig.add_subplot(gs[0])
            [plt.plot(rho, eibv_appr[:, i, i_nugget], label=r"$\sigma = {:.2f}$".format(sigma[i])) for i in range(sigma.shape[0])]
            plt.xlabel(r"Correlation coefficient $\rho$")
            plt.ylabel("EIBV estimation")
            plt.title("Approximate methods result")
            plt.legend(loc="lower left")
            plt.grid()
            plt.xlim([-.001, 1])
            plt.ylim([-.001, .26])

            ax = fig.add_subplot(gs[1])
            [plt.plot(rho, eibv_analy[:, i, i_nugget], label=r"$\sigma = {:.2f}$".format(sigma[i])) for i in range(sigma.shape[0])]
            # [plt.plot(rho, eibv_analy[:, i, 1], label=r"Analytical $\Phi_2 $") for i in range(sigma.shape[0])]
            plt.xlabel(r"Correlation coefficient $\rho$")
            plt.ylabel("EIBV estimation")
            plt.title("Analytical methods result")
            plt.legend(loc="lower left")
            plt.grid()
            plt.xlim([-.001, 1])
            plt.ylim([-.001, .26])

            ax = fig.add_subplot(gs[2])
            [plt.plot(rho, np.abs(eibv_appr[:, i, i_nugget] - eibv_analy[:, i, 0]),
                      label=r"$\sigma = {:.2f}$".format(sigma[i])) for i in range(sigma.shape[0])]
            plt.xlabel(r"Correlation coefficient $\rho$")
            plt.ylabel("Absolute")
            plt.title("Absolute discrepancy")
            plt.legend(loc="upper left")
            plt.grid()
            plt.xlim([-.001, 1])
            plt.ylim([-.001, .2])

            plt.suptitle(r"Nugget is $\tau = {:.2f}$".format(tau[i_nugget]))
            plt.savefig(filepath + "P_nugget_{:.2f}.jpg".format(tau[i_nugget]))
            plt.close("all")

        for i in range(len(tau)):
            plotf(i)
        # plt.savefig("/Users/yaolin/Downloads/fig/eibv_comp.pdf")
        # plt.gca().set_aspect("equal")
        # plt.show()
        plt.show()

        # grid = []
        # for i in range(len(rho)):
        #     for j in range(len(sigma)):
        #         for k in range(len(tau)):
        #             grid.append([rho[i], sigma[j], tau[k], eibv_appr[i][j][k]])
        # grid = np.array(grid)

        grid
        pass

    # def test_eibv(self) -> None:
    #
    #     """ Section I: check correlation plot. """
    #     # mu = np.zeros(2)
    #     # Cov = np.array([[1, .84],
    #     #                 [.84, 1]])
    #     # CovN = np.array([[1, -.84],
    #     #                  [-.84, 1]])
    #     # x1 = np.arange(-3, 3, .1)
    #     # x2 = np.arange(-3, 3, .1)
    #     #
    #     # xx1, xx2 = np.meshgrid(x1, x2)
    #     #
    #     # Fx = np.zeros_like(xx1)
    #     # px = np.zeros_like(Fx)
    #     # qx = np.zeros_like(px)
    #     #
    #     # for i in range(xx1.shape[0]):
    #     #     for j in range(xx2.shape[1]):
    #     #         Fx[i, j] = multivariate_normal.cdf(np.array([xx1[i, j], xx2[i, j]]), mu, Cov)
    #     #         px[i, j] = multivariate_normal.pdf(np.array([xx1[i, j], xx2[i, j]]), mu, Cov)
    #     #         qx[i, j] = multivariate_normal.pdf(np.array([xx1[i, j], xx2[i, j]]), mu, CovN)
    #     #
    #     # " Here comes the plotting section. "
    #     # def plotf_contour(x, y, v, title):
    #     #     plt.figure()
    #     #     c = plt.contour(x, y, v)
    #     #     plt.gca().clabel(c, inline=True, fontsize=10)
    #     #     plt.title(title)
    #     #     plt.xlabel(r"$Z_a$")
    #     #     plt.ylabel(r"$Z_b$")
    #     #     plt.gca().set_aspect("equal")
    #     #     plt.show()
    #
    #     # plotf_contour(xx1, xx2, Fx, "Cumulative density function")
    #     # plotf_contour(xx1, xx2, px, "Probability density function, positive correlation")
    #     # plotf_contour(xx1, xx2, qx, "Probability density function, negative correlation")
    #
    #
    #     eibv_appr
    #
    #     """ TOCHECK: """
    #     # a = (m - T) / sn
    #     # b = 1 / sn @ np.linalg.solve(sig2, k.T)
    #     # c = 1 + b.T * sig2 * b
    #     #
    #     # ''' MC '''
    #     # B = 1000000
    #     # u = sig * np.random.randn(B)
    #     # IntMC = np.mean(norm.cdf(a + b * u) * norm.cdf(-a - b * u))
    #     #
    #     # ''' Analytical - version 1 (Chevalier et al., 2014) '''
    #     # IntA = multivariate_normal.cdf(np.array([a, a]).squeeze(), [0, 0], np.array([[c, 1-c],
    #     #                                                                              [1-c, c]]).squeeze())
    #
    #     ''' Analytical - version 2 (should give same answer, centered differently) '''
    #     # mur = (threshold - mu) / sigma_psqrt
    #     # variance_reduction = (1 / sigma_post) * cov / (sigma**2 + tau**2) * cov
    #     # sigma_sq = 1 + variance_reduction
    #     # IntA2 = multivariate_normal.cdf(np.array([0, 0]), np.array([-mur, mur]),
    #     #                                 np.array([[sigma_sq, -variance_reduction],
    #     #                                           [-variance_reduction, sigma_sq]]))
    #
    #     ''' Analytical - version 3 (should give same answer, centered differently) '''
    #     # variance_reduction = cov / (sigma**2 + tau**2) * cov
    #     # IntA3 = multivariate_normal.cdf(np.array([threshold, -threshold]), np.array([mu, -mu]),
    #     #                                 np.array([[sigma**2, -variance_reduction],
    #     #                                           [-variance_reduction, sigma**2]]))
    #
    #     ''' Analytical - version 4 (remove extra layers for a better understanding) '''
    #     # IntA4 = multivariate_normal.cdf(np.array([threshold, -threshold]), np.array([mu, -mu]),
    #     #                                 np.array([[sigma_post + variance_reduction, -variance_reduction],
    #     #                                           [-variance_reduction, sigma_post + variance_reduction]]))
    #
    #     px
    #
    #     pass

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


