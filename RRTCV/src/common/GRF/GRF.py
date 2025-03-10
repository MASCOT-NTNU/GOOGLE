"""
GRF builds the kernel for simulation study.
- udpate the field.
- assimilate data.
- get eibv for a specific location.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-08-22
"""
from Field import Field
from SINMOD import SINMOD
from usr_func.vectorize import vectorize
from usr_func.checkfolder import checkfolder
from usr_func.normalize import normalize
from usr_func.calculate_analytical_ebv import calculate_analytical_ebv
from scipy.spatial.distance import cdist
import numpy as np
from scipy.stats import norm, multivariate_normal
from numba import jit
from joblib import Parallel, delayed
from pykdtree.kdtree import KDTree
from datetime import datetime
import time
import pandas as pd
import os


class GRF:
    """
    GRF kernel
    """
    def __init__(self, filepath_prior: str = os.getcwd() + "/../sinmod/samples_2022.05.11.nc") -> None:
        self.__ar1_coef = .965  # AR1 coef, timestep is 10 mins.
        self.__ar1_corr_range = 600   # [sec], AR1 correlation time range.
        self.__approximate_eibv = False
        self.__fast_eibv = True

        """ Empirical parameters """
        # spatial variability
        self.__sigma = .5

        # spatial correlation
        # self.__lateral_range = 200  # 680 in the experiment
        self.__lateral_range = 700  # 680 in the experiment

        # measurement noise
        self.__nugget = .1

        # threshold
        self.__threshold = 26.81189868

        # matern coefficients
        self.__eta = 4.5 / self.__lateral_range  # decay factor
        self.__tau = np.sqrt(self.__nugget)  # measurement noise

        """ Conditional field """
        # kernel mean
        self.__mu = None

        # kernel covariance matrix
        self.__Sigma = None

        """ Cost valley """
        # cost valley fields
        self.__eibv_field = None
        self.__ivr_field = None

        # s0: construct grf covariance matrix.
        self.field = Field(neighbour_distance=100)
        self.grid = self.field.get_grid()
        self.grid_kdtree = KDTree(self.grid)
        self.Ngrid = len(self.grid)
        self.__Fgrf = np.ones([1, self.Ngrid])
        self.__xg = vectorize(self.grid[:, 0])
        self.__yg = vectorize(self.grid[:, 1])
        self.__distance_matrix = None
        self.__construct_grf_field()
        self.__Sigma_prior = self.__Sigma

        # s1: update prior mean
        datestring = filepath_prior.split("/")[-1].split("_")[-1][:-3].replace('.', '-') + " 10:00:00"
        timestamp_prior = np.array([datetime.strptime(datestring, "%Y-%m-%d %H:%M:%S").timestamp()])
        self.__sinmod = SINMOD(filepath_prior)
        self.__salinity_sinmod = self.__sinmod.get_salinity()[:, 0, :, :]
        x, y, *_ = self.__sinmod.get_coordinates()
        self.__grid_sinmod = np.stack((x.flatten(), y.flatten()), axis=1)
        self.__grid_sinmod_tree = KDTree(self.__grid_sinmod)
        *_, self.__ind_sinmod4grid = self.__grid_sinmod_tree.query(self.grid)
        self.__timestamp_sinmod = self.__sinmod.get_timestamp()
        self.__timestamp_sinmod_tree = KDTree(self.__timestamp_sinmod)
        *_, ind_prior_time = self.__timestamp_sinmod_tree.query(timestamp_prior)
        self.__mu = self.__salinity_sinmod[ind_prior_time, :, :].flatten()[self.__ind_sinmod4grid].reshape(-1, 1)

        # s2: load cdf table
        self.__load_cdf_table()

    def __construct_grf_field(self) -> None:
        """ Construct distance matrix and thus Covariance matrix for the kernel. """
        self.__distance_matrix = cdist(self.grid, self.grid)
        self.__Sigma = self.__sigma ** 2 * ((1 + self.__eta * self.__distance_matrix) *
                                            np.exp(-self.__eta * self.__distance_matrix))

    def __load_cdf_table(self) -> None:
        """
        Load cdf table for the analytical solution.
        """
        table = np.load("./../prior/cdf.npz")
        self.__cdf_z1 = table["z1"]
        self.__cdf_z2 = table["z2"]
        self.__cdf_rho = table["rho"]
        self.__cdf_table = table["cdf"]

    def assimilate_data(self, dataset: np.ndarray) -> None:
        """
        Assimilate dataset to GRF kernel.
        It computes the distance matrix between gmrf grid and dataset grid. Then the values are averged to each cell.
        Args:
            dataset: np.array([x, y, sal])
            cnt_waypoint: int
        """
        distance_min, ind_min_distance = self.grid_kdtree.query(dataset[:, :2])
        ind_assimilated = np.unique(ind_min_distance)
        salinity_assimilated = np.zeros([len(ind_assimilated), 1])
        for i in range(len(ind_assimilated)):
            ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
            salinity_assimilated[i] = np.mean(dataset[ind_selected, -1])
        self.__update(ind_measured=ind_assimilated, salinity_measured=salinity_assimilated)
        # t2 = time.time()
        # print("Data assimilation takes: ", t2 - t1, " seconds")

    def __update(self, ind_measured: np.ndarray, salinity_measured: np.ndarray) -> None:
        """
        Update GRF kernel based on sampled data.
        :param ind_measured: indices where the data is assimilated.
        :param salinity_measured: measurements at sampeld locations, dimension: m x 1
        """
        msamples = salinity_measured.shape[0]
        F = np.zeros([msamples, self.Ngrid])
        for i in range(msamples):
            F[i, ind_measured[i]] = True
        R = np.eye(msamples) * self.__tau ** 2
        C = F @ self.__Sigma @ F.T + R
        self.__mu = self.__mu + self.__Sigma @ F.T @ np.linalg.solve(C, (salinity_measured - F @ self.__mu))
        self.__Sigma = self.__Sigma - self.__Sigma @ F.T @ np.linalg.solve(C, F @ self.__Sigma)

    def assimilate_temporal_data(self, dataset: np.ndarray) -> tuple:
        """
        Assimilate temporal dataset to GRF kernel.
        It computes the distance matrix between grf grid and dataset grid. Then the values are averged to each cell.
        Args:
            dataset: np.array([timestamp, x, y, sal])
            cnt_waypoint: int
        Return:
            (ind, salinity) for visualising eda plots.
        """
        t_start = dataset[0, 0]
        t_end = dataset[-1, 0]
        t_steps = int((t_end - t_start) // self.__ar1_corr_range)

        *_, ind_min_distance = self.grid_kdtree.query(dataset[:, 1:3])
        ind_assimilated = np.unique(ind_min_distance)
        salinity_assimilated = np.zeros([len(ind_assimilated), 1])
        for i in range(len(ind_assimilated)):
            ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
            salinity_assimilated[i] = np.mean(dataset[ind_selected, -1])
        self.__update_temporal(ind_measured=ind_assimilated, salinity_measured=salinity_assimilated,
                               timestep=t_steps, timestamp=np.array([t_end]))
        # t2 = time.time()
        # print("Data assimilation takes: ", t2 - t1, " seconds")
        """ Just for debugging. """
        return ind_assimilated, salinity_assimilated

    def __update_temporal(self, ind_measured: np.ndarray, salinity_measured: np.ndarray,
                          timestep=0, timestamp: np.ndarray = np.array([123424332])):
        """ Update GRF kernel with AR1 process.
        timestep here can only be 1, no larger than 1, if it is larger than 1, then the data assimilation needs to be
        properly adjusted to make sure that they correspond with each other.
        """
        #s0, create sampling index matrix
        msamples = salinity_measured.shape[0]
        F = np.zeros([msamples, self.Ngrid])
        for i in range(msamples):
            F[i, ind_measured[i]] = True
        R = np.eye(msamples) * self.__tau ** 2

        # s1, get timestamped prior mean from SINMOD
        *_, ind_time = self.__timestamp_sinmod_tree.query(timestamp)
        salinity_sinmod = self.__salinity_sinmod[ind_time, :, :].flatten()
        mu_prior = salinity_sinmod[self.__ind_sinmod4grid].reshape(-1, 1)

        t1 = time.time()
        # propagate
        mt0 = mu_prior + self.__ar1_coef * (self.__mu - mu_prior)
        St0 = self.__ar1_coef ** 2 * self.__Sigma + (1 - self.__ar1_coef ** 2) * self.__Sigma_prior
        mts = mt0
        Sts = St0
        for s in range(timestep):
            mts = mu_prior + self.__ar1_coef * (mts - mu_prior)
            Sts = self.__ar1_coef**2 * Sts + (1 - self.__ar1_coef**2) * self.__Sigma_prior

        self.__mu = mts + Sts @ F.T @ np.linalg.solve(F @ Sts @ F.T + R, salinity_measured - F @ mts)
        self.__Sigma = Sts - Sts @ F.T @ np.linalg.solve(F @ Sts @ F.T + R, F @ Sts)
        t2 = time.time()
        # print("GRF-AR1 model updates takes: ", t2 - t1)

    def get_ei_field(self) -> tuple:
        t1 = time.time()
        eibv_field = np.zeros([self.Ngrid])
        ivr_field = np.zeros([self.Ngrid])
        for i in range(self.Ngrid):
            SF = self.__Sigma[:, i].reshape(-1, 1)
            MD = 1 / (self.__Sigma[i, i] + self.__nugget)
            VR = SF @ SF.T * MD
            SP = self.__Sigma - VR
            sigma_diag = np.diag(SP).reshape(-1, 1)
            if self.__approximate_eibv:
                eibv_field[i] = self.__get_eibv_approximate(self.__mu, sigma_diag)
            else:
                vr_diag = np.diag(VR).reshape(-1, 1)
                if self.__fast_eibv:
                    eibv_field[i] = self.__get_eibv_analytical_fast(mu=self.__mu, sigma_diag=sigma_diag, vr_diag=vr_diag,
                                                                    threshold=self.__threshold, cdf_z1=self.__cdf_z1,
                                                                    cdf_z2=self.__cdf_z2, cdf_rho=self.__cdf_rho,
                                                                    cdf_table=self.__cdf_table)
                else:
                    eibv_field[i] = self.__get_eibv_analytical(self.__mu, sigma_diag, vr_diag)
            ivr_field[i] = np.sum(np.diag(VR))
        self.__eibv_field = normalize(eibv_field)
        self.__ivr_field = 1 - normalize(ivr_field)
        t2 = time.time()
        print("EI field takes: ", t2 - t1, " seconds.")
        return self.__eibv_field, self.__ivr_field

    def __get_eibv_approximate(self, mu: np.ndarray, sigma_diag: np.ndarray) -> np.ndarray:
        """ !!! Be careful with dimensions, it can lead to serious problems.
        !!! Be careful with standard deviation is not variance, so it does not cause significant issues tho.
        :param mu: n x 1 dimension
        :param sigma_diag: n x 1 dimension
        :return:
        """
        p = norm.cdf(self.__threshold, mu, np.sqrt(sigma_diag))
        bv = p * (1 - p)
        ibv = np.sum(bv)
        return ibv

    def __get_eibv_analytical(self, mu: np.ndarray, sigma_diag: np.ndarray, vr_diag: np.ndarray) -> float:
        """
        Calculate the eibv using the analytical formula with a bivariate cumulative dentisty function.
        """
        eibv = .0
        for i in range(len(mu)):
            sn2 = sigma_diag[i]
            vn2 = vr_diag[i]

            sn = np.sqrt(sn2)
            m = mu[i]

            mur = (self.__threshold - m) / sn

            sig2r_1 = sn2 + vn2
            sig2r = vn2

            eibv += multivariate_normal.cdf(np.array([0, 0]), np.array([-mur, mur]).squeeze(),
                                            np.array([[sig2r_1, -sig2r],
                                                      [-sig2r, sig2r_1]]).squeeze())
        return eibv

    @staticmethod
    @jit
    def __get_eibv_analytical_fast(mu: np.ndarray, sigma_diag: np.ndarray, vr_diag: np.ndarray,
                                   threshold: float, cdf_z1: np.ndarray, cdf_z2: np.ndarray,
                                   cdf_rho: np.ndarray, cdf_table: np.ndarray) -> float:
        """
        Calculate the eibv using the analytical formula but using a loaded cdf dataset.
        """
        eibv = .0
        for i in range(len(mu)):
            sn2 = sigma_diag[i]
            vn2 = vr_diag[i]

            sn = np.sqrt(sn2)
            m = mu[i]

            mur = (threshold - m) / sn

            sig2r_1 = sn2 + vn2
            sig2r = vn2

            z1 = mur
            z2 = -mur
            rho = -sig2r / sig2r_1

            ind1 = np.argmin(np.abs(z1 - cdf_z1))
            ind2 = np.argmin(np.abs(z2 - cdf_z2))
            ind3 = np.argmin(np.abs(rho - cdf_rho))
            eibv += cdf_table[ind1][ind2][ind3]
        return eibv

    def __get_eibv_analytical_para(self, mu: np.ndarray, sigma_diag: np.ndarray, vr_diag: np.ndarray) -> float:
        """
        Calculate the eibv using the analytical formula with a bivariate cumulative dentisty function.

        """
        eibv = .0

        threshold = self.__threshold * np.ones_like(mu.flatten())
        sn2 = sigma_diag.flatten()
        sn = np.sqrt(sn2)
        vn2 = vr_diag.flatten()

        mur = (threshold - mu.flatten()) / sn

        sig2r_1 = sn2 + vn2
        sig2r = vn2

        # for i in range(len(mu)):
        #     sn2 = sigma_diag[i]
        #     vn2 = vr_diag[i]
        #
        #     sn = np.sqrt(sn2)
        #     m = mu[i]
        #
        #     mur = (self.__threshold - m) / sn
        #
        #     sig2r_1 = sn2 + vn2
        #     sig2r = vn2
        parameter_sets = np.stack((mur, sig2r_1, sig2r), axis=1)

        # for ps in parameter_sets:
        #     print(ps)

        # Parallel(n_jobs=6)(delayed(makeGraph)(graph_type=graph, nodes=vertex, edge_probability=prob, power_exponent=exponent) for vertex in vertices for prob in edge_probabilities for exponent in power_exponents for graph in graph_types)
        res = Parallel(n_jobs=30)(delayed(calculate_analytical_ebv)(ps) for ps in parameter_sets)

        eibv = sum(res)
        return eibv

    def set_sigma(self, value: float) -> None:
        """ Set space variability. """
        self.__sigma = value

    def set_lateral_range(self, value: float) -> None:
        """ Set lateral range. """
        self.__lateral_range = value

    def set_nugget(self, value: float) -> None:
        """ Set nugget. """
        self.__nugget = value

    def set_threshold(self, value: float) -> None:
        """ Set threshold. """
        self.__threshold = value

    def set_mu(self, value: np.ndarray) -> None:
        """ Set mean of the field. """
        self.__mu = value

    def get_sigma(self) -> float:
        """ Return variability of the field. """
        return self.__sigma

    def get_lateral_range(self) -> float:
        """ Return lateral range. """
        return self.__lateral_range

    def get_nugget(self) -> float:
        """ Return nugget of the field. """
        return self.__nugget

    def get_threshold(self) -> float:
        """ Return threshold. """
        return self.__threshold

    def get_mu(self) -> np.ndarray:
        """ Return mean vector. """
        return self.__mu

    def get_covariance_matrix(self) -> np.ndarray:
        """ Return Covariance. """
        return self.__Sigma

    def get_eibv_field(self) -> np.ndarray:
        """ Return the computed eibv field, given which method to be called. """
        return self.__eibv_field

    def get_ivr_field(self) -> np.ndarray:
        """ Return the computed ivr field, given which method to be called. """
        return self.__ivr_field


if __name__ == "__main__":
    g = GRF()
