"""
GRF builds the kernel for simulation study.
- udpate the field.
- assimilate data.
- get eibv for a specific location.
"""
from Field import Field
from usr_func.vectorize import vectorize
from usr_func.checkfolder import checkfolder
from scipy.spatial.distance import cdist
import numpy as np
from scipy.stats import norm
from usr_func.normalize import normalize
import time
import pandas as pd


class GRF:
    """
    GRF kernel
    """
    def __init__(self) -> None:
        """ Initializes the parameters in GRF kernel. """

        """ Empirical parameters """
        # spatial variability
        self.__sigma = .3

        # spatial correlation
        self.__lateral_range = 480  # 680 in the experiment

        # measurement noise
        self.__nugget = .01

        # threshold
        self.__threshold = 27

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
        self.field = Field()
        self.grid = self.field.get_grid()
        self.Ngrid = len(self.grid)
        self.__Fgrf = np.ones([1, self.Ngrid])
        self.__xg = vectorize(self.grid[:, 0])
        self.__yg = vectorize(self.grid[:, 1])
        self.__distance_matrix = None
        self.__construct_grf_field()

        # s1: update prior mean
        self.__construct_prior_mean()

    def __construct_grf_field(self) -> None:
        """ Construct distance matrix and thus Covariance matrix for the kernel. """
        self.__distance_matrix = cdist(self.grid, self.grid)
        self.__Sigma = self.__sigma ** 2 * ((1 + self.__eta * self.__distance_matrix) *
                                            np.exp(-self.__eta * self.__distance_matrix))

    def __construct_prior_mean(self) -> None:
        # s0: get delft3d dataset
        dataset_sinmod = pd.read_csv("./../prior/sinmod.csv").to_numpy()
        grid_sinmod = dataset_sinmod[:, :2]
        sal_sinmod = dataset_sinmod[:, -1]

        # s1: interpolate onto grid.
        dm_grid_delft3d = cdist(self.grid, grid_sinmod)
        ind_close = np.argmin(dm_grid_delft3d, axis=1)
        self.__mu = vectorize(sal_sinmod[ind_close])

    def assimilate_data(self, dataset: np.ndarray) -> None:
        """
        Assimilate dataset to GRF kernel.
        It computes the distance matrix between gmrf grid and dataset grid. Then the values are averged to each cell.
        Args:
            dataset: np.array([x, y, sal])
            cnt_waypoint: int
        """
        # t1 = time.time()
        xd = dataset[:, 0].reshape(-1, 1)
        yd = dataset[:, 1].reshape(-1, 1)
        Fdata = np.ones([dataset.shape[0], 1])
        # t1 = time.time()
        dx = (xd @ self.__Fgrf - Fdata @ self.__xg.T) ** 2
        dy = (yd @ self.__Fgrf - Fdata @ self.__yg.T) ** 2
        dist = dx + dy
        ind_min_distance = np.argmin(dist, axis=1)  # used only for unittest.
        ind_assimilated = np.unique(ind_min_distance)
        salinity_assimilated = np.zeros([len(ind_assimilated), 1])
        for i in range(len(ind_assimilated)):
            ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
            salinity_assimilated[i] = np.mean(dataset[ind_selected, -1])
        self.__update(ind_measured=ind_assimilated, salinity_measured=salinity_assimilated)
        # t2 = time.time()
        # print("Data assimilation takes: ", t2 - t1, " seconds")

    def __update(self, ind_measured: np.ndarray, salinity_measured: np.ndarray):
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
            eibv_field[i] = self.__get_ibv(self.__mu, sigma_diag)
            ivr_field[i] = np.sum(np.diag(VR))
        self.__eibv_field = normalize(eibv_field)
        self.__ivr_field = 1 - normalize(ivr_field)
        t2 = time.time()
        # print("Total EI field takes: ", t2 - t1, " seconds.")
        return self.__eibv_field, self.__ivr_field

    def __get_ibv(self, mu: np.ndarray, sigma_diag: np.ndarray) -> np.ndarray:
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
