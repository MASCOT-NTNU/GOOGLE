""" GRF object handles GRF-related functions. """

from Field import Field
from scipy.spatial.distance import cdist
import numpy as np
from scipy.stats import norm
from usr_func.normalize import normalize
import time


class GRF:
    # set
    __distance_matrix = None
    __sigma = .1
    __lateral_range = .7
    __nugget = .03
    __threshold = .7

    # computed
    __eta = 4.5 / __lateral_range  # decay factor
    __tau = np.sqrt(__nugget)  # measurement noise

    # properties
    __mu = None
    __Sigma = None
    __eibv_field = None
    __ivr_field = None

    def __init__(self) -> None:
        np.random.seed(0)
        # s1: set up field
        self.field = Field()

        # s2: get grid
        self.grid = self.field.get_grid()
        self.Ngrid = len(self.grid)

        # s3: compute matern kernel
        self.__construct_grf_field()

        # s4: update prior mean
        x = self.grid[:, 0]
        y = self.grid[:, 1]
        # self.__mu = (.7 * (1 - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .07)) +
        #              .3 * (1 - np.exp(- ((x - .5) ** 2 + (y - 1.) ** 2) / .07))).reshape(-1, 1)
        self.__mu = (1. - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .07)).reshape(-1, 1)
        # self.__mu = (1. - np.exp(- ((x - .0) ** 2 + (y - .5) ** 2) / .07)).reshape(-1, 1)

    def __construct_grf_field(self):
        self.__distance_matrix = cdist(self.grid, self.grid)
        self.__Sigma = self.__sigma ** 2 * ((1 + self.__eta * self.__distance_matrix) *
                                            np.exp(-self.__eta * self.__distance_matrix))

    def assimilate_data(self, dataset: np.ndarray) -> None:
        """
        Assimilate dataset to GRF kernel.
        It computes the distance matrix between gmrf grid and dataset grid. Then the values are averged to each cell.
        Args:
            dataset: np.array([x, y, sal])
        """
        # t1 = time.time()
        xd = dataset[:, 0].reshape(-1, 1)
        yd = dataset[:, 1].reshape(-1, 1)
        Fgrf = np.ones([1, self.Ngrid])
        Fdata = np.ones([dataset.shape[0], 1])
        xg = self.grid[:, 0].reshape(-1, 1)
        yg = self.grid[:, 1].reshape(-1, 1)
        # t1 = time.time()
        dx = (xd @ Fgrf - Fdata @ xg.T) ** 2
        dy = (yd @ Fgrf - Fdata @ yg.T) ** 2
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
        # t1 = time.time()
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
        # t2 = time.time()
        # print("EI field takes: ", t2 - t1, " seconds.")
        return self.__eibv_field, self.__ivr_field

    def __get_ibv(self, mu: np.ndarray, sigma_diag: np.ndarray):
        """ !!! Be careful with dimensions, it can lead to serious problems.
        :param mu: n x 1 dimension
        :param sigma_diag: n x 1 dimension
        :return:
        """
        p = norm.cdf(self.__threshold, mu, np.sqrt(sigma_diag))
        bv = p * (1 - p)
        ibv = np.sum(bv)
        return ibv

    def set_sigma(self, value: float) -> None:
        self.__sigma = value

    def set_lateral_range(self, value: float) -> None:
        self.__lateral_range = value

    def set_nugget(self, value: float) -> None:
        self.__nugget = value

    def set_threshold(self, value: float) -> None:
        self.__threshold = value

    def set_mu(self, value: np.ndarray) -> None:
        self.__mu = value

    def get_sigma(self) -> float:
        return self.__sigma

    def get_lateral_range(self) -> float:
        return self.__lateral_range

    def get_nugget(self) -> float:
        return self.__nugget

    def get_threshold(self) -> float:
        return self.__threshold

    def get_mu(self) -> np.ndarray:
        return self.__mu

    def get_Sigma(self) -> np.ndarray:
        return self.__Sigma

    def get_eibv_field(self):
        return self.__eibv_field

    def get_ivr_field(self):
        return self.__ivr_field


if __name__ == "__main__":
    g = GRF()

