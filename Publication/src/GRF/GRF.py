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
from scipy.stats import norm, multivariate_normal
from usr_func.normalize import normalize
from usr_func.calculate_analytical_ebv import calculate_analytical_ebv
from joblib import Parallel, delayed
import time
import pandas as pd


class GRF:
    """
    GRF kernel
    """
    def __init__(self, sigma: float = 1., nugget: float = .4, approximate_eibv: bool = True, parallel_eibv: bool = False) -> None:
        """ Initializes the parameters in GRF kernel. """
        self.__approximate_eibv = approximate_eibv
        self.__parallel_eibv = parallel_eibv

        self.__ar1_coef = .965  # AR1 coef, timestep is 10 mins.
        self.__ar1_corr = 600   # [sec], AR1 correlation time range.

        """ Empirical parameters """
        # spatial variability
        self.__sigma = sigma

        # spatial correlation
        self.__lateral_range = 700  # 680 in the experiment

        # measurement noise
        self.__nugget = nugget

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
        self.__mu_prior = self.__mu
        self.__Sigma_prior = self.__Sigma

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
        dm_grid_sinmod = cdist(self.grid, grid_sinmod)
        ind_close = np.argmin(dm_grid_sinmod, axis=1)
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
        t_steps = int((t_end - t_start) // self.__ar1_corr)
        # t1 = time.time()
        xd = dataset[:, 1].reshape(-1, 1)
        yd = dataset[:, 2].reshape(-1, 1)
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
        self.__update_temporal(ind_measured=ind_assimilated, salinity_measured=salinity_assimilated, timestep=t_steps)
        # t2 = time.time()
        # print("Data assimilation takes: ", t2 - t1, " seconds")
        """ Just for debugging. """
        return ind_assimilated, salinity_assimilated

    def __update_temporal(self, ind_measured: np.ndarray, salinity_measured: np.ndarray, timestep=0):
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

        t1 = time.time()
        # propagate
        mt0 = self.__mu_prior + self.__ar1_coef * (self.__mu - self.__mu_prior)
        St0 = self.__ar1_coef ** 2 * self.__Sigma + (1 - self.__ar1_coef ** 2) * self.__Sigma_prior
        mts = mt0
        Sts = St0
        for s in range(timestep):
            mts = self.__mu_prior + self.__ar1_coef * (mts - self.__mu_prior)
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
                if self.__parallel_eibv:
                    eibv_field[i] = self.__get_eibv_analytical_para(self.__mu, sigma_diag, vr_diag)
                else:
                    eibv_field[i] = self.__get_eibv_analytical(self.__mu, sigma_diag, vr_diag)
            ivr_field[i] = np.sum(np.diag(VR))
        self.__eibv_field = normalize(eibv_field)
        self.__ivr_field = 1 - normalize(ivr_field)
        t2 = time.time()
        print("Approximate: ", self.__approximate_eibv, "; Total EI field takes: ", t2 - t1, " seconds.")
        return self.__eibv_field, self.__ivr_field

    # def get_ei_at_locations(self, locs: np.ndarray) -> tuple:
    #     """ Get EI values at given locations. """
    #     ind = self.field.get_ind_from_location(locs)
    #     N = len(ind)
    #     t1 = time.time()
    #     eibv = np.zeros(N)
    #     ivr = np.zeros(N)
    #     for i in range(N):
    #         id = ind[i]
    #         SF = self.__Sigma[:, id].reshape(-1, 1)
    #         MD = 1 / (self.__Sigma[id, id] + self.__nugget)
    #         VR = SF @ SF.T * MD
    #         SP = self.__Sigma - VR
    #         sigma_diag = np.diag(SP).reshape(-1, 1)
    #         eibv[i] = self.__get_ibv(self.__mu, sigma_diag)
    #         ivr[i] = np.sum(np.diag(VR))
    #     self.__eibv_field = normalize(eibv)
    #     self.__ivr_field = 1 - normalize(ivr)
    #     t2 = time.time()
    #     print("Calcuating EI at given locations takes: ", t2 - t1, " seconds.")
    #     return self.__eibv_field, self.__ivr_field

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

    def get_eibv_analytical_fast(self) -> None:
        """
        Calculate the eibv using the analytical formula with a bivariate cumulative dentisty function.
        """

        pass

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
