"""
Gaussian Random Field module handles the data assimilation and the cost valley computation.

Objectives:
    1. Construct the Gaussian Random Field (GRF) kernel.
    2. Update the prior mean and covariance matrix.
    3. Compute the cost valley fields.
    4. Assimilate in-situ data.

Methodology:
    1. Construct the GRF kernel.
        1.1. Construct the distance matrix using
            .. math::
                d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
        1.2. Construct the covariance matrix.
            .. math::
                \Sigma_{ij} = \sigma^2 (1 + \eta d_{ij}) \exp(-\eta d_{ij})
    2. Update the prior mean and covariance matrix.
        2.1. Update the prior mean.
        2.2. Update the prior covariance matrix.
    3. Compute the cost valley fields.
        3.1. Compute the expected improvement of best value (EIBV).
        3.2. Compute the inverse of the variance reduction (IVR).
        3.3. Compute the cost valley fields by summing weighted EIBV and weighted IVR fields.

"""

from Field import Field
from usr_func.vectorize import vectorize
from usr_func.checkfolder import checkfolder
from scipy.spatial.distance import cdist
import numpy as np
from scipy.stats import norm, multivariate_normal
from usr_func.normalize import normalize
from joblib import Parallel, delayed
import time
import pandas as pd


class GRF:
    def __init__(self, sigma: float = 1., nugget: float = .4, approximate_eibv: bool = True) -> None:
        """
        Initializes the parameters in GRF kernel.

        Args:
            sigma: float, spatial variability.
            nugget: float, measurement noise.
            approximate_eibv: bool, whether to use the approximate EIBV.

        Attributes:
            __approximate_eibv: bool, whether to use the approximate EIBV.
            __ar1_coef: float, AR1 coefficient.
            __ar1_corr: float, AR1 correlation time range in seconds.
            __sigma: float, spatial variability.
            __lateral_range: float, spatial correlation.
            __nugget: float, measurement noise.
            __threshold: float, threshold used for calculating EIBV.
            __eta: float, decay factor.
            __tau: float, sqrt of nugget.
            __mu: np.array, kernel mean.
            __Sigma: np.array, kernel covariance matrix.
            __eibv_field: np.array, expected integrated bernoulli variance field.
            __ivr_field: np.array, integrated variance reduction field.
            __mu_prior: np.array, prior mean.
            __Sigma_prior: np.array, prior covariance matrix.
            field: Field, field object.
            grid: np.array, grid of the field.
            Ngrid: int, number of grid points.
            __Fgrf: np.array, grf kernel.
            __xg: np.array, x coordinates of the grid.
            __yg: np.array, y coordinates of the grid.
            __distance_matrix: np.array, distance matrix between grid points.

        """
        self.__approximate_eibv = approximate_eibv

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
        """
        Construct distance matrix and thus Covariance matrix for the kernel.

        Methodology:
            1. Construct the distance matrix using
                .. math::
                    d_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}
            2. Construct the covariance matrix.
                .. math::
                    \Sigma_{ij} = \sigma^2 (1 + \eta d_{ij}) \exp(-\eta d_{ij})

        """
        self.__distance_matrix = cdist(self.grid, self.grid)
        self.__Sigma = self.__sigma ** 2 * ((1 + self.__eta * self.__distance_matrix) *
                                            np.exp(-self.__eta * self.__distance_matrix))

    def __construct_prior_mean(self) -> None:
        """
        Construct prior mean for the kernel.

        Methodology:
            1. Construct the prior mean using the SINMOD dataset.
            2. Interpolate the prior mean onto the grid.

        Returns:
            None

        """
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

        Args:
            dataset: np.array([x, y, sal])

        Methodology:
            1. Construct the distance matrix between gmrf grid and dataset grid.
            2. Average the values to each cell.
            3. Update the kernel mean and covariance matrix.

        Returns:
            None

        Examples:
            >>> dataset = np.array([[0, 0, 0], [1, 1, 1]])
            >>> grf = GRF()
            >>> grf.assimilate_data(dataset)
            >>> grf.get_mu()

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

    def __update(self, ind_measured: np.ndarray, salinity_measured: np.ndarray) -> None:
        """
        Update GRF kernel based on sampled data.

        Args:
            ind_measured: indices where the data is assimilated.
            salinity_measured: measurements at sampeld locations, dimension: m x 1

        Methodology:
            1. Loop through each measurement and construct the measurement matrix F.
            2. Construct the measurement noise matrix R.
            3. Update the kernel mean and covariance matrix using
                .. math::
                    \mu = \mu + \Sigma F^T (F \Sigma F^T + R)^{-1} (y - F \mu)
                    \Sigma = \Sigma - \Sigma F^T (F \Sigma F^T + R)^{-1} F \Sigma

        Returns:
            None

        Examples:
            >>> ind_measured = np.array([0, 1])
            >>> salinity_measured = np.array([0, 1])
            >>> grf = GRF()
            >>> grf.__update(ind_measured, salinity_measured)
            >>> grf.get_mu()

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

        Args:
            dataset: np.array([timestamp, x, y, sal])

        Methodology:
            1. Construct the distance matrix between grf grid and dataset grid.
            2. Average the values to each cell.
            3. Update the kernel mean and covariance matrix.

        Returns:
            (ind, salinity) for visualising eda plots.

        Examples:
            >>> dataset = np.array([[0, 0, 0, 0], [1, 1, 1, 1]])
            >>> grf = GRF()
            >>> grf.assimilate_temporal_data(dataset)
            >>> grf.get_mu()

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

    def __update_temporal(self, ind_measured: np.ndarray, salinity_measured: np.ndarray, timestep=0) -> None:
        """
        Update GRF kernel based on sampled data.

        Args:
            ind_measured: indices where the data is assimilated.
            salinity_measured: measurements at sampeld locations, dimension: m x 1
            timestep: timestep of the data assimilation, default is 0, which means that the data assimilation is
                performed at the same time step as the model.

        Methodology:
            1. Loop through each measurement and construct the measurement matrix F.
            2. Construct the measurement noise matrix R.
            3. Update the kernel mean and covariance matrix using
                .. math::
                    \mu_{t|t-1} = \mu + \rho * (\mu_{t-1|t-1} - \mu)
                    \Sigma_{t|t-1} = \rho^2 * \Sigma_{t-1|t-1} + (1 - \rho^2) * \Sigma
                    G_t = \Sigma_{t|t-1} F^T (F \Sigma_{t|t-1} F^T + R)^{-1}
                    \mu_{t|t} = \mu_{t|t-1} + G_t (y - F \mu_{t|t-1})
                    \Sigma_{t|t} = \Sigma_{t|t-1} - G_t F \Sigma_{t|t-1}

        Returns:
            None

        References:
            [1] https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf

        Examples:
            >>> ind_measured = np.array([0, 1])
            >>> salinity_measured = np.array([0, 1])
            >>> grf = GRF()
            >>> grf.__update_temporal(ind_measured, salinity_measured)
            >>> grf.get_mu()

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
        """
        Get expected information field.

        Methodology:
            1. Loop through each grid point and calculate the EIBV and IVR fields.
            2. Calculate the EI field using
                .. math::
                    EI = weight * EIBV + (1 - weight) * IVR

        Returns:
            eibv_field: expected information based on variance reduction, dimension: Ngrid x 1

        Examples:
            >>> grf = GRF()
            >>> eibv_field = grf.get_ei_field()

        """
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
                eibv_field[i] = self.__get_ibv(self.__mu, sigma_diag)
            else:
                vr_diag = np.diag(VR).reshape(-1, 1)
                eibv_field[i] = self.__get_eibv(self.__mu, sigma_diag, vr_diag)
            ivr_field[i] = np.sum(np.diag(VR))
        self.__eibv_field = normalize(eibv_field)
        self.__ivr_field = 1 - normalize(ivr_field)
        t2 = time.time()
        # print("Total EI field takes: ", t2 - t1, " seconds.")
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

    def __get_ibv(self, mu: np.ndarray, sigma_diag: np.ndarray) -> np.ndarray:
        """
        Calculate the ibv using the approximate formula with a univariate cumulative dentisty function.

        Args:
            mu: n x 1 dimension
            sigma_diag: n x 1 dimension

        Methodology:
            1. Calculate the probability of exceedance of the threshold.
            2. Calculate ibv by summing up the product of probability of exceedance and (1 - probability of exceedance).

        Returns:
            ibv: information based on variance reduction, dimension: n x 1

        Examples:
            >>> grf = GRF()
            >>> ibv = grf.__get_ibv(grf.__mu, grf.__sigma_diag)

        """
        p = norm.cdf(self.__threshold, mu, np.sqrt(sigma_diag))
        bv = p * (1 - p)
        ibv = np.sum(bv)
        return ibv

    def __get_eibv(self, mu: np.ndarray, sigma_diag: np.ndarray, vr_diag: np.ndarray) -> float:
        """
        Calculate the eibv using the analytical formula with a bivariate cumulative dentisty function.

        Args:
            mu: n x 1 dimension
            sigma_diag: n x 1 dimension
            vr_diag: n x 1 dimension

        Methodology:
            1. Calculate the probability of exceedance of the threshold using a bivariate cumulative dentisty function.
                .. math::
                    p = \Phi\left(\frac{\theta - \mu}{\sigma}\right) - \Phi\left(\frac{\theta - \mu}{\sigma}\right) \Phi\left(\frac{\theta - \mu}{\sigma}\right)

        Returns:
            eibv: information based on variance reduction, dimension: n x 1

        Examples:
            >>> grf = GRF()
            >>> eibv = grf.__get_eibv(grf.__mu, grf.__sigma_diag, grf.__vr_diag)

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

    def set_sigma(self, value: float) -> None:
        """
        Set space variability.

        Args:
            value: space variability

        Examples:
            >>> grf = GRF()
            >>> grf.set_sigma(0.1)

        """
        self.__sigma = value

    def set_lateral_range(self, value: float) -> None:
        """
        Set lateral range.

        Args:
            value: lateral range

        Examples:
            >>> grf = GRF()
            >>> grf.set_lateral_range(0.1)

        """
        self.__lateral_range = value

    def set_nugget(self, value: float) -> None:
        """
        Set nugget.

        Args:
            value: nugget

        Examples:
            >>> grf = GRF()
            >>> grf.set_nugget(0.1)

        """
        self.__nugget = value

    def set_threshold(self, value: float) -> None:
        """
        Set threshold.

        Args:
            value: threshold

        Examples:
            >>> grf = GRF()
            >>> grf.set_threshold(0.1)

        """
        self.__threshold = value

    def set_mu(self, value: np.ndarray) -> None:
        """
        Set mean of the field.

        Args:
            value: mean of the field

        Examples:
            >>> grf = GRF()
            >>> grf.set_mu(np.array([0.1, 0.2, 0.3]))

        """
        self.__mu = value

    def get_sigma(self) -> float:
        """
        Return variability of the field.

        Returns:
            sigma: space variability

        Examples:
            >>> grf = GRF()
            >>> grf.get_sigma()
            1.0
        """
        return self.__sigma

    def get_lateral_range(self) -> float:
        """
        Return lateral range.

        Returns:
            lateral_range: lateral range

        Examples:
            >>> grf = GRF()
            >>> grf.get_lateral_range()
            600.0
        """
        return self.__lateral_range

    def get_nugget(self) -> float:
        """
        Return nugget of the field.

        Returns:
            nugget: nugget

        Examples:
            >>> grf = GRF()
            >>> grf.get_nugget()
            0.0

        """
        return self.__nugget

    def get_threshold(self) -> float:
        """
        Return threshold.

        Returns:
            threshold: threshold

        Examples:
            >>> grf = GRF()
            >>> grf.get_threshold()
            27.0

        """
        return self.__threshold

    def get_mu(self) -> np.ndarray:
        """
        Return mean vector.

        Returns:
            mu: mean vector

        Examples:
            >>> grf = GRF()
            >>> grf.get_mu()
            array([0.1, 0.2, 0.3])

        """
        return self.__mu

    def get_covariance_matrix(self) -> np.ndarray:
        """
        Return Covariance.

        Returns:
            Sigma: Covariance matrix

        Examples:
            >>> grf = GRF()
            >>> grf.get_covariance_matrix()
            array([[1.00000000e+00, 9.99999998e-01, 9.99999994e-01],
                   [9.99999998e-01, 1.00000000e+00, 9.99999998e-01],
                   [9.99999994e-01, 9.99999998e-01, 1.00000000e+00]])

        """
        return self.__Sigma

    def get_eibv_field(self) -> np.ndarray:
        """
        Return the computed eibv field, given which method to be called.

        Returns:
            eibv_field: eibv field

        Examples:
            >>> grf = GRF()
            >>> grf.get_eibv_field()
            array([0.1, 0.2, 0.3])

        """
        return self.__eibv_field

    def get_ivr_field(self) -> np.ndarray:
        """
        Return the computed ivr field, given which method to be called.

        Returns:
            ivr_field: ivr field

        Examples:
            >>> grf = GRF()
            >>> grf.get_ivr_field()
            array([0.1, 0.2, 0.3])

        """
        return self.__ivr_field


if __name__ == "__main__":
    g = GRF()
