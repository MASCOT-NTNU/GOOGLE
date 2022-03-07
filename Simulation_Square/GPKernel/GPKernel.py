"""
This script builds the kernel for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""


from usr_func import *
from GOOGLE.Simulation_Square.Config.Config import *


class GPKernel:

    def __init__(self):
        self.getGrid()
        self.getMean()
        self.getSigma()
        self.getGroundTruth()

        self.mu_cond = self.mu_prior_vector
        self.Sigma_cond = self.Sigma_prior

    def getGrid(self):
        self.x = np.linspace(XLIM[0], XLIM[1], NX)
        self.y = np.linspace(YLIM[0], YLIM[1], NY)
        self.x_matrix, self.y_matrix = np.meshgrid(self.x, self.y)
        self.x_vector = self.x_matrix.reshape(-1, 1)
        self.y_vector = self.y_matrix.reshape(-1, 1)
        self.grid_vector = np.hstack((self.x_vector, self.y_vector))
        print("Grid is built successfully!")

    def setCoef(self):
        self.sigma = SIGMA
        self.eta = 4.5 / LATERAL_RANGE
        self.tau = NUGGET
        self.R = np.diagflat(self.tau ** 2)
        self.threshold = THRESHOLD
        print("Coef is set successfully!")

    @staticmethod
    def getPrior(x, y):
        return (1 - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .07))
                # 1 - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .05))
                # 1 - np.exp(- ((x - .5) ** 2 + (y - .0) ** 2) / .004) +
                # 1 - np.exp(- ((x - .99) ** 2 + (y - .1) ** 2) / .1))

    def getIndF(self, location):
        x, y = map(vectorise, [location.x, location.y])
        DM_x = x @ np.ones([1, len(self.x_vector)]) - np.ones([len(x), 1]) @ self.x_vector.T
        DM_y = y @ np.ones([1, len(self.y_vector)]) - np.ones([len(y), 1]) @ self.y_vector.T
        DM = DM_x ** 2 + DM_y ** 2
        ind_F = np.argmin(DM, axis = 1)
        return ind_F

    def getMean(self):
        self.mu_prior_vector = vectorise(self.getPrior(self.x_vector, self.y_vector))
        self.mu_prior_matrix = self.getPrior(self.x_matrix, self.y_matrix)

    def getSigma(self):
        self.setCoef()
        DistanceMatrix = cdist(self.grid_vector, self.grid_vector)
        self.Sigma_prior = self.sigma ** 2 * (1 + self.eta * DistanceMatrix) * np.exp(-self.eta * DistanceMatrix)

    def getGroundTruth(self):
        self.mu_truth = (self.mu_prior_vector.reshape(-1, 1) +
                         np.linalg.cholesky(self.Sigma_prior) @
                         np.random.randn(len(self.mu_prior_vector)).reshape(-1, 1))

    @staticmethod
    def GPupd(mu, Sigma, F, R, measurement):
        C = F @ Sigma @ F.T + R
        mu = mu + Sigma @ F.T @ np.linalg.solve(C, (measurement - F @ mu))
        Sigma = Sigma - Sigma @ F.T @ np.linalg.solve(C, F @ Sigma)
        return mu, Sigma

    @staticmethod
    def getEIBV(mu, Sigma, F, R, threshold):
        Sigma_updated = Sigma - Sigma @ F.T @ np.linalg.solve(F @ Sigma @ F.T + R, F @ Sigma)
        Variance = np.diag(Sigma_updated).reshape(-1, 1)
        EIBV = 0
        for i in range(mu.shape[0]):
            EIBV += (mvn.mvnun(-np.inf, threshold, mu[i], Variance[i])[0] -
                     mvn.mvnun(-np.inf, threshold, mu[i], Variance[i])[0] ** 2)
        return EIBV

    @staticmethod
    def getVarianceReduction(Sigma, F, R):
        Reduction = Sigma @ F.T @ np.linalg.solve(F @ Sigma @ F.T + R, F @ Sigma)
        vr = np.sum(np.diag(Reduction))
        return vr

    @staticmethod
    def getGradient(field):
        gradient_x, gradient_y = np.gradient(field)
        gradient_norm = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        return gradient_norm

    def getEIBVField(self):
        self.eibv = []
        for i in range(self.grid_vector.shape[0]):
            F = np.zeros([1, self.grid_vector.shape[0]])
            F[0, i] = True
            self.eibv.append(GPKernel.getEIBV(self.mu_cond, self.Sigma_cond, F, self.R, THRESHOLD))
        self.eibv = normalise(np.array(self.eibv))

    # def getBudgetField(self, goal):
    #     t1 = time.time()
    #     self.budget_field = np.sqrt((self.grid_vector[:, 0] - goal.x) ** 2 +
    #                                 (self.grid_vector[:, 1] - goal.y) ** 2)
    #     t2 = time.time()
    #     print("Budget field time consumed: ", t2 - t1)
    #
    # def getVRField(self):
    #     self.vr = np.zeros_like(self.x_matrix)
    #     t1 = time.time()
    #     for i in range(self.x_matrix.shape[0]):
    #         for j in range(self.x_matrix.shape[1]):
    #             ind_F = self.getIndF(self.x_matrix[i, j], self.y_matrix[i, j])
    #             F = np.zeros([1, self.grid_vector.shape[0]])
    #             F[0, ind_F] = True
    #             self.vr[i, j] = GPKernel.getVarianceReduction(self.Sigma_prior, F, self.R)
    #     self.vr = normalise(self.vr)
    #     t2 = time.time()
    #     print("Time consumed: ", t2 - t1)
    #     # plotf_vector(self.grid_vector, self.vr, "VR")
    #
    # def getGradientField(self):
    #     self.gradient_prior = normalise(self.getGradient(self.mu_prior_matrix))
    #
    # def getTotalCost(self):
    #     self.cost_total = normalise(self.vr + self.gradient_prior)
