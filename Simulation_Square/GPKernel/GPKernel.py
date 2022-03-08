"""
This script builds the kernel for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""
import numpy as np

from usr_func import *
from GOOGLE.Simulation_Square.Config.Config import *
from GOOGLE.Simulation_Square.Tree.Location import Location


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
        self.grid_vector = []
        for i in range(self.x_matrix.shape[0]):
            for j in range(self.x_matrix.shape[1]):
                self.grid_vector.append([self.x_matrix[i, j], self.y_matrix[i, j]])
        self.grid_vector = np.array(self.grid_vector)
        self.x_vector = self.grid_vector[:, 0].reshape(-1, 1)
        self.y_vector = self.grid_vector[:, 1].reshape(-1, 1)
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

    def getEIBVField(self):
        self.eibv = []
        for i in range(self.grid_vector.shape[0]):
            F = np.zeros([1, self.grid_vector.shape[0]])
            F[0, i] = True
            self.eibv.append(GPKernel.getEIBV(self.mu_cond, self.Sigma_cond, F, self.R, THRESHOLD))
        self.eibv = normalise(np.array(self.eibv))

    def getBudgetField(self, start_loc, end_loc, budget):
        t1 = time.time()
        if budget >= BUDGET_MARGIN:
            self.budget_middle_location = self.get_middle_location(start_loc, end_loc)
            self.budget_ellipse_angle = self.get_angle_between_locations(start_loc, end_loc)

            self.budget_ellipse_a = budget / 2
            self.budget_ellipse_c = np.sqrt((start_loc.x - end_loc.x) ** 2 +
                                            (start_loc.y - end_loc.y) ** 2) / 2
            self.budget_ellipse_b = np.sqrt(self.budget_ellipse_a ** 2 - self.budget_ellipse_c ** 2)
            print("a: ", self.budget_ellipse_a, "b: ", self.budget_ellipse_b, "c: ", self.budget_ellipse_c)

            x_wgs = self.x_vector - self.budget_middle_location.x
            y_wgs = self.y_vector - self.budget_middle_location.y
            self.penalty_budget = []
            for i in range(len(self.grid_vector)):
                x_usr = (x_wgs[i] * np.cos(self.budget_ellipse_angle) +
                         y_wgs[i] * np.sin(self.budget_ellipse_angle))
                y_usr = (- x_wgs[i] * np.sin(self.budget_ellipse_angle) +
                         y_wgs[i] * np.cos(self.budget_ellipse_angle))

                if (x_usr / self.budget_ellipse_a) ** 2 + (y_usr / self.budget_ellipse_b) ** 2 <= 1:
                    self.penalty_budget.append(0)
                else:
                    self.penalty_budget.append(np.inf)
            self.penalty_budget = np.array(self.penalty_budget)
        else:
            self.penalty_budget = np.ones_like(self.grid_vector[:, 0]) * np.inf
            ind_end_loc = self.getIndF(end_loc)
            self.penalty_budget[ind_end_loc] = 0

        t2 = time.time()
        print("budget field consumed: ", t2 - t1)

    @staticmethod
    def get_middle_location(location1, location2):
        x_middle = (location1.x + location2.x) / 2
        y_middle = (location1.y + location2.y) / 2
        return Location(x_middle, y_middle)

    @staticmethod
    def get_angle_between_locations(location1, location2):
        delta_y = location2.y - location1.y
        delta_x = location2.x - location1.x
        angle = np.math.atan2(delta_y, delta_x)
        return angle

    def getGradientField(self):
        self.mu_cond_matrix = np.zeros_like(self.x_matrix)
        for i in range(self.x_matrix.shape[0]):
            for j in range(self.x_matrix.shape[1]):
                self.mu_cond_matrix[i, j] = self.mu_cond[i*NY+j]
        gradient_x, gradient_y = np.gradient(self.mu_cond_matrix)
        self.gradient_matrix = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        self.gradient_vector = []
        for i in range(self.gradient_matrix.shape[0]):
            for j in range(self.gradient_matrix.shape[1]):
                self.gradient_vector.append(self.gradient_matrix[i, j])
        self.gradient_vector = np.array(self.gradient_vector)

    def getVarianceReductionField(self):
        self.vr = []
        for i in range(len(self.grid_vector)):
            ind_F = self.getIndF(Location(self.grid_vector[i, 0], self.grid_vector[i, 1]))
            F = np.zeros([1, self.grid_vector.shape[0]])
            F[0, ind_F] = True
            self.vr.append(self.getVarianceReduction(self.Sigma_cond, F, self.R))
        self.vr = 1 - normalise(self.vr)

    @staticmethod
    def getVarianceReduction(Sigma, F, R):
        Reduction = Sigma @ F.T @ np.linalg.solve(F @ Sigma @ F.T + R, F @ Sigma)
        vr = np.sum(np.diag(Reduction))
        return vr


