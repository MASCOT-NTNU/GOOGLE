"""
This script samples the field and returns the updated field
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05
"""


from usr_func import *
from sklearn.metrics import mean_squared_error


class Sampler:

    def __init__(self, knowledge, ground_truth, ind_sample):
        self.knowledge = knowledge
        self.ground_truth = ground_truth
        self.ind_sample = ind_sample
        self.sample()

    def sample(self):
        F = getFVector(self.ind_sample, self.knowledge.coordinates.shape[0])
        eibv = EIBV_1D(self.knowledge.threshold_salinity, self.knowledge.mu, self.knowledge.Sigma, F, self.knowledge.kernel.R)
        dist = self.getDistanceTravelled()

        self.knowledge.mu, self.knowledge.Sigma = \
            GPupd(mu_cond=self.knowledge.mu, Sigma_cond=self.knowledge.Sigma, F=F,
                  R=self.knowledge.kernel.R, y_sampled=self.ground_truth[self.ind_sample])
        self.knowledge.excursion_prob = get_excursion_prob_1d(self.knowledge.mu, self.knowledge.Sigma, self.knowledge.threshold_salinity)
        self.knowledge.trajectory.append([self.knowledge.coordinates[self.knowledge.ind_now, 0],
                                          self.knowledge.coordinates[self.knowledge.ind_now, 1],
                                          self.knowledge.coordinates[self.knowledge.ind_now, 2]])
        self.knowledge.ind_visited.append(self.knowledge.ind_now)
        self.knowledge.ind_prev = self.knowledge.ind_now
        self.knowledge.ind_now = self.ind_sample

        self.knowledge.rootMeanSquaredError.append(mean_squared_error(self.ground_truth, self.knowledge.mu, squared=False))
        self.knowledge.expectedVariance.append(np.sum(np.diag(self.knowledge.Sigma)))
        self.knowledge.integratedBernoulliVariance.append(eibv)
        self.knowledge.distance_travelled.append(dist + self.knowledge.distance_travelled[-1])

    def getDistanceTravelled(self):
        x_dist, y_dist = latlon2xy(self.knowledge.coordinates[self.knowledge.ind_now, 0],
                                   self.knowledge.coordinates[self.knowledge.ind_now, 1],
                                   self.knowledge.coordinates[self.knowledge.ind_prev, 0],
                                   self.knowledge.coordinates[self.knowledge.ind_prev, 1])
        z_dist = self.knowledge.coordinates[self.knowledge.ind_now, 2] - self.knowledge.coordinates[self.knowledge.ind_prev, 2]
        dist = np.sqrt(x_dist ** 2 + y_dist ** 2 + z_dist ** 2)
        return dist

    @property
    def Knowledge(self):
        return self.knowledge




