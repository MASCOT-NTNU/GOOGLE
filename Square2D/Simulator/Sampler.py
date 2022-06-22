"""
This script samples the field and returns the updated field
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-18
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
        F = getFVector(self.ind_sample, self.knowledge.xyz.shape[0])
        eibv = get_eibv_1d(self.knowledge.threshold, self.knowledge.mu_cond, self.knowledge.Sigma_cond,
                           F, self.knowledge.R)
        dist = self.getDistanceTravelled()
        y_sampled = self.ground_truth[self.ind_sample]
        self.knowledge.crps.append(np.sum(get_crps_1d(y_sampled,
                                                      self.knowledge.mu_cond,
                                                      self.knowledge.Sigma_cond)))

        self.knowledge.mu_cond, self.knowledge.Sigma_cond = \
            update_GP_field(mu_cond=self.knowledge.mu_cond, Sigma_cond=self.knowledge.Sigma_cond, F=F,
                            R=self.knowledge.R, y_sampled=y_sampled)
        self.knowledge.excursion_prob = get_excursion_prob_1d(self.knowledge.mu_cond,
                                                              self.knowledge.Sigma_cond,
                                                              self.knowledge.threshold)
        self.knowledge.sampling_location_plot.append([self.knowledge.xyz[self.knowledge.ind_now, 0],
                                                      self.knowledge.xyz[self.knowledge.ind_now, 1]])
        self.knowledge.ind_visited.append(self.knowledge.ind_now)
        self.knowledge.ind_prev = self.knowledge.ind_now
        self.knowledge.ind_now = self.ind_sample

        self.knowledge.rmse.append(mean_squared_error(self.ground_truth,
                                                      self.knowledge.mu_cond,
                                                      squared=False))
        self.knowledge.uncertainty.append(np.sum(np.diag(self.knowledge.Sigma_cond)))
        self.knowledge.integrated_bernoulli_variance.append(eibv)
        self.knowledge.distance_travelled.append(dist + self.knowledge.distance_travelled[-1])

    def getDistanceTravelled(self):
        dist_x = self.knowledge.xyz[self.knowledge.ind_now, 0] - self.knowledge.xyz[self.knowledge.ind_prev, 0]
        dist_y = self.knowledge.xyz[self.knowledge.ind_now, 1] - self.knowledge.xyz[self.knowledge.ind_prev, 1]
        dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
        return dist

    @property
    def Knowledge(self):
        return self.knowledge




