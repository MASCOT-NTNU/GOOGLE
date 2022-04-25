"""
This script builds the cost valley
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-25
"""


from GOOGLE.Simulation_2DNidelva.Config.Config import FILEPATH
import pandas as pd


class CostValley:


    def __init__(self, mu=None, Sigma=None):
        self.load_grf_grid()
        self.mu = mu
        self.Sigma = Sigma
        pass

    def load_grf_grid(self):
        self.grf_grid = pd.read_csv(FILEPATH+"Config/GRFGrid.csv").to_numpy()
        print("CV1: GRF Grid is loaded successfullyQ")

    def get_exploration_exploitation_field(self):
        t1 = time.time()
        self.cost_eibv = []
        for i in range(self.coordinates_xyz.shape[0]):
            F = getFVector(i, self.coordinates_xyz.shape[0])
            self.cost_eibv.append(get_eibv_1d(self.knowledge.threshold, self.knowledge.mu_cond,
                                              self.knowledge.Sigma_cond, F, self.R))
        self.cost_eibv = normalise(np.array(self.cost_eibv))
        t2 = time.time()
        print("EIBV field takes: ", t2 - t1)

        t1 = time.time()
        self.cost_vr = []
        for i in range(len(self.knowledge.coordinates_xy)):
            ind_F = get_ind_at_location2d_xy(self.knowledge.coordinates_xy,
                                             LocationXY(self.knowledge.coordinates_xy[i, 0],
                                                        self.knowledge.coordinates_xy[i, 1]))
            F = np.zeros([1, self.coordinates_xyz.shape[0]])
            F[0, ind_F] = True
            self.cost_vr.append(self.get_variance_reduction(self.knowledge.Sigma_cond, F, self.R))
        self.cost_vr = 1 - normalise(np.array(self.cost_vr))
        t2 = time.time()
        print("Variance Reduction field takes: ", t2 - t1)

        pass




