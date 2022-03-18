"""
This script generates the next waypoint based on the current knowledge and previous path
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-18
"""

"""
Usage:
lat_next, lon_next = MyopicPlanning_2D(Knowledge, Experience).next_waypoint
"""

from usr_func import *
from GOOGLE.Simulation_2DNidelva.Tree.Location import Location
import time


class MyopicPlanning_2D:

    def __init__(self, knowledge):
        self.knowledge = knowledge
        self.find_next_waypoint()

    def find_next_waypoint(self):
        self.find_candidates_loc()
        self.filter_candidates_loc()
        t1 = time.time()
        id = self.knowledge.ind_cand_filtered
        eibv = []
        for k in range(len(id)):
            F = getFVector(id[k], self.knowledge.coordinates.shape[0])
            eibv.append(get_eibv_1d(self.knowledge.kernel.threshold, self.knowledge.kernel.mu_cond,
                                    self.knowledge.kernel.Sigma_cond, F, self.knowledge.kernel.R))
        t2 = time.time()
        if len(eibv) == 0:  # in case it is in the corner and not found any valid candidate locations
            while True:
                ind_next = self.search_for_new_location()
                if not ind_next in self.knowledge.ind_visited:
                    # print("Found new: ", ind_next)
                    self.knowledge.ind_next = ind_next
                    break
        else:
            self.knowledge.ind_next = self.knowledge.ind_cand_filtered[np.argmin(np.array(eibv))]

    def find_candidates_loc(self):
        delta_x, delta_y = latlon2xy(self.knowledge.coordinates[:, 0], self.knowledge.coordinates[:, 1],
                                     self.knowledge.coordinates[self.knowledge.ind_now, 0],
                                     self.knowledge.coordinates[self.knowledge.ind_now, 1])  # using the distance
        distance_vector = np.sqrt(delta_x ** 2 + delta_y ** 2)
        self.knowledge.ind_cand = np.where((distance_vector <= self.knowledge.distance_neighbour_radar))[0]

    def filter_candidates_loc(self):
        id = []  # ind vector for containing the filtered desired candidate location
        t1 = time.time()
        dx1, dy1 = latlon2xy(self.knowledge.coordinates[self.knowledge.ind_now, 0],
                             self.knowledge.coordinates[self.knowledge.ind_now, 1],
                             self.knowledge.coordinates[self.knowledge.ind_prev, 0],
                             self.knowledge.coordinates[self.knowledge.ind_prev, 1])
        vec1 = vectorise([dx1, dy1])
        for i in range(len(self.knowledge.ind_cand)):
            if self.knowledge.ind_cand[i] != self.knowledge.ind_now:
                if not self.knowledge.ind_cand[i] in self.knowledge.ind_visited:
                    dx2, dy2 = latlon2xy(self.knowledge.coordinates[self.knowledge.ind_cand[i], 0],
                                         self.knowledge.coordinates[self.knowledge.ind_cand[i], 1],
                                         self.knowledge.coordinates[self.knowledge.ind_now, 0],
                                         self.knowledge.coordinates[self.knowledge.ind_now, 1])
                    vec2 = vectorise([dx2, dy2])
                    if np.dot(vec1.T, vec2) >= 0:
                        id.append(self.knowledge.ind_cand[i])
        # print(id)
        id = np.unique(np.array(id))  # filter out repetitive candidate locations
        self.knowledge.ind_cand_filtered = id  # refresh old candidate location
        t2 = time.time()

    def search_for_new_location(self):
        ind_next = np.random.randint(len(self.knowledge.coordinates))
        return ind_next

    @property
    def next_waypoint(self):
        return Location(self.knowledge.coordinates[self.knowledge.ind_next, 0],
                        self.knowledge.coordinates[self.knowledge.ind_next, 1])




