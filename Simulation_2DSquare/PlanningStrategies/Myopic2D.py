"""
This script generates the next waypoint based on the current knowledge and previous path
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-24
"""

"""
Usage:
loc_next = MyopicPlanning2D(Knowledge).next_waypoint
"""

from usr_func import *
from GOOGLE.Simulation_2DSquare.Tree.Location import Location
import time


class MyopicPlanning2D:

    def __init__(self, knowledge):
        self.knowledge = knowledge
        self.find_next_waypoint()

    def find_next_waypoint(self):
        self.find_candidates_loc()
        self.filter_candidates_loc()
        t1 = time.time()
        id = self.knowledge.ind_cand_filtered
        cost_from_cost_valley = []
        for k in range(len(id)):
            cost_from_cost_valley.append(self.knowledge.cost_valley[id[k]])
            # F = getFVector(id[k], self.knowledge.grid.shape[0])
            # eibv.append(get_eibv_1d(self.knowledge.threshold, self.knowledge.mu_cond,
            #                         self.knowledge.Sigma_cond, F, self.knowledge.R))
        t2 = time.time()
        if len(cost_from_cost_valley) == 0:  # in case it is in the corner and not found any valid candidate locations
            while True:
                ind_next = self.search_for_new_location()
                if not ind_next in self.knowledge.ind_visited:
                    # print("Found new: ", ind_next)
                    self.knowledge.ind_next = ind_next
                    break
        else:
            self.knowledge.ind_next = self.knowledge.ind_cand_filtered[np.argmin(np.array(cost_from_cost_valley))]

    def find_candidates_loc(self):
        delta_x = self.knowledge.grid[:, 0] - self.knowledge.grid[self.knowledge.ind_now, 0]
        delta_y = self.knowledge.grid[:, 1] - self.knowledge.grid[self.knowledge.ind_now, 1]
        distance_vector = np.sqrt(delta_x ** 2 + delta_y ** 2)
        self.knowledge.ind_cand = np.where((distance_vector <= self.knowledge.distance_neighbour_radar_myopic2d))[0]

    def filter_candidates_loc(self):
        id = []  # ind vector for containing the filtered desired candidate location
        t1 = time.time()
        # dx1 = self.knowledge.grid[self.knowledge.ind_now, 0] - self.knowledge.grid[self.knowledge.ind_prev, 0]
        # dy1 = self.knowledge.grid[self.knowledge.ind_now, 1] - self.knowledge.grid[self.knowledge.ind_prev, 1]
        # vec1 = vectorise([dx1, dy1])
        # for i in range(len(self.knowledge.ind_cand)):
        #     if self.knowledge.ind_cand[i] != self.knowledge.ind_now:
        #         if not self.knowledge.ind_cand[i] in self.knowledge.ind_visited:
        #             dx2 = (self.knowledge.grid[self.knowledge.ind_cand[i], 0] -
        #                    self.knowledge.grid[self.knowledge.ind_now, 0])
        #             dy2 = (self.knowledge.grid[self.knowledge.ind_cand[i], 1] -
        #                    self.knowledge.grid[self.knowledge.ind_now, 1])
        #             vec2 = vectorise([dx2, dy2])
        #             if np.dot(vec1.T, vec2) >= 0:
        #                 id.append(self.knowledge.ind_cand[i])
        # print(id)
        id = self.knowledge.ind_cand
        id = np.unique(np.array(id))  # filter out repetitive candidate locations
        self.knowledge.ind_cand_filtered = id  # refresh old candidate location
        t2 = time.time()

    def search_for_new_location(self):
        ind_next = np.random.randint(len(self.knowledge.grid))
        return ind_next

    @property
    def next_waypoint(self):
        return Location(self.knowledge.grid[self.knowledge.ind_next, 0],
                        self.knowledge.grid[self.knowledge.ind_next, 1])




