"""
This script only contains the knowledge node
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""


class Knowledge:

    def __init__(self, starting_location=None, ending_location=None, goal_sample_rate=None,
                 iterations=None, budget=None, mu=None, Sigma=None, F=None, EIBV=None):
        self.starting_location = starting_location
        self.ending_location = ending_location
        self.goal_sample_rate = goal_sample_rate
        self.step = iterations

        self.budget = budget

        self.mu = mu
        self.Sigma = Sigma
        self.F = F
        self.EIBV = EIBV


