"""
This script only contains the knowledge node for 3d cube case
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-15
"""


class Knowledge:

    def __init__(self, starting_location=None, ending_location=None, goal_location=None, goal_sample_rate=None,
                 step_size=None, budget=None, kernel=None):
        self.starting_location = starting_location
        self.ending_location = ending_location
        self.goal_location = goal_location
        self.goal_sample_rate = goal_sample_rate
        self.step_size = step_size

        self.kernel = kernel
        self.budget = budget


