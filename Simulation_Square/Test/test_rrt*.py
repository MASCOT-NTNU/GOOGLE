"""
This script tests rrt*
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""


from GOOGLE.Simulation_Square.Plotting.plotting_func import *
from GOOGLE.Simulation_Square.GPKernel.GPKernel import *
from GOOGLE.Simulation_Square.Config.Config import *
from GOOGLE.Simulation_Square.PlanningStrategies.RRTStar import RRTStar
from GOOGLE.Simulation_Square.Tree.Knowledge import Knowledge
from GOOGLE.Simulation_Square.Tree.Location import Location


class PathPlanner:

    trajectory = []

    def __init__(self, starting_location=None, target_location=None):
        # self.gp = GPKernel()
        # self.gp.getEIBVField()
        self.starting_location = starting_location
        self.target_location = target_location
        self.knowledge = Knowledge()
        # self.knowledge = Knowledge(self.gp.mu_prior_vector, self.gp.Sigma_prior)
        pass

    def run(self):
        knowledge = Knowledge(starting_location=self.starting_location, ending_location=self.target_location,
                              goal_sample_rate=GOAL_SAMPLE_RATE, step=STEP, knowledge=self.knowledge)
        self.rrt = RRTStar(rrtconfig)
        self.rrt.expand_trees()
        self.rrt.get_shortest_path()
        self.rrt.plot_tree()
        plt.show()


if __name__ == "__main__":
    starting_loc = Location(.0, .0)
    ending_loc = Location(1., 1.)
    p = PathPlanner(starting_loc, ending_loc)
    p.run()
