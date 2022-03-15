"""
This script tests rrt*
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""


# from GOOGLE.Simulation_3DCube.Plotting.plotting_func import *
from GOOGLE.Simulation_3DCube.GPKernel.GPKernel import *
from GOOGLE.Simulation_3DCube.Config.Config import *
from GOOGLE.Simulation_3DCube.PlanningStrategies.RRTStar import RRTStar
from GOOGLE.Simulation_3DCube.Tree.Knowledge import Knowledge
from GOOGLE.Simulation_3DCube.Tree.Location import *

BUDGET = 1


class PathPlanner:

    trajectory = []

    def __init__(self, starting_location=None, ending_location=None):
        # self.gp = GPKernel()
        # self.gp.get_eibv_field()
        self.starting_location = starting_location
        self.ending_location = ending_location
        # self.knowledge = Knowledge()
        # self.knowledge = Knowledge(self.gp.mu_prior_vector, self.gp.Sigma_prior)
        pass

    def run(self):
        knowledge = Knowledge(starting_location=self.starting_location, ending_location=self.ending_location,
                              goal_location=None, goal_sample_rate=GOAL_SAMPLE_RATE, step_size=STEPSIZE, budget=BUDGET)
        self.rrt = RRTStar(knowledge)
        self.rrt.maxiter = 3000
        self.rrt.expand_trees()
        self.rrt.get_shortest_trajectory()
        self.rrt.plot_tree()
        # plt.show()


if __name__ == "__main__":
    starting_loc = Location(.01, .01, .01)
    ending_loc = Location(.99, .99, .99)
    p = PathPlanner(starting_loc, ending_loc)
    p.run()
#%%

plt.figure()
ellipse = Ellipse(xy=(0, 0), width=2, height=1,
                  angle=180, edgecolor='r', fc='None', lw=2)
plt.gca().add_patch(ellipse)
plt.xlim(-2, 2)
plt.ylim(-2, 2)
plt.show()


