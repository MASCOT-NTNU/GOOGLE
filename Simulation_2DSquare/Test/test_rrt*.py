"""
This script tests rrt*
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""
import matplotlib.pyplot as plt

from GOOGLE.Simulation_2DSquare.Plotting.plotting_func import *
from GOOGLE.Simulation_2DSquare.GPKernel.GPKernel import *
from GOOGLE.Simulation_2DSquare.Config.Config import *
from GOOGLE.Simulation_2DSquare.PlanningStrategies.RRTStar import RRTStar
from GOOGLE.Simulation_2DSquare.Tree.Knowledge import Knowledge
from GOOGLE.Simulation_2DSquare.Tree.Location import Location

BUDGET = 5


class PathPlanner:

    trajectory = []

    def __init__(self, starting_location=None, ending_location=None):
        self.grid = pd.read_csv(FILEPATH + "PreConfig/WaypointGraph/WaypointGraph.csv").to_numpy()
        self.mu_prior = vectorise(pd.read_csv(FILEPATH + "PreConfig/Data/mu_prior.csv")['mu_prior'].to_numpy())
        self.starting_location = starting_location
        self.ending_location = ending_location
        self.knowledge = Knowledge(grid=self.grid,
                                   starting_location=self.starting_location, ending_location=self.ending_location,
                                   polygon_border=np.array(BORDER), polygon_obstacles=np.array(OBSTACLES),
                                   goal_sample_rate=GOAL_SAMPLE_RATE, step_size=STEPSIZE, budget=BUDGET,
                                   distance_neighbour_radar=DISTANCE_NEIGHBOUR_RADAR, threshold=THRESHOLD)
        self.knowledge.mu_prior = self.mu_prior
        self.gp = GPKernel(self.knowledge)
        self.gp.get_cost_valley(self.starting_location, self.starting_location, self.ending_location, BUDGET)
        pass

    def run(self):
        self.rrt = RRTStar(self.knowledge)
        t1 = time.time()
        self.rrt.expand_trees()
        self.rrt.get_shortest_trajectory()
        t2 = time.time()
        print("Path planning takes: ", t2 - t1)
        self.rrt.plot_tree()
        plt.scatter(self.knowledge.grid[:, 0], self.knowledge.grid[:, 1], c=self.knowledge.cost_valley, cmap=CMAP,
                    vmin=0, vmax=4)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    starting_loc = Location(.75, .1)
    ending_loc = Location(.1, .75)
    p = PathPlanner(starting_loc, ending_loc)
    p.run()

#%%

plt.scatter(p.knowledge.grid[:, 0], p.knowledge.grid[:, 1], c=p.knowledge.cost_, cmap=CMAP)
plt.colorbar()
plt.show()
