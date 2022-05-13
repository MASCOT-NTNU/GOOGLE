"""
This script tests the rrt* algorithm for collision avoidance
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-22
"""

from usr_func import *
FIGPATH = "/Users/yaoling/OneDrive - NTNU/Self-improvements/LearnedAlgorithms/pathplanning/fig/rrt_star/"


NUM_STEPS = 50
BUDGET = 7


if __name__ == "__main__":
    # starting_loc = Location(.0, .0)
    # ending_loc = Location(.0, 1.)
    # gp = GPKernel()
    # rrtconfig = RRTConfig(starting_location=starting_loc, ending_location=ending_loc, goal_sample_rate=GOAL_SAMPLE_RATE,
    #                       step=STEP, GPKernel=gp)
    # rrt = RRTStar(rrtconfig)
    # rrt.set_obstacles()
    # rrt.expand_trees()
    # rrt.get_shortest_path()
    # rrt.plot_tree()
    # g = GOOGLE()
    # g.pathplanner()
    x = np.arange(12)
    print(normalise(x, .5, 1))

    pass


# #%%
# cmap = get_cmap("RdBu", 10)
#
# plt.scatter(g.gp.grid_vector[:, 0], g.gp.grid_vector[:, 1], c=g.gp.budget_field, cmap=cmap)
# plt.colorbar()
# plt.show()


