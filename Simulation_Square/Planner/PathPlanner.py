"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""

from usr_func import *
from GOOGLE.Simulation_Square.Plotting.plotting_func import *
from GOOGLE.Simulation_Square.GPKernel.GPKernel import *
from GOOGLE.Simulation_Square.Tree.Location import Location
from GOOGLE.Simulation_Square.Config.Config import *
from GOOGLE.Simulation_Square.Tree.Knowledge import Knowledge
from GOOGLE.Simulation_Square.PlanningStrategies.RRTStar import RRTStar

np.random.seed(0)
BUDGET = 2.5
NUM_STEPS = 50
FIGPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/fig/rrt_star/"


class PathPlanner:

    trajectory = []

    def __init__(self, starting_location=None, target_location=None):
        self.gp = GPKernel()
        self.gp.getEIBVField()
        self.starting_location = starting_location
        self.target_location = target_location

    def run(self):

        # plotf_vector(self.gp.grid_vector, self.gp.mu_truth, "Truth")
        # plt.show()
        #
        # plotf_vector(self.gp.grid_vector, self.gp.mu_prior_vector, "Prior")
        # plt.show()

        budget = BUDGET
        distance_travelled = 0
        current_loc = self.starting_location
        end_loc = self.target_location
        self.trajectory.append(current_loc)


        ind_min_cost = np.argmin(self.gp.eibv)
        ending_loc = Location(self.gp.grid_vector[ind_min_cost, 0], self.gp.grid_vector[ind_min_cost, 1])

        for i in range(NUM_STEPS):
            print("Step: ", i)
            # print("Total budget: ", budget)

            # == path planning ==
            t1 = time.time()
            knowledge = Knowledge(starting_location=current_loc, ending_location=end_loc,
                                  goal_location=self.target_location, goal_sample_rate=GOAL_SAMPLE_RATE,
                                  step_size=STEPSIZE, budget=budget, kernel=self.gp, mu=self.gp.mu_cond,
                                  Sigma=self.gp.Sigma_cond)

            self.rrtstar = RRTStar(knowledge)
            self.rrtstar.expand_trees()
            self.rrtstar.get_shortest_trajectory()
            path = self.rrtstar.trajectory
            t2 = time.time()
            print("Path planning takes: ", t2 - t1)

            # == plotting ==
            fig = plt.figure(figsize=(25, 5))
            gs = GridSpec(nrows=1, ncols=4)
            ax = fig.add_subplot(gs[0])
            cmap = get_cmap("RdBu", 10)
            plotf_vector(self.gp.grid_vector, self.gp.mu_truth, "Ground Truth", cmap=cmap)

            ax = fig.add_subplot(gs[1])
            plotf_vector(self.gp.grid_vector, self.gp.mu_cond, "Conditional Mean", cmap=cmap)
            plotf_trajectory(self.trajectory)

            ax = fig.add_subplot(gs[2])
            plotf_vector(self.gp.grid_vector, np.sqrt(np.diag(self.gp.Sigma_cond)), "Prediction Error", cmap=cmap)
            plotf_trajectory(self.trajectory)

            ax = fig.add_subplot(gs[3])
            self.rrtstar.plot_tree()
            plotf_vector(self.gp.grid_vector, self.gp.eibv, "EIBV cost valley ", alpha=.1, cmap=cmap)
            # plotf_budget_radar([goal_loc.x, goal_loc.y], budget)
            # plt.show()
            plt.savefig(FIGPATH + "P_{:03d}.png".format(i))
            plt.close("all")
            # == end plotting ==

            next_loc = Location(path[-2, 0], path[-2, 1])
            discrepancy = np.sqrt((current_loc.x - self.target_location.x) ** 2 +
                                  (current_loc.y - self.target_location.y) ** 2)
            if discrepancy <= DISTANCE_TOLERANCE:
                print("Arrived")
                break

            self.trajectory.append(next_loc)
            distance_travelled += np.sqrt((current_loc.x - next_loc.x) ** 2 +
                                          (current_loc.y - next_loc.y) ** 2)
            current_loc = next_loc

            budget = BUDGET - distance_travelled
            print("Budget left: ", budget)
            print("Distance travelled: ", distance_travelled)

            ind_F = self.gp.getIndF(current_loc)
            F = np.zeros([1, self.gp.grid_vector.shape[0]])
            F[0, ind_F] = True
            self.gp.mu_cond, self.gp.Sigma_cond = self.gp.GPupd(self.gp.mu_cond, self.gp.Sigma_cond, F,
                                                                self.gp.R, F @ self.gp.mu_truth)
            self.gp.getEIBVField()


if __name__ == "__main__":
    starting_loc = Location(.0, 1.)
    target_loc = Location(1., .0)
    p = PathPlanner(starting_location=starting_loc, target_location=target_loc)
    p.run()

