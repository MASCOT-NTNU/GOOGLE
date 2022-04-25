"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""
import matplotlib.pyplot as plt

from usr_func import *
from GOOGLE.Simulation_2DSquare.Plotting.plotting_func import *
from GOOGLE.Simulation_2DSquare.GPKernel.GPKernel import *
from GOOGLE.Simulation_2DSquare.Tree.Location import Location
from GOOGLE.Simulation_2DSquare.Config.Config import *
from GOOGLE.Simulation_2DSquare.Tree.Knowledge import Knowledge
from GOOGLE.Simulation_2DSquare.PlanningStrategies.RRTStar import RRTStar

np.random.seed(0)
BUDGET = 5
NUM_STEPS = 70
FIGPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/fig/rrt_star/"


class PathPlanner:

    trajectory = []

    def __init__(self, starting_location=None, target_location=None):
        self.gp = GPKernel()
        self.gp.get_eibv_field()
        self.starting_location = starting_location
        self.goal_location = target_location
        self.gp.get_budget_field(self.starting_location, self.goal_location, BUDGET)
        self.gp.get_gradient_field()
        self.gp.get_variance_reduction_field()
        self.gp.get_obstacle_field()
        self.gp.get_direction_field(self.starting_location, self.goal_location)

    def run(self):

        # plotf_vector(self.gp.grid_vector, self.gp.mu_truth, "Truth")
        # plt.show()
        #
        # plotf_vector(self.gp.grid_vector, self.gp.mu_prior_vector, "Prior")
        # plt.show()

        budget = BUDGET
        gohome = False
        distance_travelled = 0
        current_loc = self.starting_location
        previous_loc = current_loc
        # ending_loc = self.target_location
        self.trajectory.append(current_loc)

        self.cost_valley = (self.gp.cost_eibv +
                            self.gp.cost_budget +
                            self.gp.gradient_vector +
                            self.gp.cost_vr +
                            self.gp.cost_obstacle +
                            self.gp.cost_direction)
        ind_min_cost = np.argmin(self.cost_valley)
        ending_loc = Location(self.gp.grid_vector[ind_min_cost, 0], self.gp.grid_vector[ind_min_cost, 1])

        for i in range(NUM_STEPS):
            print("Step: ", i)
            t1 = time.time()
            if not gohome:
                knowledge = Knowledge(starting_location=current_loc, ending_location=ending_loc,
                                      goal_location=self.goal_location, goal_sample_rate=GOAL_SAMPLE_RATE,
                                      step_size=STEPSIZE, budget=budget, kernel=self.gp, mu=self.gp.mu_cond,
                                      Sigma=self.gp.Sigma_cond)

                self.rrtstar = RRTStar(knowledge)
                self.rrtstar.expand_trees()
                self.rrtstar.get_shortest_trajectory()
                path = self.rrtstar.trajectory
                t2 = time.time()
                print("Path planning takes: ", t2 - t1)

                self.plot_knowledge(i)

                next_loc = Location(path[-2, 0], path[-2, 1])
            else:
                next_loc = self.goal_location
                self.plot_knowledge(i)

            distance_travelled += self.get_distance_between_locations(current_loc, next_loc)
            budget = BUDGET - distance_travelled
            current_loc = next_loc
            self.trajectory.append(current_loc)

            print("Budget left: ", budget)
            print("Distance travelled: ", distance_travelled)

            ind_F = self.gp.get_ind_F(current_loc)
            F = np.zeros([1, self.gp.grid_vector.shape[0]])
            F[0, ind_F] = True
            self.gp.mu_cond, self.gp.Sigma_cond = self.gp.update_GP_field(self.gp.mu_cond, self.gp.Sigma_cond, F,
                                                                          self.gp.R, F @ self.gp.mu_truth)
            self.gp.get_eibv_field()
            self.gp.get_budget_field(current_loc, self.goal_location, budget)
            self.gp.get_gradient_field()
            self.gp.get_variance_reduction_field()
            self.gp.get_obstacle_field()
            self.gp.get_direction_field(current_loc, previous_loc)
            # self.cost_valley = self.gp.eibv + self.gp.penalty_budget
            self.cost_valley = (self.gp.cost_eibv +
                                self.gp.cost_budget +
                                self.gp.gradient_vector +
                                self.gp.cost_vr +
                                self.gp.cost_obstacle +
                                self.gp.cost_direction)
            ind_min_cost = np.argmin(self.cost_valley)
            ending_loc = Location(self.gp.grid_vector[ind_min_cost, 0], self.gp.grid_vector[ind_min_cost, 1])

            previous_loc = current_loc

            if budget < BUDGET_MARGIN:
                print("Time to go home!")
                gohome = True

            if self.is_arrived(current_loc):
                print("Arrived")
                # break

    def is_arrived(self, current_loc):
        if self.get_distance_between_locations(current_loc, self.goal_location) <= DISTANCE_TOLERANCE:
            return True
        else:
            return False

    @staticmethod
    def get_distance_between_locations(loc1, loc2):
        return np.sqrt((loc1.X_START - loc2.X_START) ** 2 + (loc1.Y_START - loc2.Y_START) ** 2)

    def plot_knowledge(self, i):

        # == plotting ==
        fig = plt.figure(figsize=(40, 5))
        gs = GridSpec(nrows=1, ncols=7)
        ax = fig.add_subplot(gs[0])
        cmap = get_cmap("RdBu", 10)
        plotf_vector(self.gp.grid_vector, self.gp.mu_truth, "Ground Truth", cmap=cmap)

        ax = fig.add_subplot(gs[1])
        plotf_vector(self.gp.grid_vector, self.gp.mu_cond, "Conditional Mean", cmap=cmap)
        plotf_trajectory(self.trajectory)

        ax = fig.add_subplot(gs[2])
        self.rrtstar.plot_tree()
        plotf_vector(self.gp.grid_vector, self.cost_valley, "EIBV+BUDGET+Gradient+VR ", alpha=.1, cmap=cmap)

        ax = fig.add_subplot(gs[3])
        plotf_vector(self.gp.grid_vector, self.gp.cost_eibv, "EIBV", cmap=cmap)
        # plotf_vector(self.gp.grid_vector, np.sqrt(np.diag(self.gp.Sigma_cond)), "Prediction Error", cmap=cmap)
        plotf_trajectory(self.trajectory)

        ax = fig.add_subplot(gs[4])
        plotf_vector(self.gp.grid_vector, self.gp.gradient_vector, "Gradient", cmap=cmap)
        plotf_trajectory(self.trajectory)

        ax = fig.add_subplot(gs[5])
        plotf_vector(self.gp.grid_vector, self.gp.cost_vr, "VR", cmap=cmap)
        plotf_trajectory(self.trajectory)

        ax = fig.add_subplot(gs[6])
        plotf_vector(self.gp.grid_vector, self.gp.cost_budget, "BUDGET", cmap=cmap)
        plotf_trajectory(self.trajectory)

        plt.savefig(FIGPATH + "P_{:03d}.png".format(i))
        plt.close("all")

if __name__ == "__main__":
    starting_loc = Location(.0, .0)
    target_loc = Location(.0, 1.)
    p = PathPlanner(starting_location=starting_loc, target_location=target_loc)
    p.run()

#%%
# plotf_vector(p.gp.grid_vector, p.gp.penalty_obstacle, 'obstacle')
# plt.show()

