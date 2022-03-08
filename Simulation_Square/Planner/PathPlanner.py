"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""
import matplotlib.pyplot as plt

from usr_func import *
from GOOGLE.Simulation_Square.Plotting.plotting_func import *
from GOOGLE.Simulation_Square.GPKernel.GPKernel import *
from GOOGLE.Simulation_Square.Tree.Location import Location
from GOOGLE.Simulation_Square.Config.Config import *
from GOOGLE.Simulation_Square.Tree.Knowledge import Knowledge
from GOOGLE.Simulation_Square.PlanningStrategies.RRTStar import RRTStar

np.random.seed(0)
BUDGET = 5
NUM_STEPS = 70
FIGPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/fig/rrt_star/"


class PathPlanner:

    trajectory = []

    def __init__(self, starting_location=None, target_location=None):
        self.gp = GPKernel()
        self.gp.getEIBVField()
        self.starting_location = starting_location
        self.goal_location = target_location
        self.gp.getBudgetField(self.starting_location, self.goal_location, BUDGET)

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
        # ending_loc = self.target_location
        self.trajectory.append(current_loc)

        self.cost_valley = self.gp.eibv+self.gp.penalty_budget
        ind_min_cost = np.argmin(self.cost_valley)
        ending_loc = Location(self.gp.grid_vector[ind_min_cost, 0], self.gp.grid_vector[ind_min_cost, 1])

        for i in range(NUM_STEPS):
            print("Step: ", i)
            # print("Total budget: ", budget)

            # == path planning ==
            t1 = time.time()
            knowledge = Knowledge(starting_location=current_loc, ending_location=ending_loc,
                                  goal_location=self.goal_location, goal_sample_rate=GOAL_SAMPLE_RATE,
                                  step_size=STEPSIZE, budget=budget, kernel=self.gp, mu=self.gp.mu_cond,
                                  Sigma=self.gp.Sigma_cond, gohome=gohome)

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
            plotf_vector(self.gp.grid_vector, self.cost_valley, "EIBV+BUDGET cost valley ", alpha=.1, cmap=cmap)
            # plotf_budget_radar([goal_loc.x, goal_loc.y], budget)
            # plt.show()
            plt.savefig(FIGPATH + "P_{:03d}.png".format(i))
            plt.close("all")
            # == end plotting ==

            next_loc = Location(path[-2, 0], path[-2, 1])
            discrepancy = np.sqrt((current_loc.x - self.goal_location.x) ** 2 +
                                  (current_loc.y - self.goal_location.y) ** 2)
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
            self.gp.getBudgetField(current_loc, self.goal_location, budget)
            self.cost_valley = self.gp.eibv + self.gp.penalty_budget
            ind_min_cost = np.argmin(self.cost_valley)

            # if self.mustReturn(current_loc, budget):
            #     print("Now I must return")
            #     ending_loc = self.goal_location
            #     gohome = True
            # else:
            ending_loc = Location(self.gp.grid_vector[ind_min_cost, 0], self.gp.grid_vector[ind_min_cost, 1])

            if self.isArrived(current_loc):
                print("Arrived")
                break

    # def mustReturn(self, current_loc, budget):
    #     self.budget_ellipse_a = (budget - BUDGET_MARGIN) / 2
    #     self.budget_ellipse_c = self.get_distance_between_locations(current_loc, self.goal_location) / 2
    #     self.budget_ellipse_b = np.sqrt(self.budget_ellipse_a ** 2 - self.budget_ellipse_c ** 2)
    #
    #     x_wgs = self.goal_location.x - self.rrtstar.budget_middle_location.x
    #     y_wgs = self.goal_location.y - self.rrtstar.budget_middle_location.y
    #     x_usr = (x_wgs * np.cos(self.rrtstar.budget_ellipse_angle) +
    #              y_wgs * np.sin(self.rrtstar.budget_ellipse_angle))
    #     y_usr = (-x_wgs * np.sin(self.rrtstar.budget_ellipse_angle) +
    #              y_wgs * np.cos(self.rrtstar.budget_ellipse_angle))
    #
    #     if (x_usr / self.rrtstar.budget_ellipse_a) ** 2 + (y_usr / self.rrtstar.budget_ellipse_b) ** 2 <= 1:
    #         return False
    #     else:
    #         return True
    #     pass

    def isArrived(self, current_loc):
        if self.get_distance_between_locations(current_loc, self.goal_location) <= DISTANCE_TOLERANCE:
            return True
        else:
            return False

    @staticmethod
    def get_distance_between_locations(loc1, loc2):
        return np.sqrt((loc1.x - loc2.x) ** 2 + (loc1.y - loc2.y) ** 2)


if __name__ == "__main__":
    starting_loc = Location(.0, .0)
    target_loc = Location(.0, 1.)
    p = PathPlanner(starting_location=starting_loc, target_location=target_loc)
    p.run()

#%%
gx, gy = np.gradient(p.gp.mu_prior_matrix)
gg = np.sqrt(gx ** 2 + gy ** 2)
plt.imshow(gg)
plt.colorbar()
plt.show()

