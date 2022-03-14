"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""

from usr_func import *
from GOOGLE.Simulation_Square.Plotting.plotting_func import *
from GOOGLE.Simulation_Square.GPKernel.GPKernel import *
from GOOGLE.Simulation_Square.Tree.Location import *
from GOOGLE.Simulation_Square.Config.Config import *
from GOOGLE.Simulation_Square.Tree.Knowledge import Knowledge
from GOOGLE.Simulation_Square.PlanningStrategies.RRTStar import RRTStar


class PathPlanner:

    trajectory = []
    distance_travelled = 0
    gohome_signal = False

    def __init__(self, starting_location=None, goal_location=None, budget=None):
        self.starting_location = starting_location
        self.goal_location = goal_location
        self.budget = budget
        self.gp = GPKernel()

    def plot_synthetic_field(self):
        plotf_vector(self.gp.grid_vector, self.gp.mu_truth, "Truth")
        plt.show()
        plotf_vector(self.gp.grid_vector, self.gp.mu_prior_vector, "Prior")
        plt.show()

    def run(self):
        self.current_location = self.starting_location
        self.previous_location = self.current_location  #TODO: dot product will be zero, no effect on the first location.
        self.trajectory.append(self.current_location)
        self.gp.get_cost_valley(self.current_location, self.previous_location, self.goal_location, self.budget)

        ind_min_cost = np.argmin(self.gp.cost_valley)
        ending_loc = self.get_location_from_ind(ind_min_cost)

        for i in range(NUM_STEPS):
            print("Step: ", i)
            t1 = time.time()
            if not self.gohome_signal:
                knowledge = Knowledge(starting_location=self.current_location, ending_location=ending_loc,
                                      goal_location=self.goal_location, goal_sample_rate=GOAL_SAMPLE_RATE,
                                      step_size=STEPSIZE, budget=self.budget, kernel=self.gp, mu=self.gp.mu_cond,
                                      Sigma=self.gp.Sigma_cond)

                self.rrtstar = RRTStar(knowledge)
                self.rrtstar.expand_trees()
                self.rrtstar.get_shortest_trajectory()
                self.path_minimum_cost = self.rrtstar.trajectory
                t2 = time.time()
                print("Path planning takes: ", t2 - t1)
                self.plot_knowledge(i)
                self.next_location = Location(self.path_minimum_cost[-2, 0], self.path_minimum_cost[-2, 1])
            else:
                self.next_location = self.goal_location
                self.plot_knowledge(i)

            self.distance_travelled = get_distance_between_locations(self.current_location, self.next_location)
            self.budget = self.budget - self.distance_travelled

            self.current_location = self.next_location

            print("Budget left: ", self.budget)
            print("Distance travelled: ", self.distance_travelled)

            ind_F = self.gp.get_ind_F(self.current_location)
            F = np.zeros([1, self.gp.num_nodes])
            F[0, ind_F] = True
            self.gp.mu_cond, self.gp.Sigma_cond = self.gp.update_GP_field(self.gp.mu_cond, self.gp.Sigma_cond, F,
                                                                          self.gp.R, F @ self.gp.mu_truth)

            self.gp.get_cost_valley(self.current_location, self.previous_location, self.goal_location, self.budget)
            ind_min_cost = np.argmin(self.gp.cost_valley)
            ending_loc = self.get_location_from_ind(ind_min_cost)

            self.previous_location = self.current_location
            self.trajectory.append(self.current_location)

            if self.gp.budget_ellipse_a <= self.gp.budget_ellipse_c:
                print("Time to go home!")
                self.gohome_signal = True

            if self.is_arrived(self.current_location):
                print("Arrived")
                # break

    def get_location_from_ind(self, ind):
        return Location(self.gp.grid_vector[ind, 0], self.gp.grid_vector[ind, 1])

    def is_arrived(self, current_loc):
        if get_distance_between_locations(current_loc, self.goal_location) <= DISTANCE_TOLERANCE:
            return True
        else:
            return False

    def plot_knowledge(self, i):
        # == plotting ==
        fig = plt.figure(figsize=(30, 5))
        gs = GridSpec(nrows=1, ncols=5)
        ax = fig.add_subplot(gs[0])
        cmap = get_cmap("RdBu", 10)
        plotf_vector(self.gp.grid_vector, self.gp.mu_truth, "Ground Truth", cmap=cmap, colorbar=True)

        ax = fig.add_subplot(gs[1])
        plotf_vector(self.gp.grid_vector, self.gp.mu_cond, "Conditional Mean", cmap=cmap, colorbar=True)
        plotf_trajectory(self.trajectory)

        ax = fig.add_subplot(gs[2])
        plotf_vector(self.gp.grid_vector, self.gp.cost_eibv, "EIBV", cmap=cmap, cbar_title="Cost", colorbar=True)
        # plotf_vector(self.gp.grid_vector, np.sqrt(np.diag(self.gp.Sigma_cond)), "Prediction Error", cmap=cmap)
        plotf_trajectory(self.trajectory)

        ax = fig.add_subplot(gs[3])
        plotf_vector(self.gp.grid_vector, self.gp.cost_vr, "VR", cmap=cmap, cbar_title="Cost", colorbar=True)
        # plotf_vector(self.gp.grid_vector, np.sqrt(np.diag(self.gp.Sigma_cond)), "Prediction Error", cmap=cmap)
        plotf_trajectory(self.trajectory)

        ax = fig.add_subplot(gs[4])
        self.rrtstar.plot_tree()
        plotf_vector(self.gp.grid_vector, self.gp.cost_valley, "Cost Valley", alpha=.1,
                     cmap=cmap, cbar_title="Cost", colorbar=True, vmin=0, vmax=4)
        plt.savefig(FIGPATH + "P_{:03d}.png".format(i))
        plt.close("all")


if __name__ == "__main__":
    starting_loc = Location(.0, .0)
    goal_loc = Location(.0, 1.)
    p = PathPlanner(starting_location=starting_loc, goal_location=goal_loc, budget=BUDGET)
    p.run()




