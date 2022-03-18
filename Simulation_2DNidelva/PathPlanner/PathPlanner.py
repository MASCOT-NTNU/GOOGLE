"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""

from GOOGLE.Simulation_2DNidelva.Plotting.plotting_func import *
from GOOGLE.Simulation_2DNidelva.GPKernel.GPKernel import *
from GOOGLE.Simulation_2DNidelva.Tree.Location import *
from GOOGLE.Simulation_2DNidelva.Config.Config import *
from GOOGLE.Simulation_2DNidelva.Tree.Knowledge import Knowledge
from GOOGLE.Simulation_2DNidelva.PlanningStrategies.RRTStar import RRTStar


class PathPlanner:

    trajectory = []
    distance_travelled = 0
    waypoint_return_counter = 0

    def __init__(self, starting_location=None, goal_location=None, budget=None):
        self.starting_location = starting_location
        self.goal_location = goal_location
        self.budget = budget
        self.gp = GPKernel()

        # self.gp.mu_cond = np.zeros_like(self.gp.mu_cond) # TODO: Wrong prior

    def plot_synthetic_field(self):
        cmap = get_cmap("RdBu", 10)
        plt.plot(self.gp.polygon_border[:, 1], self.gp.polygon_border[:, 0], 'k-', linewidth=1)
        plt.plot(self.gp.polygon_obstacle[:, 1], self.gp.polygon_obstacle[:, 0], 'k-', linewidth=1)
        plotf_vector(self.gp.coordinates, self.gp.mu_truth, "Ground Truth", cmap=cmap, vmin=20, vmax=36,
                     kernel=self.gp, stepsize=1.5, threshold=self.gp.threshold)
        plt.show()
        plt.plot(self.gp.polygon_border[:, 1], self.gp.polygon_border[:, 0], 'k-', linewidth=1)
        plt.plot(self.gp.polygon_obstacle[:, 1], self.gp.polygon_obstacle[:, 0], 'k-', linewidth=1)
        plotf_vector(self.gp.coordinates, self.gp.mu_prior, "Prior", cmap=cmap, vmin=20, vmax=33,
                     kernel=self.gp, stepsize=1.5, threshold=self.gp.threshold)
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
            if not self.gp.gohome:
                knowledge = Knowledge(starting_location=self.current_location, ending_location=ending_loc,
                                      goal_location=self.goal_location, goal_sample_rate=GOAL_SAMPLE_RATE,
                                      polygon_border=self.gp.polygon_border, polygon_obstacle=self.gp.polygon_obstacle,
                                      step_size=STEPSIZE, maximum_iteration=MAXITER_EASY, distance_neighbour_radar=RADIUS_NEIGHBOUR,
                                      distance_tolerance=DISTANCE_TOLERANCE, budget=self.budget, kernel=self.gp)

                self.rrtstar = RRTStar(knowledge)
                self.rrtstar.expand_trees()
                self.rrtstar.get_shortest_trajectory()
                if len(self.rrtstar.trajectory) <= 2:
                    self.rrtstar.maxiter = MAXITER_HARD
                    self.rrtstar.expand_trees()
                    self.rrtstar.get_shortest_trajectory()
                self.path_minimum_cost = self.rrtstar.trajectory
                t2 = time.time()
                print("Path planning takes: ", t2 - t1)
                self.plot_knowledge(i)
                self.next_location = Location(self.path_minimum_cost[-2, 0], self.path_minimum_cost[-2, 1])
            else:
                if self.waypoint_return_counter == 0:
                    print("Compute route home only here once!")
                    self.waypoints = self.get_route_home()
                if self.waypoint_return_counter < self.num_waypoints_return_home:
                    self.next_location = self.waypoints[self.waypoint_return_counter]
                    self.waypoint_return_counter += 1
                    self.plot_knowledge(i)
                else:
                    print("Home already! Mission complete")
                    self.plot_knowledge(i)
                    break

            self.distance_travelled = get_distance_between_locations(self.current_location, self.next_location)
            self.budget = self.budget - self.distance_travelled

            self.current_location = self.next_location

            print("Budget left: ", self.budget)
            print("Distance travelled: ", self.distance_travelled)

            ind_F = self.gp.get_ind_F(self.current_location)
            F = np.zeros([1, self.gp.grid_xy.shape[0]])
            F[0, ind_F] = True
            self.gp.mu_cond, self.gp.Sigma_cond = self.gp.update_GP_field(self.gp.mu_cond, self.gp.Sigma_cond, F,
                                                                          self.gp.R, F @ self.gp.mu_truth)

            self.gp.get_cost_valley(self.current_location, self.previous_location, self.goal_location, self.budget)
            ind_min_cost = np.argmin(self.gp.cost_valley)
            ending_loc = self.get_location_from_ind(ind_min_cost)

            self.previous_location = self.current_location
            self.trajectory.append(self.current_location)

    def get_route_home(self):
        distance_remaining = get_distance_between_locations(self.current_location, self.goal_location)
        angle = np.math.atan2(self.goal_location.x - self.current_location.x,
                              self.goal_location.y - self.current_location.y)
        gaps = np.arange(0, distance_remaining, STEPSIZE)
        self.num_waypoints_return_home = len(gaps)
        distance_gaps = np.linspace(0, distance_remaining, self.num_waypoints_return_home)
        waypoints_location = []
        for i in range(self.num_waypoints_return_home):
            x = self.current_location.x + distance_gaps[i] * np.sin(angle)
            y = self.current_location.y + distance_gaps[i] * np.cos(angle)
            lat, lon = xy2latlon(x, y, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
            waypoints_location.append(Location(lat, lon))
        return waypoints_location

    def get_location_from_ind(self, ind):
        return Location(self.gp.coordinates[ind, 0], self.gp.coordinates[ind, 1])

    def is_arrived(self, current_loc):
        if get_distance_between_locations(current_loc, self.goal_location) <= DISTANCE_TOLERANCE:
            return True
        else:
            return False

    def plot_knowledge(self, i):
        # == plotting ==
        fig = plt.figure(figsize=(45, 8))
        gs = GridSpec(nrows=1, ncols=5)
        ax = fig.add_subplot(gs[0])
        cmap = get_cmap("BrBG", 10)
        plt.plot(self.gp.polygon_border[:, 1], self.gp.polygon_border[:, 0], 'k-', linewidth=1)
        plt.plot(self.gp.polygon_obstacle[:, 1], self.gp.polygon_obstacle[:, 0], 'k-', linewidth=1)
        plotf_vector(self.gp.coordinates, self.gp.mu_truth, "Ground Truth", cmap=cmap, colorbar=True,
                     kernel=self.gp, vmin=20, vmax=33, stepsize=1.2, threshold=self.gp.threshold)

        ax = fig.add_subplot(gs[1])
        plt.plot(self.gp.polygon_border[:, 1], self.gp.polygon_border[:, 0], 'k-', linewidth=1)
        plt.plot(self.gp.polygon_obstacle[:, 1], self.gp.polygon_obstacle[:, 0], 'k-', linewidth=1)
        plotf_vector(self.gp.coordinates, self.gp.mu_cond, "Conditional Mean", cmap=cmap, colorbar=True,
                     kernel=self.gp, vmin=20, vmax=33,stepsize=1.2, threshold=self.gp.threshold)
        plotf_trajectory(self.trajectory)

        ax = fig.add_subplot(gs[2])
        plt.plot(self.gp.polygon_border[:, 1], self.gp.polygon_border[:, 0], 'k-', linewidth=1)
        plt.plot(self.gp.polygon_obstacle[:, 1], self.gp.polygon_obstacle[:, 0], 'k-', linewidth=1)
        plotf_vector(self.gp.coordinates, self.gp.cost_eibv, "EIBV", cmap=cmap, cbar_title="Cost", colorbar=True,
                     kernel=self.gp, vmin=0, vmax=1.1, stepsize=.1)
        plotf_trajectory(self.trajectory)

        ax = fig.add_subplot(gs[3])
        plt.plot(self.gp.polygon_border[:, 1], self.gp.polygon_border[:, 0], 'k-', linewidth=1)
        plt.plot(self.gp.polygon_obstacle[:, 1], self.gp.polygon_obstacle[:, 0], 'k-', linewidth=1)
        plotf_vector(self.gp.coordinates, self.gp.cost_vr, "VR", cmap=cmap, cbar_title="Cost", colorbar=True,
                     kernel=self.gp, vmin=0, vmax=1.1, stepsize=.1)
        plotf_trajectory(self.trajectory)

        ax = fig.add_subplot(gs[4])
        self.rrtstar.plot_tree()
        plt.plot(self.starting_location.lon, self.starting_location.lat, 'kv', ms=10)
        plt.plot(self.goal_location.lon, self.goal_location.lat, 'bx', ms=10)
        plotf_vector_scatter(self.gp.coordinates, self.gp.cost_valley, "Cost Valley", alpha=.6,
                     cmap=cmap, cbar_title="Cost", colorbar=True, vmin=0, vmax=4)
        plt.savefig(FIGPATH + "P_{:03d}.png".format(i))
        plt.close("all")


if __name__ == "__main__":
    starting_location = Location(63.455674, 10.429927)
    goal_location = Location(63.440887, 10.354804)
    p = PathPlanner(starting_location=starting_location, goal_location=goal_location, budget=BUDGET)
    p.plot_synthetic_field()
    # p.run()




