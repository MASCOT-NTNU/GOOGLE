"""
This script produces the planned trajectory
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""
import matplotlib.pyplot as plt

from GOOGLE.Simulation_2DNidelva.Plotting.plotting_func import *
from GOOGLE.Simulation_2DNidelva.GPKernel.GPKernel import *
from GOOGLE.Simulation_2DNidelva.Tree.Location import *
from GOOGLE.Simulation_2DNidelva.Config.Config import *
from GOOGLE.Simulation_2DNidelva.Tree.Knowledge import Knowledge
from GOOGLE.Simulation_2DNidelva.PlanningStrategies.RRTStar import RRTStar

foldername = PATH_REPLICATES + "R_{:03d}/rrtstar/".format(0)
checkfolder(foldername)


class PathPlanner:

    trajectory = []
    distance_travelled = 0
    waypoint_return_counter = 0

    def __init__(self, starting_location=None, goal_location=None, budget=None):
        # load data
        self.dataset = pd.read_csv(PATH_DATA).to_numpy()
        self.coordinates_wgs = self.dataset[:, 0:3]
        x, y = latlon2xy(self.coordinates_wgs[:, 0], self.coordinates_wgs[:, 1], LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        x, y = map(vectorise, [x, y])
        self.coordinates_xy = np.hstack((x, y))
        self.mu_prior = vectorise(self.dataset[:, -1])
        self.polygon_border = pd.read_csv(PATH_BORDER)
        self.polygon_obstacle = pd.read_csv(PATH_OBSTACLE)
        self.polygon_border_xy = self.polygon_border[['x', 'y']].to_numpy()
        self.polygon_obstacle_xy = self.polygon_obstacle[['x', 'y']].to_numpy()
        self.random_locations = np.load(PATH_RANDOM_LOCATIONS)

        # set starting location
        self.starting_location = starting_location
        self.goal_location = goal_location
        self.budget = budget

        self.knowledge = Knowledge(coordinates_xy=self.coordinates_xy, starting_location=self.starting_location,
                                   goal_location=self.goal_location, random_locations=self.random_locations,
                                   goal_sample_rate=GOAL_SAMPLE_RATE, polygon_border_xy=self.polygon_border_xy,
                                   polygon_obstacle_xy=self.polygon_obstacle_xy,
                                   step_size=STEPSIZE, maximum_iteration=MAXITER_EASY,
                                   distance_neighbour_radar=DISTANCE_NEIGHBOUR_RADAR,
                                   distance_tolerance=DISTANCE_TOLERANCE,
                                   budget=self.budget, threshold=THRESHOLD)
        self.knowledge.mu_prior = self.mu_prior
        self.gp = GPKernel(self.knowledge)

    def plot_synthetic_field(self):
        cmap = get_cmap("RdBu", 10)
        plotf_vector(self.knowledge.coordinates_xy, self.knowledge.mu_truth, "Ground Truth", cmap=CMAP, vmin=20, vmax=36,
                     cbar_title="Salinity", knowledge=self.knowledge, stepsize=1.5, threshold=self.knowledge.threshold)
        # plt.show()
        plt.savefig(foldername+"truth.png")
        plt.close("all")

        plotf_vector(self.knowledge.coordinates_xy, self.knowledge.mu_prior, "Prior", cmap=CMAP, vmin=20, vmax=36,
                     cbar_title="Salinity", knowledge=self.knowledge, stepsize=1.5, threshold=self.knowledge.threshold)
        # plt.show()
        plt.savefig(foldername + "prior.png")
        plt.close('all')

    def run(self):
        self.current_location = self.starting_location
        self.previous_location = self.current_location  #TODO: dot product will be zero, no effect on the first location.
        self.trajectory.append(self.current_location)
        # self.gp.get_cost_valley(self.current_location, self.previous_location, self.goal_location, self.budget)
        self.gp.get_budget_field(self.current_location, self.goal_location, self.budget)
        self.gp.save_information_to_knowledge()

        ind_min_cost = np.argmin(self.knowledge.cost_valley)
        ending_location = self.get_location_from_ind(ind_min_cost)

        for i in range(NUM_STEPS):
            print("Step: ", i)
            t1 = time.time()
            if not self.knowledge.gohome:
                self.knowledge.starting_location = self.current_location
                self.knowledge.ending_location = self.goal_location
                self.rrtstar = RRTStar(self.knowledge)
                self.rrtstar.expand_trees()
                self.rrtstar.get_shortest_trajectory()
                while len(self.rrtstar.trajectory) <= 2: #TODO: check detailed double check not having path
                    self.knowledge.maximum_iteration += MAXITER_EASY
                    self.rrtstar.expand_trees()
                    self.rrtstar.get_shortest_trajectory()
                    print("New path length: ", len(self.rrtstar.trajectory))
                self.path_minimum_cost = self.rrtstar.trajectory
                t2 = time.time()
                print("Path planning takes: ", t2 - t1)
                self.plot_knowledge(i)
                self.next_location = LocationXY(self.path_minimum_cost[-2, 0], self.path_minimum_cost[-2, 1])
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

            self.distance_travelled = get_distance_between_xy_locations(self.current_location, self.next_location)
            self.budget = self.budget - self.distance_travelled

            self.current_location = self.next_location

            print("Budget left: ", self.budget)
            print("Distance travelled: ", self.distance_travelled)

            ind_F = get_ind_at_location2d_xy(self.knowledge.coordinates_xy, self.current_location)
            F = np.zeros([1, self.knowledge.coordinates_xy.shape[0]])
            F[0, ind_F] = True
            self.knowledge.mu_cond, self.knowledge.Sigma_cond = update_GP_field(self.knowledge.mu_cond,
                                                                                self.knowledge.Sigma_cond, F, self.gp.R,
                                                                                F @ self.knowledge.mu_truth)

            # self.gp.get_cost_valley(self.current_location, self.previous_location, self.goal_location, self.budget)
            self.gp.get_budget_field(self.current_location, self.goal_location, self.budget)
            self.gp.save_information_to_knowledge()
            ind_min_cost = np.argmin(self.knowledge.cost_valley)
            ending_location = self.get_location_from_ind(ind_min_cost)

            self.previous_location = self.current_location
            self.trajectory.append(self.current_location)

    def get_route_home(self):
        distance_remaining = get_distance_between_xy_locations(self.current_location, self.goal_location)
        angle = np.math.atan2(self.goal_location.x - self.current_location.x,
                              self.goal_location.y - self.current_location.y)
        gaps = np.arange(0, distance_remaining, STEPSIZE)
        self.num_waypoints_return_home = len(gaps)
        distance_gaps = np.linspace(0, distance_remaining, self.num_waypoints_return_home)
        waypoints_location = []
        for i in range(self.num_waypoints_return_home):
            x = self.current_location.x + distance_gaps[i] * np.sin(angle)
            y = self.current_location.y + distance_gaps[i] * np.cos(angle)
            waypoints_location.append(LocationXY(x, y))
        return waypoints_location

    def get_location_from_ind(self, ind):
        return LocationXY(self.knowledge.coordinates_xy[ind, 0], self.knowledge.coordinates_xy[ind, 1])

    def is_arrived(self, current_loc):
        if get_distance_between_xy_locations(current_loc, self.goal_location) <= DISTANCE_TOLERANCE:
            return True
        else:
            return False

    def plot_knowledge(self, fig_index):
        # == plotting ==
        fig = plt.figure(figsize=(60, 8))
        gs = GridSpec(nrows=1, ncols=5)
        ax = fig.add_subplot(gs[0])
        plotf_vector(self.knowledge.coordinates_xy, self.knowledge.mu_truth, "Ground Truth", cmap=CMAP, vmin=20, vmax=36,
                     cbar_title="Salinity", knowledge=self.knowledge, stepsize=1.2, threshold=self.knowledge.threshold)

        ax = fig.add_subplot(gs[1])
        plotf_vector(self.knowledge.coordinates_xy, self.knowledge.mu_cond, "Conditional Mean", cmap=CMAP, vmin=20, vmax=36,
                     cbar_title="Salinity", knowledge=self.knowledge, stepsize=1.2, threshold=self.knowledge.threshold)
        plotf_trajectory(self.trajectory)

        # ax = fig.add_subplot(gs[2])
        # plotf_vector(self.knowledge.coordinates_xy, self.knowledge.cost_eibv, "COST_{EIBV}", cmap=CMAP, cbar_title="Cost",
        #              colorbar=True, vmin=-.2, vmax=1.2, knowledge=self.knowledge, stepsize=.1)
        # plotf_trajectory(self.trajectory)
        #
        # ax = fig.add_subplot(gs[3])
        # plotf_vector(self.knowledge.coordinates_xy, self.knowledge.cost_vr, "COST_{VR}", cmap=CMAP, cbar_title="Cost",
        #              colorbar=True, vmin=-.2, vmax=1.2, knowledge=self.knowledge, stepsize=.1)
        # plotf_trajectory(self.trajectory)

        ax = fig.add_subplot(gs[4])
        self.rrtstar.plot_tree()
        cost_valley = self.knowledge.cost_valley
        # cost_valley[cost_valley == np.inf] = PENALTY # To avoid plotting interpolation
        # plotf_vector(self.knowledge.coordinates, cost_valley, "COST VALLEY", cmap=CMAP, vmin=-.2, vmax=4, cbar_title="Cost",
        #              colorbar=True, knowledge=self.knowledge, stepsize=.1)

        # plt.scatter(self.knowledge.coordinates_xy[:, 1],
        #             self.knowledge.coordinates_xy[:, 0],
        #             c=self.knowledge.cost_valley, cmap=CMAP,
        #             vmin=0, vmax=4, alpha=.6, s=105)
        plt.plot(self.knowledge.polygon_border_xy[:, 1], self.knowledge.polygon_border_xy[:, 0], 'k-', linewidth=1)
        plt.plot(self.knowledge.polygon_obstacle_xy[:, 1], self.knowledge.polygon_obstacle_xy[:, 0], 'k-', linewidth=1)
        plt.plot(self.knowledge.starting_location.y, self.knowledge.starting_location.x, 'kv', ms=10)
        plt.plot(self.knowledge.goal_location.y, self.knowledge.goal_location.x, 'rv', ms=10)
        plt.xlim([np.amin(self.knowledge.coordinates_xy[:, 1]), np.amax(self.knowledge.coordinates_xy[:, 1])])
        plt.ylim([np.amin(self.knowledge.coordinates_xy[:, 0]), np.amax(self.knowledge.coordinates_xy[:, 0])])
        # plt.colorbar()
        plt.title("COST_{Valley}")
        plt.savefig(foldername + "P_{:03d}.png".format(fig_index))
        plt.close("all")


if __name__ == "__main__":
    starting_location = WGS2XY(LocationWGS(63.455674, 10.429927))
    goal_location = WGS2XY(LocationWGS(63.440887, 10.354804))
    p = PathPlanner(starting_location=starting_location, goal_location=goal_location, budget=BUDGET)
    # p.plot_synthetic_field()
    p.run()




