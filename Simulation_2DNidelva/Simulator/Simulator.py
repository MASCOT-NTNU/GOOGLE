"""
This script generates the next waypoint based on the current knowledge and previous path
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-18
"""
import matplotlib.pyplot as plt

from GOOGLE.Simulation_2DNidelva.PlanningStrategies.Myopic2D import MyopicPlanning_2D
from GOOGLE.Simulation_2DNidelva.PlanningStrategies.Lawnmower import LawnMowerPlanning
from GOOGLE.Simulation_2DNidelva.PlanningStrategies.RRTStar import RRTStar
from GOOGLE.Simulation_2DNidelva.Simulator.Sampler import Sampler
from GOOGLE.Simulation_2DNidelva.Plotting.plotting_func import *
from GOOGLE.Simulation_2DNidelva.Plotting.KnowledgePlot import KnowledgePlot
from GOOGLE.Simulation_2DNidelva.GPKernel.GPKernel import *
from GOOGLE.Simulation_2DNidelva.Tree.Location import *
from GOOGLE.Simulation_2DNidelva.Config.Config import *
from GOOGLE.Simulation_2DNidelva.Tree.Knowledge import Knowledge


class Simulator:

    trajectory = []
    distance_travelled = 0
    waypoint_return_counter = 0
    knowledge = None


    def __init__(self, steps=10, random_seed=0):
        print("Random seed: ", random_seed)
        self.seed = random_seed
        self.steps = steps
        np.random.seed(self.seed)

        # == setup path planner

        # self.starting_location = Location(63.43402, 10.36401)
        self.starting_location = Location(63.43990, 10.35273)
        self.goal_location = Location(63.45546, 10.43784)
        self.budget = BUDGET
        self.gp = GPKernel()

        self.knowledge = Knowledge(starting_location=None,
                                   goal_location=self.goal_location, goal_sample_rate=GOAL_SAMPLE_RATE,
                                   polygon_border=self.gp.polygon_border, polygon_obstacle=self.gp.polygon_obstacle,
                                   step_size=STEPSIZE, maximum_iteration=MAXITER_EASY,
                                   distance_neighbour_radar=DISTANCE_NEIGHBOUR_RADAR,
                                   distance_tolerance=DISTANCE_TOLERANCE, budget=self.budget, kernel=self.gp)

    def plot_synthetic_field(self):
        foldername = PATH_REPLICATES + "R_{:03d}/".format(self.seed)
        checkfolder(foldername)
        plotf_vector(self.gp.coordinates, self.gp.mu_truth, "Ground Truth", cmap=CMAP, vmin=20, vmax=36,
                     cbar_title="Salinity", kernel=self.gp, stepsize=1.5, threshold=self.gp.threshold, self=self.knowledge)
        plt.savefig(foldername+"truth.png")
        plt.close('all')
        plt.show()
        plotf_vector(self.gp.coordinates, self.gp.mu_prior, "Prior", cmap=CMAP, vmin=20, vmax=33,
                     cbar_title="Salinity", kernel=self.gp, stepsize=1.5, threshold=self.gp.threshold, self=self.knowledge)
        plt.savefig(foldername + "prior.png")
        plt.close('all')
        plt.show()

        # self.knowledge_prior = self.knowledge
        # self.knowledge_prior.excursion_prob = get_excursion_prob_1d(self.gp.mu_prior,
        #                                                             self.gp.Sigma_prior,
        #                                                             self.gp.threshold)
        # plotf_vector(self.gp.coordinates, self.knowledge_prior.excursion_prob, "Prior EP", vmin=0, vmax=1, cmap=CMAP, kernel=self.gp,
        #              self=self)
        # plt.show()
        #
        # self.knowledge_ground_truth = self.knowledge
        # self.knowledge_ground_truth.mu = self.gp.mu_truth
        # self.knowledge_ground_truth.excursion_prob = get_excursion_prob_1d(self.gp.mu_truth,
        #                                                                    self.gp.Sigma_prior,
        #                                                                    self.gp.threshold)
        # plotf_vector(self.gp.coordinates, self.knowledge_prior.excursion_prob, "Truth EP", vmin=0, vmax=1, cmap=CMAP, kernel=self.gp,
        #              self=self)
        # plt.show()

    def run_2d(self):
        self.ind_start = get_grid_ind_at_nearest_loc([self.starting_location.lat, self.starting_location.lon, 0],
                                                        self.knowledge.coordinates) # get nearest neighbour
        # plt.plot(self.starting_location.lon, self.starting_location.lat, 'b.')
        # plt.plot(self.knowledge.coordinates[self.ind_start, 1], self.knowledge.coordinates[self.ind_start, 0], 'r.')
        # plt.plot(self.knowledge.coordinates[:, 1], self.knowledge.coordinates[:, 0], 'k.', alpha=.25)
        self.knowledge.ind_prev = self.knowledge.ind_now = self.ind_sample = self.ind_start
        # plt.show()

        foldername = PATH_REPLICATES + "R_{:03d}/2D/".format(self.seed)
        checkfolder(foldername)

        self.current_location = self.starting_location
        self.previous_location = self.current_location
        self.waypoint_return_counter = 0

        for i in range(self.steps):
            print("Step No. ", i)
            self.knowledge = Sampler(self.knowledge, self.knowledge.kernel.mu_truth, self.ind_sample).Knowledge

            self.gp.get_budget_field(self.current_location, self.goal_location, self.budget)

            self.distance_travelled = get_distance_between_locations(self.current_location, self.previous_location)
            self.budget = self.budget - self.distance_travelled
            print("Budget left: ", self.budget)
            print("Distance travelled: ", self.distance_travelled)

            if not self.gp.gohome:
                self.next_location = MyopicPlanning_2D(knowledge=self.knowledge).next_waypoint
            else:
                print("Go home")
                if self.waypoint_return_counter == 0:
                    print("Compute route home only here once!")
                    self.waypoints = self.get_route_home(stepsize=DISTANCE_NEIGHBOUR)
                if self.waypoint_return_counter < self.num_waypoints_return_home:
                    self.next_location = self.waypoints[self.waypoint_return_counter]
                    self.waypoint_return_counter += 1
                    # self.plot_knowledge(i)
                else:
                    print("Home already! Mission complete")
                    # self.plot_knowledge(i)
                    break
            self.ind_sample = get_grid_ind_at_nearest_loc([self.next_location.lat, self.next_location.lon, 0],
                                                          self.knowledge.coordinates)
            self.knowledge.step_no = i
            self.plot_2d(foldername, i)

            self.previous_location = self.current_location
            self.current_location = self.next_location

    def plot_2d(self, foldername, i):
        filename = foldername + "P_{:03d}.png".format(i)
        plt.plot(self.starting_location.lon, self.starting_location.lat, 'gv', ms=20)
        plt.plot(self.goal_location.lon, self.goal_location.lat, 'cv', ms=20)
        plt.plot(self.knowledge.coordinates[:, 1], self.knowledge.coordinates[:, 0], 'k.', alpha=.05)
        plt.plot(self.current_location.lon, self.current_location.lat, 'r.')
        plt.plot(self.knowledge.coordinates[self.knowledge.ind_cand, 1],
                 self.knowledge.coordinates[self.knowledge.ind_cand, 0], 'b.')
        plt.plot(self.knowledge.coordinates[self.knowledge.ind_next, 1],
                 self.knowledge.coordinates[self.knowledge.ind_next, 0], 'y.')
        plotf_vector(self.gp.coordinates, self.knowledge.kernel.mu_cond, "Mean", cmap=CMAP, vmin=20, vmax=33,
                     kernel=self.gp, stepsize=1.5, threshold=self.gp.threshold, self=self.knowledge)

        lat_temp, lon_temp = xy2latlon(2 * self.gp.budget_ellipse_a, 2 * self.gp.budget_ellipse_b,
                                       LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        ellipse = Ellipse(xy=(self.gp.budget_middle_location.lon, self.gp.budget_middle_location.lat),
                          width=lat_temp, height=lon_temp,
                          angle=math.degrees(self.gp.budget_ellipse_angle),
                          edgecolor='r', fc='None', lw=2)
        plt.gca().add_patch(ellipse)

        trajectory = np.array(self.knowledge.trajectory)
        plt.plot(trajectory[:, 1], trajectory[:, 0], 'k.-')
        lat_min, lon_min = map(np.amin, [self.gp.polygon_border[:, 0], self.gp.polygon_border[:, 1]])
        lat_max, lon_max = map(np.amax, [self.gp.polygon_border[:, 0], self.gp.polygon_border[:, 1]])

        plt.xlim([lon_min, lon_max])
        plt.ylim([lat_min, lat_max])
        plt.savefig(filename)
        plt.close('all')
        pass

    def run_lawn_mower(self):
        lat_start, lon_start, depth_start = self.lawn_mower_path_3d[self.starting_index, :]
        ind_start = get_grid_ind_at_nearest_loc_2d([lat_start, lon_start, depth_start], self.knowledge.coordinates)
        self.knowledge.ind_prev = self.knowledge.ind_now = ind_start

        foldername = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/Replicates/R_{:03d}/Lawnmower/".format(self.seed)
        checkfolder(foldername)

        for i in range(self.steps):
            # print("Step No. ", i)
            lat_next, lon_next, depth_next = self.lawn_mower_path_3d[self.starting_index + i, :]
            ind_sample = get_grid_ind_at_nearest_loc_2d([lat_next, lon_next, depth_next], self.knowledge.coordinates)

            self.knowledge.step_no = i
            self.knowledge = Sampler(self.knowledge, self.ground_truth, ind_sample).Knowledge
        KnowledgePlot(knowledge=self.knowledge, vmin=VMIN, vmax=VMAX, filename=foldername+"Field_{:03d}".format(i), html=False)

    def run_rrtstar(self):
        self.current_location = self.starting_location
        self.previous_location = self.current_location  # TODO: dot product will be zero, no effect on the first location.
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
                                      polygon_border=self.gp.polygon_border,
                                      polygon_obstacle=self.gp.polygon_obstacle,
                                      step_size=STEPSIZE, maximum_iteration=MAXITER_EASY,
                                      distance_neighbour_radar=RADIUS_NEIGHBOUR,
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


    def get_route_home(self, stepsize=None):
        distance_remaining = get_distance_between_locations(self.current_location, self.goal_location)
        angle = np.math.atan2(self.goal_location.x - self.current_location.x,
                              self.goal_location.y - self.current_location.y)
        gaps = np.arange(0, distance_remaining, stepsize)
        self.num_waypoints_return_home = len(gaps)
        distance_gaps = np.linspace(0, distance_remaining, self.num_waypoints_return_home)
        waypoints_location = []
        print("distance gaps:", distance_gaps)
        for i in range(self.num_waypoints_return_home):
            x = self.current_location.x + distance_gaps[i] * np.sin(angle)
            y = self.current_location.y + distance_gaps[i] * np.cos(angle)
            lat, lon = xy2latlon(x, y, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
            print("loc: ", lat, lon)
            waypoints_location.append(Location(lat, lon))
        return waypoints_location

    def get_location_from_ind(self, ind):
        return Location(self.gp.coordinates[ind, 0], self.gp.coordinates[ind, 1])

    def is_arrived(self, current_loc):
        if get_distance_between_locations(current_loc, self.goal_location) <= DISTANCE_TOLERANCE:
            return True
        else:
            return False


if __name__ == "__main__":
    a = Simulator(steps=100, random_seed=2)
    a.plot_synthetic_field()
    a.run_2d()
    # a.run_lawn_mower()
#%%
plt.plot(a.knowledge.expectedVariance)
plt.show()
