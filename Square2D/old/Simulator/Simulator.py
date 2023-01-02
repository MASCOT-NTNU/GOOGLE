"""
This script generates the next waypoint based on the current knowledge and previous path
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-24
"""
import matplotlib.pyplot as plt

from GOOGLE.Square2D.PlanningStrategies.Myopic2D import MyopicPlanning2D
from GOOGLE.Square2D.PlanningStrategies.Lawnmower import LawnMowerPlanning
from GOOGLE.Square2D.PlanningStrategies.RRTStar import RRTStar
from GOOGLE.Square2D.Simulator.Sampler import Sampler
from GOOGLE.Square2D.Plotting.plotting_func import *
from GOOGLE.Square2D.GPKernel.GPKernel import *
from GOOGLE.Square2D.Tree.Location import *
from GOOGLE.Square2D.Config.Config import *
from GOOGLE.Square2D.Tree.Knowledge import Knowledge


class Simulator:

    def __init__(self, steps=10, random_seed=0, replicates=False, mute=False):
        print("Random seed: ", random_seed)
        self.trajectory = []
        self.distance_travelled = 0
        self.waypoint_return_counter = 0

        self.seed = random_seed
        self.steps = steps
        self.replicates = replicates
        self.mute = mute
        np.random.seed(self.seed)

        # == setup path planner
        filepath_grid = FILEPATH + "PreConfig/WaypointGraph/WaypointGraph.csv"
        filepath_mu_prior = FILEPATH + "PreConfig/Data/mu_prior.csv"
        self.grid = pd.read_csv(filepath_grid).to_numpy()
        self.mu_prior = vectorise(pd.read_csv(filepath_mu_prior)['mu_prior'].to_numpy())

        self.polygon_border = np.array(BORDER)
        self.polygon_obstacles = np.array(OBSTACLES)

        self.starting_location = Location(0, 0)
        self.goal_location = Location(0, 1)
        self.budget = BUDGET

        self.knowledge = Knowledge(grid=self.grid,  starting_location=self.starting_location,
                                   ending_location=self.goal_location,
                                   goal_location=self.goal_location, polygon_border=self.polygon_border,
                                   polygon_obstacles=self.polygon_obstacles,
                                   goal_sample_rate=GOAL_SAMPLE_RATE, step_size=STEPSIZE,
                                   step_size_lawnmower=STEPSIZE_LAWNMOWER, maximum_iteration=MAXITER_EASY,
                                   distance_neighbour_radar=DISTANCE_NEIGHBOUR_RADAR,
                                   distance_neighbour_radar_myopic2d=DISTANCE_NEIGHBOUR_RADAR4MYOPIC2D,
                                   distance_tolerance=DISTANCE_TOLERANCE, budget=self.budget, threshold=THRESHOLD)
        self.knowledge.mu_prior = self.mu_prior
        self.gp = GPKernel(self.knowledge)
        # self.gp.save_information_to_knowledge()
        self.knowledge.mu_truth = self.gp.mu_truth
        self.knowledge.R = self.gp.R

        # == initialise lawnmower
        self.lawnmower2d = LawnMowerPlanning(knowledge=self.knowledge)
        self.lawnmower2d.get_lawnmower_path()
        self.lawnmower2d.get_refined_trajectory(stepsize=DISTANCE_NEIGHBOUR)

    def plot_synthetic_field(self):
        foldername = PATH_REPLICATES + "R_{:03d}/".format(self.seed)
        checkfolder(foldername)
        plotf_vector_triangulated(self.knowledge.grid, self.knowledge.mu_truth, "Ground Truth", cmap=CMAP, vmin=-.2, vmax=1.2,
                     cbar_title="Salinity", knowledge=self.knowledge, stepsize=.1, threshold=self.knowledge.threshold)
        plt.savefig(foldername+"truth.png")
        plt.close('all')
        plt.show()
        plotf_vector_triangulated(self.knowledge.grid, self.knowledge.mu_prior, "Prior", cmap=CMAP, vmin=-.2, vmax=1.2,
                     cbar_title="Salinity", knowledge=self.knowledge, stepsize=.1, threshold=self.knowledge.threshold)
        plt.savefig(foldername + "prior.png")
        plt.close('all')
        plt.show()

    def run_2d(self):
        self.ind_start = get_ind_at_location2d_xy(self.knowledge.grid,
                                                  Location(self.starting_location.x, self.starting_location.y))
        self.knowledge.ind_prev = self.knowledge.ind_now = self.ind_sample = self.ind_start

        foldername = PATH_REPLICATES + "R_{:03d}/2D/".format(self.seed)
        checkfolder(foldername)

        self.current_location = self.starting_location
        self.previous_location = self.current_location
        self.waypoint_return_counter = 0

        for i in range(self.steps):
            print("Step No. ", i)
            self.knowledge = Sampler(self.knowledge, self.knowledge.mu_truth, self.ind_sample).Knowledge
            self.gp.get_cost_valley(current_loc=self.current_location, previous_loc=self.previous_location, 
                                    goal_loc=self.goal_location, budget=self.budget)
            self.distance_travelled = get_distance_between_locations(self.current_location, self.previous_location)
            self.budget = self.budget - self.distance_travelled
            print("Budget left: ", self.budget)
            print("Distance travelled: ", self.distance_travelled)

            if not self.knowledge.gohome:
                self.next_location = MyopicPlanning2D(knowledge=self.knowledge).next_waypoint
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
            self.ind_sample = get_ind_at_location2d_xy(self.knowledge.xyz,
                                                       Location(self.next_location.x, self.next_location.y))
            self.knowledge.step_no = i
            if not self.replicates:
                self.plot_2d(foldername, i)

            self.previous_location = self.current_location
            self.current_location = self.next_location
        # print(self.knowledge.root_mean_squared_error)

    def plot_2d(self, foldername, i):
        filename = foldername + "P_{:03d}.png".format(i)

        fig = plt.figure(figsize=(60, 8))
        gs = GridSpec(nrows=1, ncols=5)
        ax = fig.add_subplot(gs[0])
        self.plot_additional_candidate_locations()
        plotf_vector_triangulated(self.knowledge.grid, self.knowledge.mu_truth, "Ground Truth", cmap=CMAP, vmin=-.2,
                                  vmax=1.2,
                                  cbar_title="Salinity", knowledge=self.knowledge, stepsize=.1,
                                  threshold=self.knowledge.threshold)
        trajectory = np.array(self.knowledge.sampling_location_plot)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'k.-')

        ax = fig.add_subplot(gs[1])
        self.plot_additional_candidate_locations()
        plotf_vector_triangulated(self.knowledge.grid, self.knowledge.mu_cond, "Conditional Mean", cmap=CMAP, vmin=-.2,
                                  vmax=1.2,
                                  cbar_title="Salinity", knowledge=self.knowledge, stepsize=.1,
                                  threshold=self.knowledge.threshold)
        # plotf_trajectory(self.trajectory)
        trajectory = np.array(self.knowledge.sampling_location_plot)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'k.-')

        ax = fig.add_subplot(gs[2])
        self.plot_additional_candidate_locations()
        plotf_vector_triangulated(self.knowledge.grid, self.knowledge.cost_eibv, "COST_{EIBV}", cmap=CMAP,
                                  cbar_title="Cost",
                                  colorbar=True, vmin=-.2, vmax=1.2, knowledge=self.knowledge, stepsize=.1)
        # plotf_trajectory(self.trajectory)
        trajectory = np.array(self.knowledge.sampling_location_plot)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'k.-')

        ax = fig.add_subplot(gs[3])
        self.plot_additional_candidate_locations()
        plotf_vector_triangulated(self.knowledge.grid, self.knowledge.cost_vr, "COST_{VR}", cmap=CMAP,
                                  cbar_title="Cost",
                                  colorbar=True, vmin=-.2, vmax=1.2, knowledge=self.knowledge, stepsize=.1)
        # plotf_trajectory(self.trajectory)
        trajectory = np.array(self.knowledge.sampling_location_plot)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'k.-')

        ax = fig.add_subplot(gs[4])
        # self.rrtstar.plot_tree()
        cost_valley = self.knowledge.cost_valley
        self.plot_additional_candidate_locations()
        # cost_valley[cost_valley == np.inf] = PENALTY # To avoid plotting interpolation
        plt.scatter(self.knowledge.grid[:, 0], self.knowledge.grid[:, 1], c=self.knowledge.cost_valley, cmap=CMAP,
                    vmin=0, vmax=4, alpha=.5, s=105)
        plt.colorbar()
        plt.xlim([np.amin(self.knowledge.grid[:, 0]), np.amax(self.knowledge.grid[:, 0])])
        plt.ylim([np.amin(self.knowledge.grid[:, 1]), np.amax(self.knowledge.grid[:, 1])])
        # plotf_vector_triangulated(self.knowledge.grid, cost_valley, "COST VALLEY", cmap=CMAP, vmin=-.2, vmax=4,
        #                           cbar_title="Cost",
        #                           colorbar=True, knowledge=self.knowledge, stepsize=.1)
        trajectory = np.array(self.knowledge.sampling_location_plot)
        plt.plot(trajectory[:, 0], trajectory[:, 1], 'k.-')

        # plt.colorbar()
        plt.title("COST_{Valley}")
        # plt.savefig(foldername + "P_{:03d}.png".format(i))
        # plt.close("all")

        plt.savefig(filename)
        plt.close('all')
        pass

    def plot_additional_candidate_locations(self):
        plt.plot(self.knowledge.grid[:, 0], self.knowledge.grid[:, 1], 'k.', alpha=.05)
        # plt.plot(self.current_location.x, self.current_location.y, 'r.')
        plt.plot(self.knowledge.grid[self.knowledge.ind_now, 0],
                 self.knowledge.grid[self.knowledge.ind_now, 1], 'r.')
        plt.plot(self.knowledge.grid[self.knowledge.ind_cand, 0],
                 self.knowledge.grid[self.knowledge.ind_cand, 1], 'b.')
        plt.plot(self.knowledge.grid[self.knowledge.ind_next, 0],
                 self.knowledge.grid[self.knowledge.ind_next, 1], 'y.')
        if self.knowledge.budget_middle_location:
            ellipse = Ellipse(xy=(self.knowledge.budget_middle_location.X_START, self.knowledge.budget_middle_location.Y_START),
                              width=self.knowledge.budget_ellipse_a, height=self.knowledge.budget_ellipse_b,
                              angle=math.degrees(self.knowledge.budget_ellipse_angle),
                              edgecolor='r', fc='None', lw=2)
            plt.gca().add_patch(ellipse)
        pass

    def run_lawn_mower(self):
        self.lawnmower_trajectory = np.array(self.lawnmower2d.lawnmower_refined_trajectory)
        self.starting_location = Location(self.lawnmower_trajectory[0, 0], self.lawnmower_trajectory[0, 1])
        self.current_location = self.previous_location = self.starting_location
        ind_start = get_ind_at_location2d_xy(self.knowledge.grid, self.starting_location)
        self.knowledge.ind_prev = self.knowledge.ind_now = ind_start
        # plt.plot(np.array(self.lawnmower2d.lawnmower_trajectory)[:, 0], np.array(self.lawnmower2d.lawnmower_trajectory)[:, 1], 'b.-')
        # plt.plot(self.lawnmower_trajectory[:, 0], self.lawnmower_trajectory[:, 1], 'r.-')
        # plt.show()

        foldername = PATH_REPLICATES + "R_{:03d}/Lawnmower/".format(self.seed)
        checkfolder(foldername)

        for i in range(self.steps):
        # for i in range(len(self.lawnmower_trajectory) - 1):
            print("Step No. ", i)
            self.gp.get_cost_valley(current_loc=self.current_location, previous_loc=self.previous_location,
                                    goal_loc=self.goal_location, budget=self.budget)

            self.distance_travelled = get_distance_between_locations(self.current_location, self.previous_location)
            self.budget = self.budget - self.distance_travelled
            print("Budget left: ", self.budget)
            print("Distance travelled: ", self.distance_travelled)

            self.next_location = Location(self.lawnmower_trajectory[i, 0], self.lawnmower_trajectory[i, 1])
            ind_sample = get_ind_at_location2d_xy(self.knowledge.xyz, self.next_location)

            self.knowledge.step_no = i
            self.knowledge = Sampler(self.knowledge, self.knowledge.mu_truth, ind_sample).Knowledge

            if not self.replicates:
                self.plot_2d(foldername, i)

            self.previous_location = self.current_location
            self.current_location = self.next_location

    def run_rrtstar(self):
        foldername = PATH_REPLICATES + "R_{:03d}/rrtstar/".format(self.seed)
        checkfolder(foldername)

        self.current_location = self.starting_location
        self.previous_location = self.current_location  # TODO: dot product will be zero, no effect on the first location.
        self.trajectory.append(self.current_location)
        self.gp.get_cost_valley(self.current_location, self.previous_location, self.goal_location, self.budget)

        ind_min_cost = np.argmin(self.gp.cost_valley)
        ending_loc = self.get_location_from_ind(ind_min_cost)

        self.ind_sample = get_ind_at_location2d_xy(self.knowledge.grid, self.current_location)

        for i in range(self.steps):
        # for i in range(NUM_STEPS):
            print("Step: ", i)
            t1 = time.time()
            self.knowledge.current_location = self.current_location

            self.knowledge = Sampler(self.knowledge, self.knowledge.mu_truth, self.ind_sample).Knowledge

            if not self.knowledge.gohome:
                self.knowledge.ending_location = ending_loc
                self.knowledge.starting_location = self.current_location
                self.rrtstar = RRTStar(self.knowledge)
                self.next_location = self.rrtstar.get_next_waypoint()
                t2 = time.time()
                print("Path planning takes: ", t2 - t1)
                if not self.replicates:
                    self.plot_knowledge(i, foldername)
            else:
                if self.waypoint_return_counter == 0:
                    print("Compute route home only here once!")
                    self.waypoints = self.get_route_home()
                if self.waypoint_return_counter < self.num_waypoints_return_home:
                    self.next_location = self.waypoints[self.waypoint_return_counter]
                    self.waypoint_return_counter += 1
                    if not self.replicates:
                        self.plot_knowledge(i, foldername)
                else:
                    print("Home already! Mission complete")
                    if not self.replicates:
                        self.plot_knowledge(i, foldername)
                    break

            self.distance_travelled = get_distance_between_locations(self.current_location, self.next_location)
            self.budget = self.budget - self.distance_travelled

            self.current_location = self.next_location

            print("Budget left: ", self.budget)
            print("Distance travelled: ", self.distance_travelled)

            self.ind_sample = get_ind_at_location2d_xy(self.knowledge.xyz, self.current_location)
            self.gp.get_cost_valley(self.current_location, self.previous_location, self.goal_location, self.budget)
            ind_min_cost = np.argmin(self.knowledge.cost_valley)
            ending_loc = self.get_location_from_ind(ind_min_cost)

            self.previous_location = self.current_location
            self.trajectory.append(self.current_location)

    def get_route_home(self, stepsize=None):
        distance_remaining = get_distance_between_locations(self.current_location, self.goal_location)
        angle = np.math.atan2(self.goal_location.y - self.current_location.y,
                              self.goal_location.x - self.current_location.x)
        gaps = np.arange(0, distance_remaining, stepsize)
        self.num_waypoints_return_home = len(gaps)
        distance_gaps = np.linspace(0, distance_remaining, self.num_waypoints_return_home)
        waypoints_location = []
        print("distance gaps:", distance_gaps)
        for i in range(self.num_waypoints_return_home):
            x = self.current_location.x + distance_gaps[i] * np.cos(angle)
            y = self.current_location.y + distance_gaps[i] * np.sin(angle)
            waypoints_location.append(Location(x, y))
        return waypoints_location

    def get_location_from_ind(self, ind):
        return Location(self.knowledge.grid[ind, 0], self.knowledge.grid[ind, 1])

    def is_arrived(self, current_loc):
        if get_distance_between_locations(current_loc, self.goal_location) <= DISTANCE_TOLERANCE:
            return True
        else:
            return False

    def plot_knowledge(self, fig_index, foldername):
        # == plotting ==
        fig = plt.figure(figsize=(60, 8))
        gs = GridSpec(nrows=1, ncols=5)
        ax = fig.add_subplot(gs[0])
        plotf_vector_triangulated(self.knowledge.grid, self.knowledge.mu_truth, "Ground Truth", cmap=CMAP, vmin=-.2, vmax=1.2,
                     cbar_title="Salinity", knowledge=self.knowledge, stepsize=.1, threshold=self.knowledge.threshold)

        ax = fig.add_subplot(gs[1])
        plotf_vector_triangulated(self.knowledge.grid, self.knowledge.mu_cond, "Conditional Mean", cmap=CMAP, vmin=-.2, vmax=1.2,
                     cbar_title="Salinity", knowledge=self.knowledge, stepsize=.1, threshold=self.knowledge.threshold)
        plotf_trajectory(self.trajectory)

        ax = fig.add_subplot(gs[2])
        plotf_vector_triangulated(self.knowledge.grid, self.knowledge.cost_eibv, "COST_{EIBV}", cmap=CMAP, cbar_title="Cost",
                     colorbar=True, vmin=-.2, vmax=1.2, knowledge=self.knowledge, stepsize=.1)
        plotf_trajectory(self.trajectory)

        ax = fig.add_subplot(gs[3])
        plotf_vector_triangulated(self.knowledge.grid, self.knowledge.cost_vr, "COST_{VR}", cmap=CMAP, cbar_title="Cost",
                     colorbar=True, vmin=-.2, vmax=1.2, knowledge=self.knowledge, stepsize=.1)
        plotf_trajectory(self.trajectory)

        ax = fig.add_subplot(gs[4])
        self.rrtstar.plot_tree()
        cost_valley = self.knowledge.cost_valley
        # cost_valley[cost_valley == np.inf] = PENALTY # To avoid plotting interpolation
        # plotf_vector_triangulated(self.knowledge.grid, cost_valley, "COST VALLEY", cmap=CMAP, vmin=-.2, vmax=4, cbar_title="Cost",
        #              colorbar=True, knowledge=self.knowledge, stepsize=.1)

        plt.scatter(self.knowledge.grid[:, 1],
                    self.knowledge.grid[:, 0],
                    c=self.knowledge.cost_valley, cmap=CMAP,
                    vmin=0, vmax=4, alpha=.6, s=105)
        plt.plot(self.knowledge.polygon_border[:, 0], self.knowledge.polygon_border[:, 1], 'k-', linewidth=1)
        for i in range(len(self.knowledge.polygon_obstacles)):
            plt.plot(self.knowledge.polygon_obstacles[:, 0], self.knowledge.polygon_obstacles[:, 1], 'k-', linewidth=1)
        plt.plot(self.knowledge.starting_location.X_START, self.knowledge.starting_location.Y_START, 'kv', ms=10)
        plt.plot(self.knowledge.goal_location.X_START, self.knowledge.goal_location.Y_START, 'rv', ms=10)
        plt.xlim([np.amin(self.knowledge.grid[:, 0]), np.amax(self.knowledge.grid[:, 0])])
        plt.ylim([np.amin(self.knowledge.grid[:, 0]), np.amax(self.knowledge.grid[:, 1])])

        plt.title("COST_{Valley}")
        plt.savefig(foldername + "P_{:03d}.png".format(fig_index))
        plt.close("all")


if __name__ == "__main__":
    a = Simulator(steps=20, random_seed=3, replicates=False)
    a.plot_synthetic_field()
    # a.run_2d()
    # a.run_lawn_mower()
    a.run_rrtstar()



