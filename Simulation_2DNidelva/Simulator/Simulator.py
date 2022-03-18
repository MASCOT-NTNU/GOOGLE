"""
This script generates the next waypoint based on the current knowledge and previous path
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-05 ~ 2022-01-08
"""


from Nidelva.Simulation.Plotter.KnowledgePlot import KnowledgePlot
from Nidelva.Simulation.Plotter.SimulationResultsPlot import SimulationResultsPlot
from Nidelva.Simulation.ES_Strategies.PathPlanner_Myopic3D import MyopicPlanning_3D
from Nidelva.Simulation.ES_Strategies.PathPlanner_Myopic2D import MyopicPlanning_2D
from Nidelva.Simulation.ES_Strategies.PathPlanner_Lawnmower import LawnMowerPlanning
from Nidelva.Simulation.ES_Strategies.Knowledge import Knowledge
from Nidelva.Simulation.Field.Data.DataInterpolator import DataInterpolator
from Nidelva.Simulation.Field.Grid.gridWithinPolygonGenerator import GridGenerator
from Nidelva.Simulation.GP_kernel.Matern_kernel import MaternKernel
from Nidelva.Simulation.Simulator.Sampler import Sampler
from Nidelva.Simulation.Simulator.SimulationResultContainer import SimulationResultContainer
from usr_func import *
import time


# ==== Field Config ====
DEPTH = [.5, 1, 1.5, 2.0, 2.5]
DISTANCE_LATERAL = 120
DISTANCE_VERTICAL = np.abs(DEPTH[1] - DEPTH[0])
DISTANCE_TOLERANCE = 1
DISTANCE_SELF = 20
THRESHOLD = 28
# ==== End Field Config ====

# ==== GP Config ====
SILL = .5
RANGE_LATERAL = 550
RANGE_VERTICAL = 2
NUGGET = .04
# ==== End GP Config ====

# ==== Plot Config ======
VMIN = 16
VMAX = 30
# ==== End Plot Config ==


class Simulator:

    knowledge = None

    def __init__(self, steps=10, random_seed=0):
        # print("Random seed: ", random_seed)
        self.seed = random_seed
        np.random.seed(self.seed)
        self.steps = steps
        self.simulation_config()
        # self.save_benchmark_figure()

    def simulation_config(self):
        t1 = time.time()
        self.polygon = np.array([[6.344800000000000040e+01, 1.040000000000000036e+01],
                                [6.344800000000000040e+01, 1.041999999999999993e+01],
                                [6.346000000000000085e+01, 1.041999999999999993e+01],
                                [6.346000000000000085e+01, 1.040000000000000036e+01]])
        gridGenerator = GridGenerator(polygon=self.polygon, depth=DEPTH, distance_neighbour=DISTANCE_LATERAL, no_children=6)
        # grid = gridGenerator.grid
        coordinates = gridGenerator.coordinates
        data_interpolator = DataInterpolator(coordinates=coordinates)
        mu_prior = vectorise(data_interpolator.dataset_interpolated["salinity"])
        matern_kernel = MaternKernel(coordinates=coordinates, sill=SILL, range_lateral=RANGE_LATERAL,
                                     range_vertical=RANGE_VERTICAL, nugget=NUGGET)
        self.knowledge = Knowledge(coordinates=coordinates, polygon=self.polygon, mu=mu_prior, Sigma=matern_kernel.Sigma,
                                   threshold_salinity=THRESHOLD, kernel=matern_kernel, ind_prev=[], ind_now=[],
                                   distance_lateral=DISTANCE_LATERAL, distance_vertical=DISTANCE_VERTICAL,
                                   distance_tolerance=DISTANCE_TOLERANCE, distance_self=DISTANCE_SELF)

        self.ground_truth = np.linalg.cholesky(self.knowledge.Sigma) @ \
                            vectorise(np.random.randn(self.knowledge.coordinates.shape[0])) + self.knowledge.mu
        LawnMowerPlanningSetup = LawnMowerPlanning(knowledge=self.knowledge)
        LawnMowerPlanningSetup.build_3d_lawn_mower()
        self.lawn_mower_path_3d = LawnMowerPlanningSetup.lawn_mower_path_3d
        self.starting_index = 30
        t2 = time.time()
        print("Simulation config is done, time consumed: ", t2 - t1)

    def save_benchmark_figure(self):
        foldername = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/Replicates/R_{:03d}/".format(self.seed)
        checkfolder(foldername)
        self.knowledge_prior = self.knowledge
        self.knowledge_prior.excursion_prob = get_excursion_prob_1d(self.knowledge_prior.mu, self.knowledge_prior.Sigma, self.knowledge_prior.threshold_salinity)
        KnowledgePlot(knowledge=self.knowledge_prior, vmin=VMIN, vmax=VMAX, filename=foldername+"Field_prior", html=False)
        self.knowledge_ground_truth = self.knowledge
        self.knowledge_ground_truth.mu = self.ground_truth
        self.knowledge_ground_truth.excursion_prob = get_excursion_prob_1d(self.knowledge_ground_truth.mu, self.knowledge_ground_truth.Sigma, self.knowledge_ground_truth.threshold_salinity)
        KnowledgePlot(knowledge=self.knowledge_ground_truth, vmin=VMIN, vmax=VMAX, filename=foldername+"Field_ground_truth", html=False)

    def run_2d(self):
        self.starting_loc = self.lawn_mower_path_3d[self.starting_index, :]
        self.ind_start = get_grid_ind_at_nearest_loc(self.starting_loc, self.knowledge.coordinates) # get nearest neighbour
        self.knowledge.ind_prev = self.knowledge.ind_now = self.ind_sample = self.ind_start

        foldername = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/Replicates/R_{:03d}/2D/".format(self.seed)
        checkfolder(foldername)
        for i in range(self.steps):
            # print("Step No. ", i)
            self.knowledge = Sampler(self.knowledge, self.ground_truth, self.ind_sample).Knowledge
            lat_next, lon_next, depth_next = MyopicPlanning_2D(knowledge=self.knowledge).next_waypoint
            self.ind_sample = get_grid_ind_at_nearest_loc([lat_next, lon_next, depth_next], self.knowledge.coordinates)
            self.knowledge.step_no = i
        KnowledgePlot(knowledge=self.knowledge, vmin=VMIN, vmax=VMAX, filename=foldername + "Field_{:03d}".format(i),
                      html=False)

    def run_3d(self):
        self.starting_loc = self.lawn_mower_path_3d[self.starting_index, :]
        self.ind_start = get_grid_ind_at_nearest_loc(self.starting_loc, self.knowledge.coordinates) # get nearest neighbour
        self.knowledge.ind_prev = self.knowledge.ind_now = self.ind_sample = self.ind_start

        foldername = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/Replicates/R_{:03d}/3D/".format(self.seed)
        checkfolder(foldername)
        for i in range(self.steps):
            # print("Step No. ", i)
            self.knowledge = Sampler(self.knowledge, self.ground_truth, self.ind_sample).Knowledge
            lat_next, lon_next, depth_next = MyopicPlanning_3D(knowledge=self.knowledge).next_waypoint
            self.ind_sample = get_grid_ind_at_nearest_loc([lat_next, lon_next, depth_next], self.knowledge.coordinates)
            self.knowledge.step_no = i
            # KnowledgePlot(knowledge=self.knowledge, vmin=VMIN, vmax=VMAX,
            #               filename=foldername + "Field_{:03d}".format(i), html=False)
        KnowledgePlot(knowledge=self.knowledge, vmin=VMIN, vmax=VMAX, filename=foldername + "Field_{:03d}".format(i), html=False)

    def run_lawn_mower(self):
        lat_start, lon_start, depth_start = self.lawn_mower_path_3d[self.starting_index, :]
        ind_start = get_grid_ind_at_nearest_loc([lat_start, lon_start, depth_start], self.knowledge.coordinates)
        self.knowledge.ind_prev = self.knowledge.ind_now = ind_start

        foldername = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/Replicates/R_{:03d}/Lawnmower/".format(self.seed)
        checkfolder(foldername)

        for i in range(self.steps):
            # print("Step No. ", i)
            lat_next, lon_next, depth_next = self.lawn_mower_path_3d[self.starting_index + i, :]
            ind_sample = get_grid_ind_at_nearest_loc([lat_next, lon_next, depth_next], self.knowledge.coordinates)

            self.knowledge.step_no = i
            self.knowledge = Sampler(self.knowledge, self.ground_truth, ind_sample).Knowledge
        KnowledgePlot(knowledge=self.knowledge, vmin=VMIN, vmax=VMAX, filename=foldername+"Field_{:03d}".format(i), html=False)


a = Simulator(steps=10, random_seed=0)
a.run_lawn_mower()

