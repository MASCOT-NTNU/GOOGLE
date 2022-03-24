"""
This script builds the kernel for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""
from usr_func import *
from GOOGLE.Simulation_2DSquare.Config.Config import *
from GOOGLE.Simulation_2DSquare.Tree.Location import *


class GPKernel:

    def __init__(self, knowledge=None):
        self.knowledge = knowledge
        self.grid = self.knowledge.grid
        self.x_vector = vectorise(self.grid[:, 0])
        self.y_vector = vectorise(self.grid[:, 1])
        self.mu_prior = self.knowledge.mu_prior
        self.polygon_obstacles_shapely = self.knowledge.polygon_obstacles_shapely
        self.get_Sigma_prior()
        self.get_ground_truth()
        self.knowledge.mu_cond = self.knowledge.mu_prior
        self.knowledge.Sigma_cond = self.knowledge.Sigma_prior
        self.get_obstacle_field()
        self.cost_direction = None
        self.cost_vr = None
        self.cost_eibv = None
        self.cost_budget = None
        self.cost_valley = None
        self.budget_middle_location = None
        self.budget_ellipse_angle = None
        self.budget_ellipse_a = None
        self.budget_ellipse_b = None
        self.budget_ellipse_c = None

    def get_Sigma_prior(self):
        self.set_coef()
        DistanceMatrix = cdist(self.grid, self.grid)
        self.knowledge.Sigma_prior = self.sigma ** 2 * (1 + self.eta * DistanceMatrix) * np.exp(-self.eta * DistanceMatrix)

    def set_coef(self):
        self.sigma = SIGMA
        self.eta = 4.5 / LATERAL_RANGE
        self.tau = np.sqrt(NUGGET)
        self.R = np.diagflat(self.tau ** 2)
        print("Coef is set successfully!")

    def get_ground_truth(self):
        self.mu_truth = (self.knowledge.mu_prior +
                         np.linalg.cholesky(self.knowledge.Sigma_prior) @
                         np.random.randn(len(self.knowledge.mu_prior)).reshape(-1, 1))

    def get_cost_valley(self, current_loc=None, previous_loc=None, goal_loc=None, budget=None):
        t1 = time.time()
        self.get_eibv_field()
        self.get_variance_reduction_field()
        self.get_direction_field(current_loc, previous_loc)
        self.get_budget_field(current_loc, goal_loc, budget)
        self.cost_valley = (self.cost_eibv +
                            self.cost_budget +
                            self.cost_vr +
                            self.cost_obstacle +
                            self.cost_direction)
        self.save_information_to_knowledge()
        t2 = time.time()
        print("Cost valley computed successfully!, time consumed: ", t2 - t1)

    def save_information_to_knowledge(self):
        self.knowledge.cost_eibv = self.cost_eibv
        self.knowledge.cost_vr = self.cost_vr
        self.knowledge.cost_budget = self.cost_budget
        self.knowledge.cost_direction = self.cost_direction
        self.knowledge.cost_obstacle = self.cost_obstacle
        self.knowledge.cost_valley = self.cost_valley
        self.knowledge.budget_middle_location = self.budget_middle_location
        self.knowledge.budget_ellipse_angle = self.budget_ellipse_angle
        self.knowledge.budget_ellipse_a = self.budget_ellipse_a
        self.knowledge.budget_ellipse_b = self.budget_ellipse_b
        self.knowledge.budget_ellipse_c = self.budget_ellipse_c

    def get_eibv_field(self):
        t1 = time.time()
        self.cost_eibv = []
        for i in range(self.grid.shape[0]):
            F = np.zeros([1, self.grid.shape[0]])
            F[0, i] = True
            self.cost_eibv.append(get_eibv_1d(self.knowledge.threshold, self.knowledge.mu_cond,
                                              self.knowledge.Sigma_cond, F, self.R))
        self.cost_eibv = normalise(np.array(self.cost_eibv))
        t2 = time.time()
        print("EIBV field takes: ", t2 - t1)

    def get_budget_field(self, current_loc, goal_loc, budget):
        t1 = time.time()
        if budget >= BUDGET_MARGIN:
            self.budget_middle_location = self.get_middle_location(current_loc, goal_loc)
            self.budget_ellipse_angle = self.get_angle_between_locations(current_loc, goal_loc)
            self.budget_ellipse_a = budget / 2
            self.budget_ellipse_c = get_distance_between_locations(current_loc, goal_loc) / 2
            self.budget_ellipse_b = np.sqrt(self.budget_ellipse_a ** 2 - self.budget_ellipse_c ** 2)
            print("a: ", self.budget_ellipse_a, "b: ", self.budget_ellipse_b, "c: ", self.budget_ellipse_c)
            if self.budget_ellipse_b > BUDGET_ELLIPSE_B_MARGIN:
                x_wgs = self.x_vector - self.budget_middle_location.x
                y_wgs = self.y_vector - self.budget_middle_location.y
                self.cost_budget = []
                for i in range(len(self.grid)):
                    x_usr = (x_wgs[i] * np.cos(self.budget_ellipse_angle) +
                             y_wgs[i] * np.sin(self.budget_ellipse_angle))
                    y_usr = (- x_wgs[i] * np.sin(self.budget_ellipse_angle) +
                             y_wgs[i] * np.cos(self.budget_ellipse_angle))
                    if (x_usr / self.budget_ellipse_a) ** 2 + (y_usr / self.budget_ellipse_b) ** 2 <= 1:
                        self.cost_budget.append(0)
                    else:
                        self.cost_budget.append(np.inf)
                self.cost_budget = np.array(self.cost_budget)
            else:
                self.knowledge.gohome = True
        else:
            self.knowledge.gohome = True
        t2 = time.time()
        print("budget field consumed: ", t2 - t1)

    @staticmethod
    def get_middle_location(location1, location2):
        x_middle = (location1.x + location2.x) / 2
        y_middle = (location1.y + location2.y) / 2
        return Location(x_middle, y_middle)

    @staticmethod
    def get_angle_between_locations(location1, location2):
        delta_y = location2.y - location1.y
        delta_x = location2.x - location1.x
        angle = np.math.atan2(delta_y, delta_x)
        return angle

    def get_variance_reduction_field(self):
        t1 = time.time()
        self.cost_vr = []
        for i in range(len(self.grid)):
            ind_F = get_ind_at_location2d_xy(self.grid, Location(self.grid[i, 0], self.grid[i, 1]))
            F = np.zeros([1, self.grid.shape[0]])
            F[0, ind_F] = True
            self.cost_vr.append(self.get_variance_reduction(self.knowledge.Sigma_cond, F, self.R))
        self.cost_vr = 1 - normalise(np.array(self.cost_vr))
        t2 = time.time()
        print("Variance Reduction field takes: ", t2 - t1)

    @staticmethod
    def get_variance_reduction(Sigma, F, R):
        Reduction = Sigma @ F.T @ np.linalg.solve(F @ Sigma @ F.T + R, F @ Sigma)
        vr = np.sum(np.diag(Reduction))
        return vr

    def get_obstacle_field(self):
        self.cost_obstacle = []
        for i in range(len(self.grid)):
            if self.is_within_obstacles(Location(self.grid[i, 0], self.grid[i, 1])):
                self.cost_obstacle.append(np.inf)
            else:
                self.cost_obstacle.append(0)
        self.cost_obstacle = np.array(self.cost_obstacle)

    def is_within_obstacles(self, location):
        point = Point(location.x, location.y)
        within = False
        for i in range(len(self.polygon_obstacles_shapely)):
            if self.polygon_obstacles_shapely[i].contains(point):
                within = True
        return within

    def get_direction_field(self, current_loc, previous_loc):
        t1 = time.time()
        dx = current_loc.x - previous_loc.x
        dy = current_loc.y - previous_loc.y
        vec1 = np.array([[dx, dy]])
        self.cost_direction = []
        for i in range(len(self.grid)):
            dx = self.grid[i, 0] - current_loc.x
            dy = self.grid[i, 1] - current_loc.y
            vec2 = np.array([[dx, dy]])
            if np.dot(vec1, vec2.T) >= 0:
                self.cost_direction.append(0)
            else:
                self.cost_direction.append(PENALTY)
        self.cost_direction = np.array(self.cost_direction)
        t2 = time.time()
        print("Direction field takes: ", t2 - t1)



