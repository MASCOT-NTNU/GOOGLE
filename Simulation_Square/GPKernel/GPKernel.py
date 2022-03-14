"""
This script builds the kernel for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""
from usr_func import *
from GOOGLE.Simulation_Square.Config.Config import *
from GOOGLE.Simulation_Square.Tree.Location import *


class GPKernel:

    def __init__(self):
        self.get_grid()
        self.get_mean_field()
        self.get_Sigma()
        self.get_ground_truth()

        self.mu_cond = self.mu_prior_vector
        self.Sigma_cond = self.Sigma_prior

        self.get_obstacle_field()

    def get_grid(self):
        self.x = np.linspace(XLIM[0], XLIM[1], NX)
        self.y = np.linspace(YLIM[0], YLIM[1], NY)
        self.x_matrix, self.y_matrix = np.meshgrid(self.x, self.y)
        self.grid_vector = []
        for i in range(self.x_matrix.shape[0]):
            for j in range(self.x_matrix.shape[1]):
                self.grid_vector.append([self.x_matrix[i, j], self.y_matrix[i, j]])
        self.grid_vector = np.array(self.grid_vector)
        self.x_vector = self.grid_vector[:, 0].reshape(-1, 1)
        self.y_vector = self.grid_vector[:, 1].reshape(-1, 1)
        self.num_nodes = len(self.grid_vector)
        print("Grid is built successfully!")

    def set_coef(self):
        self.sigma = SIGMA
        self.eta = 4.5 / LATERAL_RANGE
        self.tau = NUGGET
        self.R = np.diagflat(self.tau ** 2)
        self.threshold = THRESHOLD
        print("Coef is set successfully!")

    @staticmethod
    def get_prior(x, y):
        return (1 - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .07))
                # 1 - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .05))
                # 1 - np.exp(- ((x - .5) ** 2 + (y - .0) ** 2) / .004) +
                # 1 - np.exp(- ((x - .99) ** 2 + (y - .1) ** 2) / .1))

    def get_ind_F(self, location):
        x, y = map(vectorise, [location.x, location.y])
        DM_x = x @ np.ones([1, len(self.x_vector)]) - np.ones([len(x), 1]) @ self.x_vector.T
        DM_y = y @ np.ones([1, len(self.y_vector)]) - np.ones([len(y), 1]) @ self.y_vector.T
        DM = DM_x ** 2 + DM_y ** 2
        ind_F = np.argmin(DM, axis = 1)
        return ind_F

    def get_mean_field(self):
        self.mu_prior_vector = vectorise(self.get_prior(self.x_vector, self.y_vector))
        self.mu_prior_matrix = self.get_prior(self.x_matrix, self.y_matrix)

    def get_Sigma(self):
        self.set_coef()
        DistanceMatrix = cdist(self.grid_vector, self.grid_vector)
        self.Sigma_prior = self.sigma ** 2 * (1 + self.eta * DistanceMatrix) * np.exp(-self.eta * DistanceMatrix)

    def get_ground_truth(self):
        self.mu_truth = (self.mu_prior_vector.reshape(-1, 1) +
                         np.linalg.cholesky(self.Sigma_prior) @
                         np.random.randn(len(self.mu_prior_vector)).reshape(-1, 1))

    @staticmethod
    def update_GP_field(mu, Sigma, F, R, measurement):
        C = F @ Sigma @ F.T + R
        mu = mu + Sigma @ F.T @ np.linalg.solve(C, (measurement - F @ mu))
        Sigma = Sigma - Sigma @ F.T @ np.linalg.solve(C, F @ Sigma)
        return mu, Sigma

    @staticmethod
    def get_eibv(mu, Sigma, F, R, threshold):
        Sigma_updated = Sigma - Sigma @ F.T @ np.linalg.solve(F @ Sigma @ F.T + R, F @ Sigma)
        Variance = np.diag(Sigma_updated).reshape(-1, 1)
        EIBV = 0
        for i in range(mu.shape[0]):
            EIBV += (mvn.mvnun(-np.inf, threshold, mu[i], Variance[i])[0] -
                     mvn.mvnun(-np.inf, threshold, mu[i], Variance[i])[0] ** 2)
        return EIBV

    def get_eibv_field(self):
        t1 = time.time()
        self.cost_eibv = []
        for i in range(self.grid_vector.shape[0]):
            F = np.zeros([1, self.grid_vector.shape[0]])
            F[0, i] = True
            self.cost_eibv.append(self.get_eibv(self.mu_cond, self.Sigma_cond, F, self.R, THRESHOLD))
        self.cost_eibv = normalise(np.array(self.cost_eibv))
        t2 = time.time()
        print("EIBV field takes: ", t2 - t1)

    # TODO: check multiprocessing
    # def get_eibv_field_mp(self):
    #     t1 = time.time()
    #     ind = np.arange(self.grid_vector.shape[0])
    #     pool = Pool(cpu_count())
    #     pool.map(self.get_eibv_mp, ind)
    #     t2 = time.time()
    #     print("MP EIBV field takes: ", t2 - t1)
    #
    # def get_eibv_mp(self, ind):
    #     F = np.zeros([1, self.grid_vector.shape[0]])
    #     F[0, ind] = True
    #     Sigma_updated = self.Sigma_cond - self.Sigma_cond @ F.T @ np.linalg.solve(F @ self.Sigma_cond @ F.T + self.R,
    #                                                                               F @ self.Sigma_cond)
    #     Variance = np.diag(Sigma_updated).reshape(-1, 1)
    #     EIBV = 0
    #     for i in range(self.mu_cond.shape[0]):
    #         EIBV += (mvn.mvnun(-np.inf, THRESHOLD, self.mu_cond[i], Variance[i])[0] -
    #                  mvn.mvnun(-np.inf, THRESHOLD, self.mu_cond[i], Variance[i])[0] ** 2)

    def get_budget_field(self, current_loc, goal_loc, budget):
        t1 = time.time()
        if budget >= BUDGET_MARGIN:
            self.budget_middle_location = self.get_middle_location(current_loc, goal_loc)
            self.budget_ellipse_angle = self.get_angle_between_locations(current_loc, goal_loc)

            self.budget_ellipse_a = budget / 2
            self.budget_ellipse_c = get_distance_between_locations(current_loc, goal_loc) / 2
            if self.budget_ellipse_a > self.budget_ellipse_c:
                self.budget_ellipse_b = np.sqrt(self.budget_ellipse_a ** 2 - self.budget_ellipse_c ** 2)
                print("a: ", self.budget_ellipse_a, "b: ", self.budget_ellipse_b, "c: ", self.budget_ellipse_c)

                x_wgs = self.x_vector - self.budget_middle_location.x
                y_wgs = self.y_vector - self.budget_middle_location.y
                self.cost_budget = []
                for i in range(len(self.grid_vector)):
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
                self.cost_budget = np.ones_like(self.grid_vector[:, 0]) * np.inf
                ind_goal_loc = self.get_ind_F(goal_loc)
                self.cost_budget[ind_goal_loc] = 0
        else:
            self.cost_budget = np.ones_like(self.grid_vector[:, 0]) * np.inf
            ind_goal_loc = self.get_ind_F(goal_loc)
            self.cost_budget[ind_goal_loc] = 0

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

    def get_gradient_field(self):
        self.mu_cond_matrix = np.zeros_like(self.x_matrix)
        for i in range(self.x_matrix.shape[0]):
            for j in range(self.x_matrix.shape[1]):
                self.mu_cond_matrix[i, j] = self.mu_cond[i*NY+j]
        gradient_x, gradient_y = np.gradient(self.mu_cond_matrix)
        self.gradient_matrix = np.sqrt(gradient_x ** 2 + gradient_y ** 2)
        self.gradient_vector = []
        for i in range(self.gradient_matrix.shape[0]):
            for j in range(self.gradient_matrix.shape[1]):
                self.gradient_vector.append(self.gradient_matrix[i, j])
        self.gradient_vector = np.array(self.gradient_vector)

    def get_variance_reduction_field(self):
        t1 = time.time()
        self.cost_vr = []
        for i in range(len(self.grid_vector)):
            ind_F = self.get_ind_F(Location(self.grid_vector[i, 0], self.grid_vector[i, 1]))
            F = np.zeros([1, self.grid_vector.shape[0]])
            F[0, ind_F] = True
            self.cost_vr.append(self.get_variance_reduction(self.Sigma_cond, F, self.R))
        self.cost_vr = 1 - normalise(np.array(self.cost_vr))
        t2 = time.time()
        print("Variance Reduction field takes: ", t2 - t1)

    @staticmethod
    def get_variance_reduction(Sigma, F, R):
        Reduction = Sigma @ F.T @ np.linalg.solve(F @ Sigma @ F.T + R, F @ Sigma)
        vr = np.sum(np.diag(Reduction))
        return vr

    def get_obstacle_field(self):
        self.obstacles = np.array(OBSTACLES)
        self.polygon_obstacles = []
        for i in range(self.obstacles.shape[0]):
            self.polygon_obstacles.append(Polygon(list(map(tuple, self.obstacles[i]))))

        self.cost_obstacle = []
        for i in range(len(self.grid_vector)):
            if self.is_within_obstacles(Location(self.grid_vector[i, 0], self.grid_vector[i, 1])):
                self.cost_obstacle.append(np.inf)
            else:
                self.cost_obstacle.append(0)
        self.cost_obstacle = np.array(self.cost_obstacle)

    def is_within_obstacles(self, location):
        point = Point(location.x, location.y)
        within = False
        for i in range(len(self.polygon_obstacles)):
            if self.polygon_obstacles[i].contains(point):
                within = True
        return within

    def get_direction_field(self, current_loc, previous_loc):
        t1 = time.time()
        dx = current_loc.x - previous_loc.x
        dy = current_loc.y - previous_loc.y
        vec1 = np.array([[dx, dy]])
        self.cost_direction = []
        for i in range(len(self.grid_vector)):
            dx = self.grid_vector[i, 0] - current_loc.x
            dy = self.grid_vector[i, 1] - current_loc.y
            vec2 = np.array([[dx, dy]])
            if np.dot(vec1, vec2.T) >= 0:
                self.cost_direction.append(0)
            else:
                self.cost_direction.append(PENALTY)
        self.cost_direction = np.array(self.cost_direction)
        t2 = time.time()
        print("Direction field takes: ", t2 - t1)

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
        t2 = time.time()
        print("Cost valley computed successfully!, time consumed: ", t2 - t1)


