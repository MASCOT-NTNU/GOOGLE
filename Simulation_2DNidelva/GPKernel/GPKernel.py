"""
This script builds the kernel for simulation
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""

from usr_func import *
from GOOGLE.Simulation_2DNidelva.Tree.Location import *


class GPKernel:

    gohome = False

    def __init__(self):
        self.threshold = THRESHOLD

        self.load_grid_and_data()
        self.get_Sigma_prior()
        self.get_ground_truth()

        self.mu_cond = self.mu_prior
        self.Sigma_cond = self.Sigma_prior

        self.get_obstacle_field()

    def load_grid_and_data(self):
        self.dataset = pd.read_csv(PATH_DATA).to_numpy()
        self.coordinates = self.dataset[:, 0:3]
        self.mu_prior = vectorise(self.dataset[:, -1])
        self.polygon_border = pd.read_csv(PATH_BORDER).to_numpy()
        self.polygon_obstacle = pd.read_csv(PATH_OBSTACLE).to_numpy()

        # == setup border & obstacle
        self.polygon_obstacle_path = Polygon(self.polygon_obstacle)
        self.polygon_border_path = Polygon(self.polygon_border)
        self.borderline_path = LineString(self.polygon_border)

    def get_Sigma_prior(self):
        self.set_coef()
        self.wgs2xy()
        DistanceMatrix = cdist(self.grid_xy, self.grid_xy)
        self.Sigma_prior = self.sigma ** 2 * (1 + self.eta * DistanceMatrix) * np.exp(-self.eta * DistanceMatrix)

    def set_coef(self):
        self.sigma = SIGMA
        self.eta = 4.5 / LATERAL_RANGE
        self.tau = NUGGET
        self.R = np.diagflat(self.tau ** 2)
        print("Coef is set successfully!")

    def wgs2xy(self):
        x, y = latlon2xy(self.coordinates[:, 0], self.coordinates[:, 1], LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        self.x_vector, self.y_vector = map(vectorise, [x, y])
        self.grid_xy = np.hstack((vectorise(x), vectorise(y)))

    def get_ground_truth(self):
        self.mu_truth = (self.mu_prior.reshape(-1, 1) +
                         np.linalg.cholesky(self.Sigma_prior) @
                         np.random.randn(len(self.mu_prior)).reshape(-1, 1))

    def get_ind_F(self, location):
        x, y = map(vectorise, [location.x, location.y])
        DM_x = x @ np.ones([1, len(self.x_vector)]) - np.ones([len(x), 1]) @ self.x_vector.T
        DM_y = y @ np.ones([1, len(self.y_vector)]) - np.ones([len(y), 1]) @ self.y_vector.T
        DM = DM_x ** 2 + DM_y ** 2
        ind_F = np.argmin(DM, axis = 1)
        return ind_F

    def get_eibv_field(self):
        t1 = time.time()
        self.cost_eibv = []
        for i in range(self.grid_xy.shape[0]):
            F = getFVector(i, self.grid_xy.shape[0])
            self.cost_eibv.append(get_eibv_1d(self.threshold, self.mu_cond, self.Sigma_cond, F, self.R))
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
            self.budget_ellipse_b = np.sqrt(self.budget_ellipse_a ** 2 - self.budget_ellipse_c ** 2)
            print("a: ", self.budget_ellipse_a, "b: ", self.budget_ellipse_b, "c: ", self.budget_ellipse_c)
            if self.budget_ellipse_b > BUDGET_ELLIPSE_B_MARGIN:
                x_wgs = self.x_vector - self.budget_middle_location.x
                y_wgs = self.y_vector - self.budget_middle_location.y
                self.cost_budget = []
                for i in range(len(self.grid_xy)):
                    x_usr = (x_wgs[i] * np.cos(self.budget_ellipse_angle) -
                             y_wgs[i] * np.sin(self.budget_ellipse_angle))
                    y_usr = (x_wgs[i] * np.sin(self.budget_ellipse_angle) +
                             y_wgs[i] * np.cos(self.budget_ellipse_angle))

                    if (x_usr / self.budget_ellipse_b) ** 2 + (y_usr / self.budget_ellipse_a) ** 2 <= 1:
                        self.cost_budget.append(0)
                    else:
                        self.cost_budget.append(np.inf)
                self.cost_budget = np.array(self.cost_budget)
            else:
                self.gohome = True
        else:
            self.gohome = True

        t2 = time.time()
        print("budget field consumed: ", t2 - t1)

    @staticmethod
    def get_middle_location(location1, location2):
        x_middle = (location1.x + location2.x) / 2
        y_middle = (location1.y + location2.y) / 2
        lat_middle, lon_middle = xy2latlon(x_middle, y_middle, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        return Location(lat_middle, lon_middle)

    @staticmethod
    def get_angle_between_locations(location1, location2):
        delta_y = location2.y - location1.y
        delta_x = location2.x - location1.x
        angle = np.math.atan2(delta_x, delta_y)
        return angle

    def get_variance_reduction_field(self):
        t1 = time.time()
        self.cost_vr = []
        for i in range(len(self.coordinates)):
            ind_F = self.get_ind_F(Location(self.coordinates[i, 0], self.coordinates[i, 1]))
            F = np.zeros([1, self.grid_xy.shape[0]])
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

    # TODO: finish obstacle field
    def get_obstacle_field(self):
        self.cost_obstacle = []
        for i in range(len(self.coordinates)):
            if self.is_within_obstacles(Location(self.coordinates[i, 0], self.coordinates[i, 1])):
                self.cost_obstacle.append(np.inf)
            else:
                self.cost_obstacle.append(0)
        self.cost_obstacle = np.array(self.cost_obstacle)

    def is_within_obstacles(self, location):
        point = Point(location.lat, location.lon)
        within = False
        if self.polygon_obstacle_path.contains(point):
            within = True
        return within

    def get_direction_field(self, current_loc, previous_loc):
        t1 = time.time()
        dx = current_loc.x - previous_loc.x
        dy = current_loc.y - previous_loc.y
        vec1 = np.array([[dx, dy]])
        self.cost_direction = []
        for i in range(len(self.grid_xy)):
            dx = self.grid_xy[i, 0] - current_loc.x
            dy = self.grid_xy[i, 1] - current_loc.y
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







