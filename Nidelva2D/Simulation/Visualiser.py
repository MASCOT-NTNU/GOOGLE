"""
This script simulates GOOGLE
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-27
"""
import matplotlib.pyplot as plt
import numpy as np

from usr_func import *
from GOOGLE.Nidelva2D.Config.Config import *
from GOOGLE.Nidelva2D.grf_model import GRF
from GOOGLE.Nidelva2D.CostValley import CostValley
from GOOGLE.Nidelva2D.RRTStarCV import RRTStarCV, TARGET_RADIUS
from GOOGLE.Nidelva2D.RRTStarHome import RRTStarHome
from GOOGLE.Nidelva2D.StraightLinePathPlanner import StraightLinePathPlanner
from GOOGLE.Nidelva2D.grfar_model import GRFAR
import multiprocessing as mp

# == Set up
# 0, lade
LAT_START = 63.456232
LON_START = 10.435198

# 1, back munkhomen
# LAT_START = 63.449664
# LON_START = 10.363366

# 2, north munkhomen
# LAT_START = 63.46797
# LON_START = 10.39416

# 3, corner munkholmen
# LAT_START = 63.459906
# LON_START = 10.381345

# 4, skansen
# LAT_START = 63.436006
# LON_START = 10.381842

# 5, home
# LAT_START = LATITUDE_HOME
# LON_START = LONGITUDE_HOME

X_START, Y_START = latlon2xy(LAT_START, LON_START, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
NUM_STEPS = 120
# ==


class Simulator:

    def __init__(self):
        # self.load_grf_model()
        self.load_grfar_model()
        self.load_cost_valley()
        self.load_rrtstar()
        self.load_rrthome()
        self.load_straight_line_planner()
        self.gohome = False
        self.obstacle_in_the_way = True
        self.pool = mp.Pool(1)

    # def load_grf_model(self):
    #     self.grf_model = GRF()
    #     print("S1: GRF model is loaded successfully!")

    def load_grfar_model(self):
        self.grfar_model = GRFAR()
        self.grf_grid = self.grfar_model.grf_grid
        self.N_grf_grid = self.grf_grid.shape[0]
        print("S2: GRFAR model is loaded successfully!")

    def load_cost_valley(self):
        self.CV = CostValley()
        print("S3: Cost Valley is loaded successfully!")

    def load_rrtstar(self):
        self.rrtstar = RRTStarCV()
        print("S4: RRTStarCV is loaded successfully!")

    def load_rrthome(self):
        self.rrthome = RRTStarHome()
        print("S5: RRTStarHome is loaded successfully!")

    def load_straight_line_planner(self):
        self.straight_line_planner = StraightLinePathPlanner()
        print("S6: Straight line planner is loaded successfully!")

    def run(self):
        x_current = X_START
        y_current = Y_START
        x_previous = x_current
        y_previous = y_current
        trajectory = []

        # get pioneer waypoint
        mu = self.grfar_model.mu_cond.flatten()
        Sigma = self.grfar_model.Sigma_cond
        self.CV.update_cost_valley(mu=mu, Sigma=Sigma, x_current=x_current, y_current=y_current,
                                   x_previous=x_previous, y_previous=y_previous)
        # self.rrtstar.search_path_from_trees(cost_valley=self.CV.cost_valley,
        #                                     polygon_budget_ellipse=self.CV.budget.polygon_budget_ellipse,
        #                                     line_budget_ellipse=self.CV.budget.line_budget_ellipse,
        #                                     x_current=x_current, y_current=y_current)
        # x_next = self.rrtstar.x_next
        # y_next = self.rrtstar.y_next
        # print("x_next", x_next)
        # print("y_next", y_next)
        t1 = time.time()
        res = self.pool.apply_async(self.rrtstar.search_path_from_trees, args=(self.CV.cost_valley,
                                                                             self.CV.budget.polygon_budget_ellipse,
                                                                             self.CV.budget.line_budget_ellipse,
                                                                             x_current,
                                                                             y_current))
        t2 = time.time()
        print("Time consuemd: ", t2 - t1)
        # self.rrtstar.search_path_from_trees(cost_valley=self.CV.cost_valley,
        #                                     polygon_budget_ellipse=self.CV.budget.polygon_budget_ellipse,
        #                                     line_budget_ellipse=self.CV.budget.line_budget_ellipse,
        #                                     x_current=x_current, y_current=y_current)
        # x_next = self.rrtstar.x_next
        # y_next = self.rrtstar.y_next
        x_next, y_next, self.path_to_target = res.get()
        # print("x_next", x_next)
        # print("y_next", y_next)


        for j in range(NUM_STEPS):
            print("Step: ", j)
            trajectory.append([x_current, y_current])

            # ind_measured = self.grf_model.get_ind_from_location(x_current, y_current)
            # self.grf_model.update_grf_model(ind_measured, self.grf_model.mu_truth[ind_measured])
            # mu = self.grf_model.mu_cond
            # Sigma = self.grf_model.Sigma_cond
            # print("mu: ", mu.shape)
            # print("Sigma: ", Sigma.shape)

            # ind_measured = self.grfar_model.get_ind_from_location(x_current, y_current)
            ind_measured = self.get_ind_sample(x1=x_previous, y1=y_previous, x2=x_current, y2=y_current)
            self.grfar_model.update_grfar_model(ind_measured=vectorise(ind_measured),
                                                salinity_measured=vectorise(self.grfar_model.mu_truth[ind_measured]),
                                                timestep=1)
            mu = self.grfar_model.mu_cond.flatten()
            Sigma = self.grfar_model.Sigma_cond
            self.CV.update_cost_valley(mu=mu, Sigma=Sigma, x_current=x_current, y_current=y_current,
                                       x_previous=x_previous, y_previous=y_previous)
            self.gohome = self.CV.budget.gohome_alert

            if not self.gohome:
                t1 = time.time()
                res = self.pool.apply_async(self.rrtstar.search_path_from_trees, args=(self.CV.cost_valley,
                                                                                       self.CV.budget.polygon_budget_ellipse,
                                                                                       self.CV.budget.line_budget_ellipse,
                                                                                       x_next,
                                                                                       y_next))
                t2 = time.time()
                print("Path planning takes: ", t2 - t1)
                x_pioneer, y_pioneer, self.path_to_target = res.get()
                # self.rrtstar.search_path_from_trees(cost_valley=self.CV.cost_valley,
                #                                     polygon_budget_ellipse=self.CV.budget.polygon_budget_ellipse,
                #                                     line_budget_ellipse=self.CV.budget.line_budget_ellipse,
                #                                     x_current=x_next, y_current=y_next)
                # x_pioneer = self.rrtstar.x_next
                # y_pioneer = self.rrtstar.y_next
            else:
                self.obstacle_in_the_way = self.is_obstacle_in_the_way(x1=x_next, y1=y_next, x2=X_HOME, y2=Y_HOME)
                if self.obstacle_in_the_way:
                    t1 = time.time()
                    res = self.pool.apply_async(self.rrthome.search_path_from_trees, args=(x_next, y_next,
                                                                                     X_HOME, Y_HOME))
                    t2 = time.time()
                    print("Path planning takes: ", t2 - t1)
                    # self.rrthome.search_path_from_trees(x_current=x_next, y_current=y_next,
                    #                                     x_target=X_HOME, y_target=Y_HOME)
                    # x_pioneer = self.rrthome.x_next
                    # y_pioneer = self.rrthome.y_next
                    x_pioneer, y_pioneer, self.path_to_target = res.get()
                else:
                    self.straight_line_planner.get_waypoint_from_straight_line(x_current=x_next, y_current=y_next,
                                                                               x_target=X_HOME, y_target=Y_HOME)
                    x_pioneer = self.straight_line_planner.x_next
                    y_pioneer = self.straight_line_planner.y_next


            fig = plt.figure(figsize=(30, 10))
            gs = GridSpec(nrows=1, ncols=2)
            ax = fig.add_subplot(gs[0])

            ax.plot(self.rrtstar.polygon_border[:, 1], self.rrtstar.polygon_border[:, 0], 'k-.')
            ax.plot(self.rrtstar.polygon_obstacle[:, 1], self.rrtstar.polygon_obstacle[:, 0], 'k-.')

            if not self.gohome:
                # for node in self.rrtstar.tree_nodes:
                #     if node.parent is not None:
                #         plt.plot([node.y, node.parent.y],
                #                  [node.x, node.parent.x], "g-")
                # ax.plot(self.rrtstar.path_to_target[:, 1], self.rrtstar.path_to_target[:, 0], 'r')
                ax.plot(self.path_to_target[:, 1], self.path_to_target[:, 0], 'r')
            else:
                if self.obstacle_in_the_way:
                    # for node in self.rrthome.tree_nodes:
                    #     if node.parent is not None:
                    #         plt.plot([node.y, node.parent.y],
                    #                  [node.x, node.parent.x], "g-")
                    # ax.plot(self.rrthome.path_to_target[:, 1], self.rrthome.path_to_target[:, 0], 'r')
                    ax.plot(self.path_to_target[:, 1], self.path_to_target[:, 0], 'r')


            # plt.scatter(self.grf_model.grf_grid[:, 1], self.grf_model.grf_grid[:, 0], c=mu, cmap=get_cmap('RdBu', 15),
            #             vmin=10, vmax=30, alpha=.5)

            xplot = self.grf_grid[:, 1]
            yplot = self.grf_grid[:, 0]
            # if not self.gohome:
            #     triangulated = tri.Triangulation(xplot, yplot)
            #     x_triangulated = xplot[triangulated.triangles].mean(axis=1)
            #     y_triangulated = yplot[triangulated.triangles].mean(axis=1)
            #
            #     ind_mask = []
            #     for i in range(len(x_triangulated)):
            #         ind_mask.append(self.is_masked(y_triangulated[i], x_triangulated[i]))
            #     triangulated.set_mask(ind_mask)
            #     refiner = tri.UniformTriRefiner(triangulated)
            #     triangulated_refined, value_refined = refiner.refine_field(mu.flatten(), subdiv=3)
            #     contourplot = ax.tricontourf(triangulated_refined, value_refined, cmap=get_cmap("RdBu", 15), alpha=.5)
            #     im = ax.tricontour(triangulated_refined, value_refined, vmin=10, vmax=30, alpha=.5)
            # else:
            im = ax.scatter(xplot, yplot, c=mu, s=200, cmap=get_cmap("BrBG", 15), vmin=10, vmax=30, alpha=.5)
            plt.colorbar(im)

            ellipse = Ellipse(xy=(self.CV.budget.y_middle, self.CV.budget.x_middle), width=2 * self.CV.budget.ellipse_a,
                              height=2 * self.CV.budget.ellipse_b, angle=math.degrees(self.CV.budget.angle),
                              edgecolor='r', fc='None', lw=2)
            plt.gca().add_patch(ellipse)

            ax.plot(y_current, x_current, 'bs')
            ax.plot(y_previous, x_previous, 'y^')
            ax.plot(y_next, x_next, 'r*')
            ax.plot(y_pioneer, x_pioneer, 'mP')
            ax.plot(Y_HOME, X_HOME, 'k*')

            p_traj = np.array(trajectory)
            ax.plot(p_traj[:, 1], p_traj[:, 0], 'k.-')
            plt.xlim([np.min(self.rrtstar.polygon_border[:, 1]), np.max(self.rrtstar.polygon_border[:, 1])])
            plt.ylim([np.min(self.rrtstar.polygon_border[:, 0]), np.max(self.rrtstar.polygon_border[:, 0])])

            plt.xlabel("East")
            plt.ylabel("North")
            plt.title("Updated mean after step: " + str(j))

            ax = fig.add_subplot(gs[1])
            ax.plot(self.rrtstar.polygon_border[:, 1], self.rrtstar.polygon_border[:, 0], 'k-.')
            ax.plot(self.rrtstar.polygon_obstacle[:, 1], self.rrtstar.polygon_obstacle[:, 0], 'k-.')
            if not self.gohome:
                # for node in self.rrtstar.tree_nodes:
                #     if node.parent is not None:
                #         plt.plot([node.y, node.parent.y],
                #                  [node.x, node.parent.x], "g-")
                # ax.plot(self.rrtstar.path_to_target[:, 1], self.rrtstar.path_to_target[:, 0], 'r')
                ax.plot(self.path_to_target[:, 1], self.path_to_target[:, 0], 'r')
            else:
                if self.obstacle_in_the_way:
                    # for node in self.rrthome.tree_nodes:
                    #     if node.parent is not None:
                    #         plt.plot([node.y, node.parent.y],
                    #                  [node.x, node.parent.x], "g-")
                    # ax.plot(self.rrthome.path_to_target[:, 1], self.rrthome.path_to_target[:, 0], 'r')
                    ax.plot(self.path_to_target[:, 1], self.path_to_target[:, 0], 'r')

            xplot = self.grf_grid[:, 1]
            yplot = self.grf_grid[:, 0]
            im = ax.scatter(xplot, yplot, c=self.CV.cost_valley, s=200, cmap=get_cmap("BrBG", 10), vmin=0, vmax=2, alpha=.5)
            plt.colorbar(im)
            ellipse = Ellipse(xy=(self.CV.budget.y_middle, self.CV.budget.x_middle), width=2 * self.CV.budget.ellipse_a,
                              height=2 * self.CV.budget.ellipse_b, angle=math.degrees(self.CV.budget.angle),
                              edgecolor='r', fc='None', lw=2)
            plt.gca().add_patch(ellipse)
            ax.plot(y_current, x_current, 'bs')
            ax.plot(y_previous, x_previous, 'y^')
            ax.plot(y_next, x_next, 'r*')
            ax.plot(y_pioneer, x_pioneer, 'mP')
            ax.plot(Y_HOME, X_HOME, 'k*')

            p_traj = np.array(trajectory)
            ax.plot(p_traj[:, 1], p_traj[:, 0], 'k.-')
            plt.xlim([np.min(self.rrtstar.polygon_border[:, 1]), np.max(self.rrtstar.polygon_border[:, 1])])
            plt.ylim([np.min(self.rrtstar.polygon_border[:, 0]), np.max(self.rrtstar.polygon_border[:, 0])])

            plt.xlabel("East")
            plt.ylabel("North")

            plt.title("Updated cost valley after step: " + str(j))
            plt.savefig(FILEPATH+"fig/rrtstar/P_{:03d}.jpg".format(j))
            plt.close("all")

            if self.gohome:
                if np.sqrt((X_HOME-x_current)**2 + (Y_HOME-y_current)**2)<=TARGET_RADIUS:
                    print("I am home, mission complete!")
                    break

            x_previous = x_current
            y_previous = y_current
            x_current = x_next
            y_current = y_next
            x_next = x_pioneer
            y_next = y_pioneer

    def is_obstacle_in_the_way(self, x1, y1, x2, y2):
        line = LineString([(x1, y1), (x2, y2)])
        if self.rrtstar.line_obstacle_shapely.intersects(line):
            return True
        else:
            return False

    def is_masked(self, x, y):
        point = Point(x, y)
        masked = False
        if (self.rrtstar.polygon_obstacle_shapely.contains(point) or
                not self.rrtstar.polygon_border_shapely.contains(point) or
                not self.rrtstar.polygon_budget_ellipse.contains(point)):
            masked = True
        return masked

    def check_ind_sample(self):
        x1 = 2000
        y1 = -2000
        x2 = 1500
        y2 = -1000
        ind = s.get_ind_sample(x1, y1, x2, y2)
        plt.plot(s.grf_grid[:, 1], s.grf_grid[:, 0], 'r.')
        plt.plot([y1, y2], [x1, x2], 'b-')
        plt.plot(s.grf_grid[ind, 1], s.grf_grid[ind, 0], 'g.')
        plt.show()
        pass

    def get_ind_sample(self, x1, y1, x2, y2):
        N = 20
        x_path = np.linspace(x1, x2, N)
        y_path = np.linspace(y1, y2, N)
        dataset = np.vstack((x_path, y_path, np.ones_like(x_path), np.zeros_like(x_path))).T
        ind, value = self.assimilate_data(dataset)
        return ind

    def assimilate_data(self, dataset):
        print("dataset before filtering: ", dataset[:10, :])
        depth_dataset = np.abs(dataset[:, 2])
        ind_selected_depth_layer = np.where((depth_dataset >= .25) * (depth_dataset <= DEPTH_LAYER + .5))[0]
        dataset = dataset[ind_selected_depth_layer, :]
        print("dataset after filtering: ", dataset[:10, :])
        t1 = time.time()
        dx = (vectorise(dataset[:, 0]) @ np.ones([1, self.N_grf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.grf_grid[:, 0]).T) ** 2
        dy = (vectorise(dataset[:, 1]) @ np.ones([1, self.N_grf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.grf_grid[:, 1]).T) ** 2
        dist = dx + dy
        ind_min_distance = np.argmin(dist, axis=1)
        t2 = time.time()
        ind_assimilated = np.unique(ind_min_distance)
        salinity_assimilated = np.zeros(len(ind_assimilated))
        for i in range(len(ind_assimilated)):
            ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
            salinity_assimilated[i] = np.mean(dataset[ind_selected, 3])
        print("Data assimilation takes: ", t2 - t1)
        self.auv_data = []
        print("Reset auv_data: ", self.auv_data)
        return vectorise(ind_assimilated), vectorise(salinity_assimilated)


if __name__ == "__main__":
    s = Simulator()
    s.run()




