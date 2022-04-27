"""
This script simulates GOOGLE
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-27
"""
import matplotlib.pyplot as plt

from usr_func import *
from GOOGLE.Simulation_2DNidelva.Config.Config import *
from GOOGLE.Simulation_2DNidelva.grf_model import GRF
from GOOGLE.Simulation_2DNidelva.CostValley import CostValley
from GOOGLE.Simulation_2DNidelva.RRTStarCV import RRTStarCV


# == Set up
LAT_START = 63.450912
LON_START = 10.431774
X_START, Y_START = latlon2xy(LAT_START, LON_START, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
NUM_STEPS = 50
# ==


class Simulator:

    def __init__(self):
        self.load_grf_model()
        self.load_cost_valley()
        self.load_rrtstar()

    def load_grf_model(self):
        self.grf_model = GRF()
        print("S1: GRF model is loaded successfully!")

    def load_cost_valley(self):
        self.CV = CostValley()
        print("S2: Cost Valley is loaded successfully!")

    def load_rrtstar(self):
        self.rrtstar = RRTStarCV()
        print("S3: RRTStarCV is loaded successfully!")

    def run(self):
        x_current = X_START
        y_current = Y_START
        x_previous = x_current
        y_previous = y_current
        trajectory = []

        for j in range(NUM_STEPS):
            print("Step: ", j)
            print("x_next, y_next", x_current, y_current)
            trajectory.append([x_current, y_current])

            ind_measured = self.grf_model.get_ind_from_location(x_current, y_current)
            self.grf_model.update_grf_model(ind_measured, self.grf_model.mu_truth[ind_measured])
            mu = self.grf_model.mu_cond
            Sigma = self.grf_model.Sigma_cond
            self.CV.update_cost_valley(mu, Sigma, x_current, y_current, x_previous, y_previous)
            self.rrtstar.search_path_from_trees(self.CV.cost_valley, self.CV.budget.polygon_budget_ellipse,
                                                self.CV.budget.line_budget_ellipse, x_current, y_current)
            x_next = self.rrtstar.x_next
            y_next = self.rrtstar.y_next


            plt.figure(figsize=(10, 10))
            fig, ax = plt.subplots(1, 1)
            ax.plot(self.rrtstar.polygon_border[:, 1], self.rrtstar.polygon_border[:, 0], 'k-.')
            ax.plot(self.rrtstar.polygon_obstacle[:, 1], self.rrtstar.polygon_obstacle[:, 0], 'k-.')


            for i in range(self.rrtstar.tree_table.shape[0]):
                x1 = self.rrtstar.tree_table[i, 0]
                y1 = self.rrtstar.tree_table[i, 1]
                x2 = self.rrtstar.tree_table[int(self.rrtstar.tree_table[i, 3]), 0]
                y2 = self.rrtstar.tree_table[int(self.rrtstar.tree_table[i, 3]), 1]
                plt.plot([y1, y2], [x1, x2], 'g-')
            ax.plot(self.rrtstar.path_to_target[:, 1], self.rrtstar.path_to_target[:, 0], 'r')

            # plt.scatter(self.grf_model.grf_grid[:, 1], self.grf_model.grf_grid[:, 0], c=mu, cmap=get_cmap('RdBu', 15),
            #             vmin=10, vmax=30, alpha=.5)

            xplot = self.grf_model.grf_grid[:, 1]
            yplot = self.grf_model.grf_grid[:, 0]
            triangulated = tri.Triangulation(xplot, yplot)
            x_triangulated = xplot[triangulated.triangles].mean(axis=1)
            y_triangulated = yplot[triangulated.triangles].mean(axis=1)

            ind_mask = []
            for i in range(len(x_triangulated)):
                ind_mask.append(self.is_masked(y_triangulated[i], x_triangulated[i]))
            triangulated.set_mask(ind_mask)
            refiner = tri.UniformTriRefiner(triangulated)
            triangulated_refined, value_refined = refiner.refine_field(mu.flatten(), subdiv=3)
            contourplot = ax.tricontourf(triangulated_refined, value_refined, cmap=get_cmap("RdBu", 15), alpha=.5)
            im = ax.tricontour(triangulated_refined, value_refined, vmin=10, vmax=30, alpha=.5)
            plt.colorbar(im)

            ellipse = Ellipse(xy=(self.CV.budget.y_middle, self.CV.budget.x_middle), width=2 * self.CV.budget.ellipse_a,
                              height=2 * self.CV.budget.ellipse_b, angle=math.degrees(self.CV.budget.angle),
                              edgecolor='r', fc='None', lw=2)
            plt.gca().add_patch(ellipse)

            ax.plot(y_current, x_current, 'bs')
            ax.plot(y_previous, x_previous, 'y^')
            ax.plot(y_next, x_next, 'r*')
            ax.plot(Y_HOME, X_HOME, 'k*')

            p_traj = np.array(trajectory)
            ax.plot(p_traj[:, 1], p_traj[:, 0], 'k.-')
            plt.xlim([np.min(self.rrtstar.polygon_border[:, 1]), np.max(self.rrtstar.polygon_border[:, 1])])
            plt.ylim([np.min(self.rrtstar.polygon_border[:, 0]), np.max(self.rrtstar.polygon_border[:, 0])])

            plt.xlabel("East")
            plt.ylabel("North")

            plt.title("Updated mean after step: " + str(j))
            plt.savefig(FILEPATH+"fig/rrtstar/P_{:03d}.jpg".format(j))
            plt.close("all")


            x_previous = x_current
            y_previous = y_current
            x_current = x_next
            y_current = y_next

    def is_masked(self, x, y):
        point = Point(x, y)
        masked = False
        if (self.rrtstar.polygon_obstacle_shapely.contains(point) or
                not self.rrtstar.polygon_border_shapely.contains(point) or
                not self.rrtstar.polygon_budget_ellipse.contains(point)):
            masked = True
        return masked

if __name__ == "__main__":
    s = Simulator()
    s.run()





