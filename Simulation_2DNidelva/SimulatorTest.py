"""
This script simulates GOOGLE
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-27
"""

from usr_func import *
from GOOGLE.Simulation_2DNidelva.Config.Config import *
from GOOGLE.Simulation_2DNidelva.grf_model import GRF
from GOOGLE.Simulation_2DNidelva.CostValley import CostValley
from GOOGLE.Simulation_2DNidelva.RRTStarCV import RRTStarCV


# == Set up
LAT_START = 63.448747
LON_START = 10.416038
X_START, Y_START = latlon2xy(LAT_START, LON_START, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
NUM_STEPS = 10
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
        for i in range(NUM_STEPS):
            print("Step: ", i)
            print("x_next, y_next", x_current, y_current)
            ind_measured = self.grf_model.get_ind_from_location(x_current, y_current)
            self.grf_model.update_grf_model(ind_measured, self.grf_model.mu_truth[ind_measured])
            mu = self.grf_model.mu_cond
            Sigma = self.grf_model.Sigma_cond
            self.CV.update_cost_valley(mu, Sigma, x_current, y_current, x_previous, y_previous)
            self.rrtstar.search_path_from_trees(self.CV.cost_valley, self.CV.budget.polygon_budget_ellipse,
                                                self.CV.budget.line_budget_ellipse, x_current, y_current)
            x_next = self.rrtstar.x_next
            y_next = self.rrtstar.y_next
            x_previous = x_current
            y_previous = y_current
            x_current = x_next
            y_current = y_next


if __name__ == "__main__":
    s = Simulator()
    s.run()





