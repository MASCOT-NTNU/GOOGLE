"""
This script builds the RRTStarCV
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-26
"""

from GOOGLE.Simulation_2DNidelva.Config.Config import FILEPATH
from usr_func import Polygon
import pandas as pd
import numpy as np

# == Setup
GOAL_SAMPLE_RATE = .01
MAX_ITERATION = 500
STEPSIZE = 500
RADIUS_NEIGHBOUR = 250
HOME_TOLERANCE = 100
# ==


class RRTStarCV:

    def __init__(self):
        self.load_grf_grid()
        self.load_polygon_border_obstacle()
        self.trees = np.zeros([MAX_ITERATION, 4])
        pass

    def load_grf_grid(self):
        self.grf_grid = pd.read_csv(FILEPATH+"Config/GRFGrid.csv").to_numpy()
        print("R1: GRF Grid is loaded successfully!")

    def load_polygon_border_obstacle(self):
        file_polygon_border = FILEPATH + "Config/polygon_border.csv"
        file_polygon_obstacle = FILEPATH + "Config/polygon_obstacle.csv"
        self.polygon_border = pd.read_csv(file_polygon_border).to_numpy()
        self.polygon_obstacle = pd.read_csv(file_polygon_obstacle).to_numpy()
        self.polygon_border_shapely = Polygon(self.polygon_border)
        self.polygon_obstacle_shapely = Polygon(self.polygon_obstacle)
        print("R2: Polygon border / obstacle are loaded successfully!")

    def update(self, cost_valley, ):
        #
        pass


if __name__ == "__main__":
    r = RRTStarCV()






