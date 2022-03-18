"""
This script generates grid and save them
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""
import numpy as np
import pandas as pd
from GOOGLE.Simulation_2DNidelva.Field.Grid.HexagonalGrid2D import HexgonalGrid2DGenerator
from GOOGLE.Simulation_2DNidelva.Config.Config import *
PATH_FILE = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Simulation_2DNidelva/"

# PATH_OPERATION_AREA = PATH_FILE + "Config/OpArea.csv"
# PATH_MUNKHOLMEN = PATH_FILE + "Config/Munkholmen.csv"
# polygon_border = pd.read_csv(PATH_OPERATION_AREA).to_numpy()
# polygon_obstacle = pd.read_csv(PATH_MUNKHOLMEN).to_numpy()

polygon_border = PATH_FILE + "Config/polygon_border.csv"
polygon_obstacle = PATH_FILE + "Config/polygon_obstacle.csv"
polygon_border = pd.read_csv(polygon_border).to_numpy()
polygon_obstacle = pd.read_csv(polygon_obstacle).to_numpy()

grid = HexgonalGrid2DGenerator(polygon_border=polygon_border, polygon_obstacle=polygon_obstacle,
                               distance_neighbour=DISTANCE_NEIGHBOUR)
coordinates = grid.coordinates2d
coordinates = np.hstack((coordinates, np.zeros_like(coordinates[:, 0].reshape(-1, 1))))
df = pd.DataFrame(coordinates, columns=['lat', 'lon', 'depth'])
df.to_csv(PATH_FILE + "Field/Grid/Grid.csv", index=False)

coordinates.shape

