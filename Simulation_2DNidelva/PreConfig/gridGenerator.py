"""
This script generates grid and save them
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""
import pandas as pd

from GOOGLE.Simulation_2DNidelva.PreConfig.HexagonalGrid2D import HexgonalGrid2DGenerator
from GOOGLE.Simulation_2DNidelva.Config.Config import FILEPATH, LATITUDE_ORIGIN, LONGITUDE_ORIGIN
from usr_func import *

DISTANCE_NEIGHBOUR = 120


polygon_border = FILEPATH + "Config/polygon_border.csv"
polygon_obstacle = FILEPATH + "Config/polygon_obstacle.csv"
polygon_border = pd.read_csv(polygon_border).to_numpy()
polygon_obstacle = pd.read_csv(polygon_obstacle).to_numpy()

grid = HexgonalGrid2DGenerator(polygon_border=polygon_border, polygon_obstacle=polygon_obstacle,
                               distance_neighbour=DISTANCE_NEIGHBOUR)

grid_wgs = grid.grid_wgs
grid_xy = grid.grid_xy

df = pd.DataFrame(grid_xy, columns=['x', 'y'])
df.to_csv(FILEPATH + "Config/GRFGrid.csv", index=False)

df = pd.DataFrame(grid_wgs, columns=['lat', 'lon'])
df.to_csv(FILEPATH + "Test/GRFGrid.csv", index=False)

#%%
import matplotlib.pyplot as plt
# plt.plot(coordinates_wgs[:, 1], coordinates_wgs[:, 0], 'k.')
plt.plot(grid_xy[:, 1], grid_xy[:, 0], 'k.')
plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'r-.')
plt.plot(polygon_obstacle[:, 1], polygon_obstacle[:, 0], 'b-.')

plt.show()
# plt.plot(y, x, 'r.')
# plt.show()


