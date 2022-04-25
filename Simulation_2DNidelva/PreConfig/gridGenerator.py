"""
This script generates grid and save them
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""

from GOOGLE.Simulation_2DNidelva.PreConfig.HexagonalGrid2D import HexgonalGrid2DGenerator
from GOOGLE.Simulation_2DNidelva.Config.Config import FILEPATH, LATITUDE_ORIGIN, LONGITUDE_ORIGIN
from usr_func import *

DISTANCE_NEIGHBOUR = 120


# PATH_OPERATION_AREA = PATH_FILE + "Config/OpArea.csv"
# PATH_MUNKHOLMEN = PATH_FILE + "Config/Munkholmen.csv"
# polygon_border = pd.read_csv(PATH_OPERATION_AREA).to_numpy()
# polygon_obstacle = pd.read_csv(PATH_MUNKHOLMEN).to_numpy()

polygon_border = FILEPATH + "Config/polygon_border.csv"
polygon_obstacle = FILEPATH + "Config/polygon_obstacle.csv"
polygon_border = pd.read_csv(polygon_border).to_numpy()
polygon_obstacle = pd.read_csv(polygon_obstacle).to_numpy()

grid = HexgonalGrid2DGenerator(polygon_border=polygon_border, polygon_obstacle=polygon_obstacle,
                               distance_neighbour=DISTANCE_NEIGHBOUR)
coordinates_wgs = grid.grid_wgs
coordinates_wgs = np.hstack((coordinates_wgs, np.zeros_like(coordinates_wgs[:, 0].reshape(-1, 1))))

lat_origin, lon_origin = ORIGIN
# TODO: add this feature into grid generator
x, y = latlon2xy(coordinates_wgs[:, 0], coordinates_wgs[:, 1], lat_origin, lon_origin)
z = np.zeros_like(x)
x, y, z = map(vectorise, [x, y, z])
coordinates_xyz = np.hstack((x, y, z))
vector_lat_origin = np.ones([len(coordinates_wgs), 1]) * lat_origin
vector_lon_origin = np.ones([len(coordinates_wgs), 1]) * lon_origin
dataset_coordinates = np.hstack((coordinates_wgs, coordinates_xyz, vector_lat_origin, vector_lon_origin))

df = pd.DataFrame(dataset_coordinates, columns=['lat', 'lon', 'depth', 'x', 'y', 'z', 'lat_origin', 'lon_origin'])
df.to_csv(FILEPATH + "PreConfig/WaypointGraph/WaypointGraph.csv", index=False)

import matplotlib.pyplot as plt
plt.plot(coordinates_wgs[:, 1], coordinates_wgs[:, 0], 'k.')
plt.show()
plt.plot(y, x, 'r.')
plt.show()


