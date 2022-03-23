"""
This script generates the grid for the simulation study
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-23
"""

from GOOGLE.Simulation_2DSquare.Config.Config import *
from GOOGLE.Simulation_2DSquare.Tree.Location import *
from GOOGLE.Simulation_2DSquare.Field.Grid.HexagonalGrid2D import HexgonalGrid2DGenerator
from usr_func import *

x = np.linspace(XLIM[0], XLIM[1], NX)
y = np.linspace(YLIM[0], YLIM[1], NY)
x_matrix, y_matrix = np.meshgrid(x, y)
grid_vector = []
for i in range(x_matrix.shape[0]):
    for j in range(x_matrix.shape[1]):
        grid_vector.append([x_matrix[i, j], y_matrix[i, j]])
grid_vector = np.array(grid_vector)
x_vector = grid_vector[:, 0].reshape(-1, 1)
y_vector = grid_vector[:, 1].reshape(-1, 1)
num_nodes = len(grid_vector)
print("Grid is built successfully!")
        
polygon_border = np.array(BORDER)
polygon_obstacle = np.array(OBSTACLE)
grid = HexgonalGrid2DGenerator(polygon_border=polygon_border, polygon_obstacle=polygon_obstacle,
                               distance_neighbour=DISTANCE_NEIGHBOUR)

coordinates_wgs = grid.grid_wgs
coordinates_wgs = np.hstack((coordinates_wgs, np.zeros_like(coordinates_wgs[:, 0].reshape(-1, 1))))

#%%
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
df.to_csv(FILEPATH + "Field/Grid/Grid.csv", index=False)

import matplotlib.pyplot as plt
plt.plot(coordinates_wgs[:, 1], coordinates_wgs[:, 0], 'k.')
plt.show()
plt.plot(y, x, 'r.')
plt.show()



        
        