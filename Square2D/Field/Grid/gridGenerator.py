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
        
polygon_border = np.array(BORDER)
polygon_obstacle = np.array(OBSTACLE)
grid = HexgonalGrid2DGenerator(polygon_border=polygon_border, polygon_obstacle=polygon_obstacle,
                               distance_neighbour=DISTANCE_NEIGHBOUR)

coordinates = grid.grid_xy

df = pd.DataFrame(coordinates, columns=["x", 'y'])
df.to_csv(FILEPATH + "PreConfig/WaypointGraph/WaypointGraph.csv", index=False)

plt.plot(coordinates[:, 0], coordinates[:, 1], 'k.')
plt.plot(polygon_border[:, 0], polygon_border[:,1], 'r-')
plt.plot(polygon_obstacle[:, 0], polygon_obstacle[:,1], 'r-')
# plt.plot(y, x, 'r.')
plt.show()


        