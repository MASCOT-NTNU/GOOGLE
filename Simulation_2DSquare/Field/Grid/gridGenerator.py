"""
This script generates the grid for the simulation study
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-23
"""

from GOOGLE.Simulation_2DSquare.Config.Config import *
from GOOGLE.Simulation_2DSquare.Tree.Location import *


class gridGenerator:

    def __init__(self):
        self.get_grid()

        pass

    def get_grid(self):
        self.x = np.linspace(XLIM[0], XLIM[1], NX)
        self.y = np.linspace(YLIM[0], YLIM[1], NY)
        self.x_matrix, self.y_matrix = np.meshgrid(self.x, self.y)
        self.grid_vector = []
        for i in range(self.x_matrix.shape[0]):
            for j in range(self.x_matrix.shape[1]):
                self.grid_vector.append([self.x_matrix[i, j], self.y_matrix[i, j]])
        self.grid_vector = np.array(self.grid_vector)
        self.x_vector = self.grid_vector[:, 0].reshape(-1, 1)
        self.y_vector = self.grid_vector[:, 1].reshape(-1, 1)
        self.num_nodes = len(self.grid_vector)
        print("Grid is built successfully!")