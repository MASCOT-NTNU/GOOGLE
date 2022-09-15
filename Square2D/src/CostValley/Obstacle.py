"""
Obstacle tackles obstacle-related problems.
"""

import numpy as np


class Obstacle:
    __grid = None
    __obstacle_field = None
    __field = None

    def __init__(self, grid: np.ndarray, field):
        self.__grid = grid
        self.__field = field
        self.__compute_obstacle_field()

    def __compute_obstacle_field(self) -> None:
        obs = np.zeros_like(self.__grid[:, 0])
        for i in range(len(self.__grid)):
            if self.__field.obstacles_contain(self.__grid[i, :2]):
                obs[i] = np.inf
            else:
                obs[i] = 0
        self.__obstacle_field = obs

    def get_obstacle_field(self):
        return self.__obstacle_field

