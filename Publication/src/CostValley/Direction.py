"""
Direction module computes the directional penalty based on the current location, and previous location.
It calculates the dot product between the vector from the previous location to the current location and
the vector from the current location to the rest of the possible field. Based on this, it gives penalty to
location which is located behind the agent's inertial path.
"""

from usr_func.vectorize import vectorize
import numpy as np


class Direction:
    __PENALTY_AZIMUTH = 10
    __x_prev = .0
    __y_prev = .0
    __x_now = .0
    __y_now = .0
    __vector_azimuth = vectorize(np.array([__x_now - __x_prev,
                                           __y_now - __y_prev]))
    __grid = None
    __azimuth_field = None

    def __init__(self, grid):
        self.__grid = grid

    def get_direction_field(self, x_now, y_now) -> np.ndarray:
        self.__x_now = x_now
        self.__y_now = y_now
        self.__update_vector_azimuth()
        self.__azimuth_field = np.zeros_like(self.__grid[:, 0])
        dx1 = self.__grid[:, 0] - self.__x_now
        dy1 = self.__grid[:, 1] - self.__y_now
        vec1 = np.vstack((dx1, dy1)).T
        res = vec1 @ self.__vector_azimuth
        ind = np.where(res < 0)[0]
        self.__azimuth_field[ind] = self.__PENALTY_AZIMUTH
        return self.__azimuth_field

    def __update_vector_azimuth(self) -> None:
        dx = self.__x_now - self.__x_prev
        dy = self.__y_now - self.__y_prev
        self.__vector_azimuth = vectorize(np.array([dx, dy]))
        self.__x_prev = self.__x_now
        self.__y_prev = self.__y_now

    def get_current_location(self) -> np.ndarray:
        return np.array([self.__x_now, self.__y_now])

    def get_previous_location(self) -> np.ndarray:
        return np.array([self.__x_prev, self.__y_prev])

    def set_current_location(self, loc: np.ndarray) -> None:
        self.__x_now, self.__y_now = loc

    def set_previous_location(self, loc: np.ndarray) -> None:
        self.__x_prev, self.__y_prev = loc

