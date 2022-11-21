""" CostValley object handles CostValley-related functions. """

from GRF import GRF
from CostValley.Budget import Budget
from CostValley.Obstacle import Obstacle
from CostValley.Direction import Direction
import numpy as np
import time


class CostValley:

    __x_now = .0
    __y_now = .0

    # fundamental components
    __grf = GRF()
    __field = __grf.field
    __grid = __field.get_grid()
    __Budget = Budget(__grid)
    __Obstacle = Obstacle(__grid, __field)
    __Direction = Direction(__grid)

    # fundamental layers
    __eibv_field, __ivr_field = __grf.get_ei_field()
    __obstacle_field = __Obstacle.get_obstacle_field()
    __azimuth_field = __Direction.get_direction_field(__x_now, __y_now)
    __budget_field = __Budget.get_budget_field(__x_now, __y_now)
    __cost_valley = (__obstacle_field +
                     __eibv_field +
                     __ivr_field +
                     __azimuth_field +
                     __budget_field)

    def update_cost_valley(self, loc_now: np.ndarray):
        x_now, y_now = loc_now
        # t1 = time.time()
        self.__budget_field = self.__Budget.get_budget_field(x_now, y_now)
        self.__azimuth_field = self.__Direction.get_direction_field(x_now, y_now)
        self.__eibv_field, self.__ivr_field = self.__grf.get_ei_field()
        self.__cost_valley = (self.__obstacle_field +
                              self.__eibv_field +
                              self.__ivr_field +
                              self.__azimuth_field +
                              self.__budget_field)
        # t2 = time.time()
        # print("Update cost valley takes: ", t2 - t1)

    def get_cost_valley(self) -> np.ndarray:
        return self.__cost_valley

    def get_eibv_field(self) -> np.ndarray:
        return self.__eibv_field

    def get_ivr_field(self) -> np.ndarray:
        return self.__ivr_field

    def get_direction_field(self) -> np.ndarray:
        return self.__azimuth_field

    def get_obstacle_field(self) -> np.ndarray:
        return self.__obstacle_field

    def get_budget_field(self) -> np.ndarray:
        return self.__budget_field

    def get_grid(self):
        return self.__grid

    def get_grf_model(self):  # TODO: delete
        return self.__grf

    def get_Budget(self):
        return self.__Budget

    def get_field(self):
        return self.__field

    def get_cost_at_location(self, loc: np.ndarray) -> float:
        """ Return cost associated with location. """
        ind = self.__field.get_ind_from_location(loc)
        return self.__cost_valley[ind]**10

    def get_cost_along_path(self, loc_start: np.ndarray, loc_end: np.ndarray) -> float:
        """ Return cost associated with a path. """
        dx = loc_start[0] - loc_end[0]
        dy = loc_start[1] - loc_end[1]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        c1 = self.get_cost_at_location(loc_start)
        c2 = self.get_cost_at_location(loc_end)
        ct = ((c1 + c2) / 2 * dist)
        return ct

    def get_minimum_cost_location(self):
        """ Return minimum cost location. """
        ind = np.argmin(self.__cost_valley)
        return self.__grid[ind]

