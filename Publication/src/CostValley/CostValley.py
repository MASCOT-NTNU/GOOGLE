"""
CostValley module computes the overall cost field associated with a given grid.
At the moment, we use four components to construct the cost field to guide the agent to conduct
the adaptive sampling with a goal of balancing exploration and exploitation.

Cost components:
- Budget penalty: hard constraint and must-satisfy
- Directional penalty: soft constraint
- Obstacle penalty (not obvious for static island)
- EIBV reward: exploitation component
- IVR reward: exploration component

Another critical construction is to select a suitable weightset for all the components. Different weights on different
component might lead to different patterns of the final behaviour.

It is flexible to add or remove elements from its construction. One can add their own component
to make the system adaptive to their specific need and application.
"""

from GRF.GRF import GRF
from CostValley.Budget import Budget
from CostValley.Direction import Direction
import numpy as np
import time


class CostValley:

    def __init__(self) -> None:
        # fundamental components
        self.__grf = GRF()
        self.__field = self.__grf.field
        self.__grid = self.__field.get_grid()
        self.__Budget = Budget(self.__grid)
        self.__Direction = Direction(self.__grid)

        # get current location.
        self.__x_now, self.__y_now = self.__Budget.get_loc_now()

        # fundamental layers
        self.__eibv_field, self.__ivr_field = self.__grf.get_ei_field_total()
        # __azimuth_field = __Direction.get_direction_field(__x_now, __y_now)
        self.__budget_field = self.__Budget.get_budget_field(self.__x_now, self.__y_now)
        self.__cost_field = (self.__eibv_field +
                             self.__ivr_field +
                             # __azimuth_field +
                             self.__budget_field)

    def update_cost_valley(self, loc_now: np.ndarray):
        x_now, y_now = loc_now
        # t1 = time.time()
        self.__budget_field = self.__Budget.get_budget_field(x_now, y_now)
        # self.__azimuth_field = self.__Direction.get_direction_field(x_now, y_now)
        self.__eibv_field, self.__ivr_field = self.__grf.get_ei_field_total()
        self.__cost_field = (self.__eibv_field +
                             self.__ivr_field +
                             # self.__azimuth_field +
                             self.__budget_field)
        # t2 = time.time()
        # print("Update cost valley takes: ", t2 - t1)

    def get_cost_field(self) -> np.ndarray:
        return self.__cost_field

    def get_eibv_field(self) -> np.ndarray:
        return self.__eibv_field

    def get_ivr_field(self) -> np.ndarray:
        return self.__ivr_field

    # def get_direction_field(self) -> np.ndarray:
    #     return self.__azimuth_field

    def get_budget_field(self) -> np.ndarray:
        return self.__budget_field

    def get_grid(self) -> np.ndarray:
        return self.__grid

    def get_grf_model(self) -> 'GRF':
        return self.__grf

    def get_Budget(self) -> 'Budget':
        return self.__Budget

    def get_field(self):
        return self.__field

    def get_cost_at_location(self, loc: np.ndarray) -> float:
        """ Return cost associated with location. """
        ind = self.__field.get_ind_from_location(loc)
        return self.__cost_field[ind] ** 10

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
        ind = np.argmin(self.__cost_field)
        return self.__grid[ind]


if __name__ == "__main__":
    cv = CostValley()



