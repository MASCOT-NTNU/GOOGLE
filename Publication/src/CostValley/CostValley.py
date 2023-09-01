"""
CostValley module computes the overall cost field associated with a given grid.
For the simulation study, I only use two components to construct the cost field to guide the agent to conduct
the adaptive sampling with a goal of balancing exploration and exploitation.

Cost components:
- EIBV reward: exploitation component
- IVR reward: exploration component

Another critical construction is to select a suitable weightset for all the components. Different weights on different
component might lead to different patterns of the final behaviour.

It is flexible to add or remove elements from its construction. One can add their own component
to make the system adaptive to their specific need and application.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-08-24
"""
from CostValley.Budget import Budget
from GRF.GRF import GRF
import numpy as np
import time


class CostValley:
    """ Cost fields construction. """
    def __init__(self, weight_eibv: float = 1., weight_ivr: float = 1., sigma: float = 1., nugget: float = .4,
                 budget_mode: bool = False, approximate_eibv: bool = False, fast_eibv: bool = True) -> None:

        """ Budget mode """
        self.__budget_mode = budget_mode

        """ GRF """
        self.__grf = GRF(sigma=sigma, nugget=nugget, approximate_eibv=approximate_eibv, fast_eibv=fast_eibv)
        self.__field = self.__grf.field
        self.__grid = self.__field.get_grid()

        """ Weights """
        self.__weight_eibv = weight_eibv
        self.__weight_ivr = weight_ivr

        """ Cost field """
        self.__eibv_field, self.__ivr_field = self.__grf.get_ei_field()

        """ Budget field """
        self.__Budget = Budget(self.__grid)
        if self.__budget_mode:
            xnow, ynow = self.__Budget.get_loc_now()
            self.__budget_field = self.__Budget.get_budget_field(xnow, ynow)
            self.__cost_field = (self.__eibv_field * self.__weight_eibv + self.__ivr_field * self.__weight_ivr)
            # self.__cost_field = (self.__eibv_field * self.__weight_eibv + self.__ivr_field * self.__weight_ivr +
            #                      self.__budget_field)
        else:
            self.__cost_field = (self.__eibv_field * self.__weight_eibv + self.__ivr_field * self.__weight_ivr)

    def update_cost_valley(self, loc_now: np.ndarray = np.array([0, 0])) -> None:
        # t1 = time.time()
        self.__eibv_field, self.__ivr_field = self.__grf.get_ei_field()
        if self.__budget_mode:
            xnow, ynow = loc_now
            self.__budget_field = self.__Budget.get_budget_field(xnow, ynow)
            self.__cost_field = (self.__eibv_field * self.__weight_eibv + self.__ivr_field * self.__weight_ivr +
                                 self.__budget_field)
        else:
            self.__cost_field = (self.__eibv_field * self.__weight_eibv + self.__ivr_field * self.__weight_ivr)
        # t2 = time.time()
        # print("Update cost valley takes: ", t2 - t1)

    def get_cost_field(self) -> np.ndarray:
        return self.__cost_field

    def get_eibv_field(self) -> np.ndarray:
        return self.__eibv_field

    def get_ivr_field(self) -> np.ndarray:
        return self.__ivr_field

    def get_budget_field(self) -> np.ndarray:
        return self.__budget_field

    def get_Budget(self) -> 'Budget':
        return self.__Budget

    def get_grf_model(self) -> 'GRF':
        return self.__grf

    def get_field(self):
        return self.__field

    def get_cost_at_location(self, loc: np.ndarray) -> float:
        """ Return cost associated with location. """
        ind = self.__field.get_ind_from_location(loc)
        return self.__cost_field[ind]

    def get_cost_along_path(self, loc_start: np.ndarray, loc_end: np.ndarray) -> float:
        """ Return cost associated with a path. """
        dx = loc_start[0] - loc_end[0]
        dy = loc_start[1] - loc_end[1]
        dist = np.sqrt(dx ** 2 + dy ** 2)
        c1 = self.get_cost_at_location(loc_start)
        c2 = self.get_cost_at_location(loc_end)
        ct = ((c1 + c2) / 2 * dist)
        return ct

    def get_minimum_cost_location(self) -> np.ndarray:
        """ Return minimum cost location. """
        ind = np.argmin(self.__cost_field)
        return self.__grid[ind]

    def set_weight_eibv(self, value: float) -> None:
        """ Set weight for EIBV field. """
        self.__weight_eibv = value

    def set_weight_ivr(self, value: float) -> None:
        """ Set weight for IVR field. """
        self.__weight_ivr = value

    def get_eibv_weight(self) -> float:
        """ Return weight for EIBV field. """
        return self.__weight_eibv

    def get_ivr_weight(self) -> float:
        """ Return weight for IVR field. """
        return self.__weight_ivr
    

if __name__ == "__main__":
    cv = CostValley()



