"""
Planner plans the next waypoint according to Sense, Plan, Act process.
It wraps all the essential components together to ease the procedure for the agent during adaptive sampling.

- It first updates the cost valley based on the conditional field.
- It then computes the next waypoint based on two strategies
    - If it has enough budget, then it will employ rrtstar with cost valley.
    - It the budget is running out, it will then use straight line planer instead.

Args:
    _wp_now: current waypoint
    _wp_next: next waypoint
    _wp_pion: pioneer waypoint

"""
from Config import Config
from Planner.RRTSCV.RRTStarCV import RRTStarCV
import numpy as np


class Planner:

    def __init__(self, loc_start: np.ndarray, weight_eibv: float = 1., weight_ivr: float = 1.) -> None:
        """ Initial phase
        - Update the starting location to be loc.
        - Update current waypoint to be starting location.
        - Calculate two steps ahead in the pioneer planning.
        """

        # s0: load configuration
        self.__config = Config()

        # s1: set up path planning strategies
        self.__rrtstarcv = RRTStarCV(weight_eibv=weight_eibv, weight_ivr=weight_ivr)
        self.__stepsize = self.__rrtstarcv.get_stepsize()

        # s1: setup cost valley.
        self.__cv = self.__rrtstarcv.get_CostValley()
        self.__wp_min_cv = self.__cv.get_minimum_cost_location()

        # s2: set up data assimilation kernel
        self.__grf = self.__cv.get_grf_model()
        self.__grid = self.__grf.grid

        # s3: update the current waypoint location and append to trajectory and then get the minimum cost location.
        self.__wp_now = loc_start

        # s4: compute angle between the starting location to the minimum cost location.
        angle = np.math.atan2(self.__wp_min_cv[0] - self.__wp_now[0],
                              self.__wp_min_cv[1] - self.__wp_now[1])

        # s5: compute next location and pioneer location.
        xn = self.__wp_now[0] + self.__stepsize * np.sin(angle)
        yn = self.__wp_now[1] + self.__stepsize * np.cos(angle)
        self.__wp_next = np.array([xn, yn])

        xp = xn + self.__stepsize * np.sin(angle)
        yp = yn + self.__stepsize * np.cos(angle)
        self.__wp_pion = np.array([xp, yp])

        self.__trajectory = [[self.__wp_now[0], self.__wp_now[1]]]
        self.__wp_min_cv = self.__cv.get_minimum_cost_location()

    def update_planning_trackers(self) -> None:
        """ Move the pointer one step ahead. """
        self.__wp_now = self.__wp_next
        self.__wp_next = self.__wp_pion
        self.__trajectory.append([self.__wp_now[0], self.__wp_now[1]])

    def update_pioneer_waypoint(self, ctd_data: np.ndarray) -> None:
        """
        Compute the next waypoint:
        - Step I: assimilate data to the kernel.
        - Step II: update the cost valley.
        - Step III: get the minimum cost location in the cost field.
        - Step IV: plan one step ahead.
        """
        # s1: assimilate data to the kernel.
        self.__grf.assimilate_data(ctd_data)

        # s2: update cost valley
        self.__cv.update_cost_valley()

        # s3: get minimum cost location.
        self.__wp_min_cv = self.__cv.get_minimum_cost_location()

        # s4: plan one step based on cost valley and rrtstar
        self.__wp_pion = self.__rrtstarcv.get_next_waypoint(self.__wp_next, self.__wp_min_cv)

    def get_pioneer_waypoint(self) -> np.ndarray:
        return self.__wp_pion

    def get_next_waypoint(self) -> np.ndarray:
        return self.__wp_next

    def get_current_waypoint(self) -> np.ndarray:
        return self.__wp_now

    def get_trajectory(self) -> list:
        return self.__trajectory

    def get_rrtstarcv(self) -> 'RRTStarCV':
        return self.__rrtstarcv


if __name__ == "__main__":
    p = Planner()


