"""
StraightLinePathPLanner plans one step ahead according to the desired angle.
"""

import time
import numpy as np


class StraightLinePathPlanner:

    __home_radius = 150  # metres.
    __step_size = 120  # metres.
    __loc_next = np.empty([0, 2])

    def get_waypoint_from_straight_line(self, loc_now: np.ndarray, loc_target: np.ndarray):
        x_now, y_now = loc_now
        x_target, y_target = loc_target
        t1 = time.time()
        distance_remaining = np.sqrt((x_now - x_target)**2 +
                                     (y_now - y_target)**2)
        if distance_remaining <= self.__home_radius:
            x_next = x_target
            y_next = y_target
        else:
            angle = np.math.atan2(y_target - y_now,
                                  x_target - x_now)
            x_next = x_now + self.__step_size * np.cos(angle)
            y_next = y_now + self.__step_size * np.sin(angle)
        return np.array([x_next, y_next])


if __name__ == "__main__":
    slpp = StraightLinePathPlanner()

