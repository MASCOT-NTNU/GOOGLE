"""
This script computes the straight line path between current location and target location
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-05-06
"""


from RRTStarCV import TARGET_RADIUS, STEPSIZE
import time
import numpy as np
import matplotlib.pyplot as plt


class StraightLinePathPlanner:

    def __init__(self):
        print("StraightLinePathPlanner is initialised successfully!")

    def get_waypoint_from_straight_line(self, x_current, y_current, x_target, y_target):
        t1 = time.time()
        distance_remaining = np.sqrt((x_current - x_target)**2 +
                                     (y_current - y_target)**2)
        if distance_remaining <= TARGET_RADIUS:
            self.x_next = x_target
            self.y_next = y_target
        else:
            angle = np.math.atan2(x_target - x_current,
                                  y_target - y_current)
            self.y_next = y_current + STEPSIZE * np.cos(angle)
            self.x_next = x_current + STEPSIZE * np.sin(angle)
        t2 = time.time()
        print("StraightLine planning takes: ", t2 - t1)

    def check(self):
        xn = 20
        yn = 20
        xt = 700
        yt = 500
        self.get_waypoint_from_straight_line(0, 0, 1000, 1000)
        plt.plot(yn, xn, 'r.')
        plt.plot(yt, xt, 'k*')
        plt.plot(self.y_next, self.x_next, 'bs')
        plt.xlim([0, 1000])
        plt.ylim([0, 1000])
        plt.show()
        pass


if __name__ == "__main__":
    slpp = StraightLinePathPlanner()
    slpp.check()


