"""
This script handles the cost effect in the simulation study.
"""

from Planner.Planner import Planner
from AUVSimulator.AUVSimulator import AUVSimulator
import numpy as np
import os
import time
# from Visualiser.AgentPlot import AgentPlot


class Agent:

    __NUM_STEP = 40
    __home_radius = 150

    # s0: set up trajectory
    traj = np.empty([0, 2])

    def __init__(self) -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        # s1: set up planner.
        loc_start = np.array([3000, 1000])
        self.planner = Planner(loc_start)

        # s2: setup AUV simulator.
        self.auv = AUVSimulator()

        # s3: set up the counter
        self.__counter = 0

        # s4: setup Visualiser.
        # self.tp = TreePlotter()
        # self.ap = AgentPlot(self, figpath=os.getcwd() + "/../../fig/Nidelva2D_LongHorizon/")
        # self.visualiser = Visualiser(self, figpath=os.getcwd() + "/../fig/Myopic3D/")

        # s5: set trajectory
        self.__trajectory = np.empty([])

    def run(self):
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """
        wp_start = self.planner.get_starting_waypoint()
        wp_end = self.planner.get_end_waypoint()

        # a1: move to current location
        self.auv.move_to_location(wp_start)

        t_start = time.time()
        t_pop_last = time.time()

        # self.ap.plot_agent()

        while True:
            t_end = time.time()
            """
            Simulating the AUV behaviour, not the actual one
            """
            t_gap = t_end - t_start

            if t_gap >= 5:
                self.auv.arrive()
                t_start = time.time()

            if self.__counter == 0:
                if t_end - t_pop_last >= 50:
                    self.auv.popup()
                    print("POP UP")
                    t_pop_last = time.time()

            if self.auv.is_arrived():
                if t_end - t_pop_last >= 20:
                    self.auv.popup()
                    print("POPUP")
                    t_pop_last = time.time()

                # t1 = time.time()
                # self.ap.plot_agent()
                # t2 = time.time()
                # print("Plotting takes ", t2 - t1)

                # s0: update the planning trackers.
                self.planner.update_planning_trackers()

                # p1: parallel move AUV to the first location
                wp_now = self.planner.get_current_waypoint()
                self.auv.move_to_location(wp_now)

                # s2: obtain CTD data
                ctd_data = self.auv.get_ctd_data()

                # s3: update pioneer waypoint
                t1 = time.time()
                self.planner.update_pioneer_waypoint(ctd_data)
                t2 = time.time()
                print("Update pioneer waypoint takes: ", t2 - t1)

                # s8: check arrival
                dist = np.sqrt((wp_now[0] - wp_end[0]) ** 2 +
                               (wp_now[1] - wp_end[1]) ** 2)
                if dist <= self.__home_radius or self.__counter >= self.__NUM_STEP:
                    break
                print("counter: ", self.__counter)
                self.__counter += 1
                # np.savetxt("counter.txt", np.array([self.__counter]))

    def get_counter(self):
        return self.__counter


class TestCostEffect:

    def __init__(self) -> None:
        self.ag = Agent()

    def test_agent_run(self):
        self.ag.run()


if __name__ == "__main__":
    t = TestCostEffect()
    t.test_agent_run()
