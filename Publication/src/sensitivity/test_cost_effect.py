"""
This script handles the cost effect in the simulation study.
"""

from Planner.Planner import Planner
from AUVSimulator.AUVSimulator import AUVSimulator
import numpy as np
import os
import time
# from Visualiser.AgentPlot import AgentPlot
import matplotlib.pyplot as plt


class Agent:

    __NUM_STEP = 50
    __home_radius = 150

    # s0: set up trajectory
    # traj = np.empty([0, 2])

    def __init__(self) -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        self.loc_start = np.array([3000, 1500])
        self.planner = Planner(self.loc_start)

        # s2: setup AUV simulator.
        self.auv = AUVSimulator()
        pass

        # s4: setup Visualiser.
        # self.tp = TreePlotter()
        # self.ap = AgentPlot(self, figpath=os.getcwd() + "/../../fig/Nidelva2D_LongHorizon/")
        # self.visualiser = Visualiser(self, figpath=os.getcwd() + "/../fig/Myopic3D/")

    def run(self):
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """
        # s0: initialises the trajectory and counter
        self.planner = Planner(self.loc_start)
        self.trajectory = np.empty([0, 2])
        self.__counter = 0

        wp_start = self.planner.get_starting_waypoint()
        wp_end = self.planner.get_end_waypoint()

        # a1: move to current location
        self.auv.move_to_location(wp_start)

        t_start = time.time()
        t_pop_last = time.time()

        # self.ap.plot_agent()

        for i in range(self.__NUM_STEP):
            # print("counter: ", self.__counter)
            # t1 = time.time()
            # self.ap.plot_agent()
            # t2 = time.time()
            # print("Plotting takes ", t2 - t1)

            # s0: update the planning trackers.
            self.planner.update_planning_trackers()

            # p1: parallel move AUV to the first location
            wp_now = self.planner.get_current_waypoint()
            self.auv.move_to_location(wp_now)

            # print("trajectory: ", self.trajectory)
            self.trajectory = np.append(self.trajectory, wp_now.reshape(1, -1), axis=0)

            # s2: obtain CTD data
            ctd_data = self.auv.get_ctd_data()

            # s3: update pioneer waypoint
            # t1 = time.time()
            self.planner.update_pioneer_waypoint(ctd_data)
            # t2 = time.time()
            # print("Update pioneer waypoint takes: ", t2 - t1)
            self.__counter += 1
            # np.savetxt("counter.txt", np.array([self.__counter]))

    def get_counter(self) -> int:
        return self.__counter

    def get_num_steps(self) -> int:
        return self.__NUM_STEP


class TestCostEffect:

    def __init__(self) -> None:
        self.ag = Agent()
        self.rrts = self.ag.planner.get_rrstarcv()
        self.cv = self.rrts.get_CostValley()

        self.filepath = "./../sim_result/"

    def test_agent_run(self):
        NUM_REPLICATES = 30
        num_steps = self.ag.get_num_steps()

        # c1: more EIBV, less IVR
        self.cv.set_weight_eibv(1.9)
        self.cv.set_weight_ivr(.1)
        print("Weight EIBV / IVR: ", self.cv.get_eibv_weight(), self.cv.get_ivr_weight())
        self.traj_eibv = np.empty([0, num_steps, 2])

        # self.ag.run()

        for i in range(NUM_REPLICATES):
            t1 = time.time()
            self.ag.run()
            trj_temp = self.ag.trajectory
            # trj_temp = trj_temp.reshape(1, num_steps, 2)
            self.traj_eibv = np.append(self.traj_eibv, trj_temp.reshape(1, num_steps, 2), axis=0)
            t2 = time.time()
            print("Replicate: ", i)
            print("Time consumed: ", t2 - t1)

        np.save(self.filepath + "eibv_ivr_{:.2f}_{:.2f}.npy".format(self.cv.get_eibv_weight(),
                                                                    self.cv.get_ivr_weight()), self.traj_eibv)

        # c2: more IVR, less EIBV
        self.cv.set_weight_eibv(.1)
        self.cv.set_weight_ivr(1.9)
        print("Weight EIBV / IVR: ", self.cv.get_eibv_weight(), self.cv.get_ivr_weight())
        self.traj_ivr = np.empty([0, num_steps, 2])
        for i in range(NUM_REPLICATES):
            t1 = time.time()
            self.ag.run()
            trj_temp = self.ag.trajectory
            # trj_temp = trj_temp.reshape(1, num_steps, 2)
            self.traj_ivr = np.append(self.traj_ivr, trj_temp.reshape(1, num_steps, 2), axis=0)
            t2 = time.time()
            print("Replicate: ", i)
            print("Time consumed: ", t2 - t1)

        np.save(self.filepath + "eibv_ivr_{:.2f}_{:.2f}.npy".format(self.cv.get_eibv_weight(),
                                                                    self.cv.get_ivr_weight()), self.traj_ivr)


        # c3: equal
        self.cv.set_weight_eibv(1.)
        self.cv.set_weight_ivr(1.)
        print("Weight EIBV / IVR: ", self.cv.get_eibv_weight(), self.cv.get_ivr_weight())
        self.traj_eq = np.empty([0, num_steps, 2])
        for i in range(NUM_REPLICATES):
            t1 = time.time()
            self.ag.run()
            trj_temp = self.ag.trajectory
            # trj_temp = trj_temp.reshape(1, num_steps, 2)
            self.traj_eq = np.append(self.traj_eq, trj_temp.reshape(1, num_steps, 2), axis=0)
            t2 = time.time()
            print("Replicate: ", i)
            print("Time consumed: ", t2 - t1)

        np.save(self.filepath + "eibv_ivr_{:.2f}_{:.2f}.npy".format(self.cv.get_eibv_weight(),
                                                                    self.cv.get_ivr_weight()), self.traj_eq)




if __name__ == "__main__":

    t = TestCostEffect()
    t.test_agent_run()

#%% Simulation analysis

[plt.plot(t.traj_eibv[i, :, 1], t.traj_eibv[i, :, 0], 'k.-', alpha=.4) for i in range(len(t.traj_eibv))]; plt.show()