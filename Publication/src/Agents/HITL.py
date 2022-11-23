"""
Agent object abstract the entire adaptive agent by wrapping all the other components together inside the class.
It handles the procedure of the execution by integrating all essential modules and expand its functionalities.

The goal of the agent is to conduct the autonomous sampling operation by using the following procedure:
- Sense
- Plan
- Act

Sense refers to the in-situ measurements. Once the agent obtains the sampled values in the field. Then it can plan based
on the updated knowledge for the field. Therefore, it can act according to the planned manoeuvres.
"""

from Planner.Planner import Planner
from usr_func.get_resume_state import get_resume_state
from AUV.AUV import AUV
from WGS import WGS
import numpy as np
import time
import os
import math
import rospy


class Agent:

    __NUM_STEP = 40
    __home_radius = 150  # [m] for the radius to the home.
    __counter = 0

    def __init__(self) -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        # s0: setup AUV.
        self.auv = AUV()

        # s1: setup planner.
        loc_auv = self.auv.get_vehicle_pos()
        loc_start = loc_auv[:2]
        self.planner = Planner(loc_start)

        # s2: load the counter
        resume = get_resume_state()
        if not resume:
            self.__counter = 0
        else:
            self.__counter = int(np.loadtxt("counter.txt")) + 1

    def run(self):
        """
        Run the autonomous operation according to Sense, Plan, Act philosophy.
        """

        # c1: start the operation from scratch.
        wp_depth = .5
        wp_start = self.planner.get_starting_waypoint()
        wp_end = self.planner.get_end_waypoint()

        speed = self.auv.get_speed()
        max_submerged_time = self.auv.get_max_submerged_time()
        popup_time = self.auv.get_min_popup_time()
        phone = self.auv.get_phone_number()
        iridium = self.auv.get_iridium()

        # a1: move to current location
        lat, lon = WGS.xy2latlon(wp_start[0], wp_start[1])
        self.auv.auv_handler.setWaypoint(math.radians(lat), math.radians(lon), wp_depth, speed=speed)

        t_pop_last = time.time()
        update_time = rospy.get_time()

        ctd_data = []
        while not rospy.is_shutdown():
            if self.auv.init:
                t_now = time.time()
                print("counter: ", self.__counter)

                # s1: append data
                loc_auv = self.auv.get_vehicle_pos()
                ctd_data.append([loc_auv[0], loc_auv[1], loc_auv[2], self.auv.get_salinity()])

                if self.__counter == 0:
                    if t_now - t_pop_last >= max_submerged_time:
                        self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=popup_time,
                                                   phone_number=phone, iridium_dest=iridium)
                        t_pop_last = time.time()

                if ((self.auv.auv_handler.getState() == "waiting") and
                        (rospy.get_time() - update_time) > 5.):
                    if t_now - t_pop_last >= max_submerged_time:
                        self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=popup_time,
                                                   phone_number=phone, iridium_dest=iridium)
                        t_pop_last = time.time()

                    # s0: update the planning trackers.
                    self.planner.update_planning_trackers()

                    # s1: parallel move AUV to the first location
                    wp_now = self.planner.get_current_waypoint()
                    lat, lon = WGS.xy2latlon(wp_now[0], wp_now[1])
                    self.auv.auv_handler.setWaypoint(math.radians(lat), math.radians(lon), wp_depth, speed=speed)
                    update_time = rospy.get_time()

                    # s2: obtain CTD data
                    ctd_data = np.array(ctd_data)

                    # s3: calculate pioneer waypoint.
                    t1 = time.time()
                    self.planner.update_pioneer_waypoint(ctd_data)
                    t2 = time.time()
                    print("Planning time consumed: ", t2 - t1)

                    ctd_data = []

                    # s4: check arrival
                    dist = np.sqrt((wp_now[0] - wp_end[0]) ** 2 +
                                   (wp_now[1] - wp_end[1]) ** 2)
                    if dist <= self.__home_radius or self.__counter >= self.__NUM_STEP:
                        self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=popup_time,
                                                   phone_number=phone,
                                                   iridium_dest=iridium)  # self.ada_state = "surfacing"

                        print("Mission complete! Congrates!")
                        self.auv.send_SMS_mission_complete()
                        rospy.signal_shutdown("Mission completed!!!")
                        break
                    print("counter: ", self.__counter)
                    self.__counter += 1
                    np.savetxt("counter.txt", np.array([self.__counter]))

                self.auv.last_state = self.auv.auv_handler.getState()
                self.auv.auv_handler.spin()
            self.auv.rate.sleep()

    def get_counter(self):
        return self.__counter


if __name__ == "__main__":
    a = Agent()
    a.run()


