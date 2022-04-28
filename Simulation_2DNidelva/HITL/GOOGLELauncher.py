"""
This script simulates GOOGLE
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-27
"""


from usr_func import *
from Config.Config import *
from Config.AdaframeConfig import * # !!!! ROSPY important
from grf_model import GRF
from CostValley import CostValley
from RRTStarCV import RRTStarCV, STEPSIZE
from AUV import AUV


# == Set up
# LAT_START = 63.449664
# LON_START = 10.363366
LAT_START = 63.456232
LON_START = 10.435198
X_START, Y_START = latlon2xy(LAT_START, LON_START, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
NUM_STEPS = 50
# ==


class GOOGLELauncher:

    def __init__(self):
        self.load_grf_model()
        self.load_cost_valley()
        self.load_rrtstar()
        self.setup_AUV()

    def load_grf_model(self):
        self.grf_model = GRF()
        print("S1: GRF model is loaded successfully!")

    def load_cost_valley(self):
        self.CV = CostValley()
        print("S2: Cost Valley is loaded successfully!")

    def load_rrtstar(self):
        self.rrtstar = RRTStarCV()
        print("S3: RRTStarCV is loaded successfully!")

    def setup_AUV(self):
        self.auv = AUV()
        print("S4: AUV is setup successfully!")

    def run(self):
        x_current = X_START
        y_current = Y_START
        x_previous = x_current
        y_previous = y_current

        self.counter_waypoint = 0
        self.salinity = []
        self.set_waypoint_to_xy(x_current, y_current)

        t_start = time.time()

        while not rospy.is_shutdown():
            if self.auv.init:
                print("Waypoint step: ", self.counter_waypoint)
                t_end = time.time()

                self.salinity.append(self.auv.currentSalinity)

                self.auv.current_state = self.auv.auv_handler.getState()
                print("AUV state: ", self.auv.current_state)

                if ((t_end - t_start) / self.auv.max_submerged_time >= 1 and
                        (t_end - t_start) % self.auv.max_submerged_time >= 0):
                    print("Longer than 10 mins, need a long break")
                    self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=self.auv.min_popup_time,
                                           phone_number=self.auv.phone_number,
                                           iridium_dest=self.auv.iridium_destination)  # self.ada_state = "surfacing"
                    t_start = time.time()

                if self.auv.auv_handler.getState() == "waiting":
                    print("Arrived the current location")

                    self.salinity_measured = np.mean(self.salinity[-10:])

                    print("Sampled salinity: ", self.salinity_measured)

                    ind_measured = self.grf_model.get_ind_from_location(x_current, y_current)
                    self.grf_model.update_grf_model(ind_measured, self.salinity_measured)
                    mu = self.grf_model.mu_cond
                    Sigma = self.grf_model.Sigma_cond
                    self.CV.update_cost_valley(mu, Sigma, x_current, y_current, x_previous, y_previous)
                    self.rrtstar.search_path_from_trees(self.CV.cost_valley, self.CV.budget.polygon_budget_ellipse,
                                                        self.CV.budget.line_budget_ellipse, x_current, y_current)
                    x_next = self.rrtstar.x_next
                    y_next = self.rrtstar.y_next
                    print("x_next, y_next", x_next, y_next)

                    self.counter_waypoint += 1

                    x_previous = x_current
                    y_previous = y_current
                    x_current = x_next
                    y_current = y_next

                    if self.counter_waypoint >= NUM_STEPS:
                        self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=self.auv.min_popup_time,
                                               phone_number=self.auv.phone_number,
                                               iridium_dest=self.auv.iridium_destination)  # self.ada_state = "surfacing"
                        lat_waypoint, lon_waypoint = xy2latlon(x_current, y_current, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
                        self.auv.auv_handler.setWaypoint(deg2rad(lat_waypoint), deg2rad(lon_waypoint), .5,
                                                         speed=self.auv.speed)
                        rospy.signal_shutdown("Mission completed!!!")
                        break
                    else:
                        lat_waypoint, lon_waypoint = xy2latlon(x, y, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
                        self.auv.auv_handler.setWaypoint(deg2rad(lat_waypoint), deg2rad(lon_waypoint), .5,
                                                         speed=self.auv.speed)

                self.auv.last_state = self.auv.auv_handler.getState()
                self.auv.auv_handler.spin()
            self.auv.rate.sleep()

    def set_waypoint_to_xy(self, x, y):
        lat_waypoint, lon_waypoint = xy2latlon(x, y, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        self.auv.auv_handler.setWaypoint(deg2rad(lat_waypoint), deg2rad(lon_waypoint), .5, speed=self.auv.speed)
        print("Set waypoint successfully!")


if __name__ == "__main__":
    s = GOOGLELauncher()
    s.run()





