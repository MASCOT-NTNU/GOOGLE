"""
This script simulates GOOGLE
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-27
"""


from usr_func import *
from Config.Config import *
from Config.AUVConfig import * # !!!! ROSPY important
from AUV import AUV
from CostValley import CostValley
from RRTStarCV import RRTStarCV, TARGET_RADIUS
from RRTStarHome import RRTStarHome
from StraightLinePathPlanner import StraightLinePathPlanner
from grfar_model import GRFAR
import multiprocessing as mp


# == Set up
LAT_START = 63.456232
LON_START = 10.435198
# LAT_START = LATITUDE_HOME
# LON_START = LONGITUDE_HOME
X_START, Y_START = latlon2xy(LAT_START, LON_START, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
NUM_STEPS = 120
np.savetxt(FILEPATH + "Backup/current_location.txt", np.array([X_START, Y_START]), delimiter = ", ")
print("Current location is saved successfully!")
# ==


class GOOGLE2Launcher:

    def __init__(self):
        self.load_grfar_model()
        self.load_cost_valley()
        self.load_rrtstar()
        self.load_rrthome()
        self.load_straight_line_planner()
        self.setup_AUV()
        self.update_time = rospy.get_time()
        self.gohome = False
        self.obstacle_in_the_way = True
        self.straight_line = False
        self.popup = False
        self.pool = mp.Pool(1)
        print("S1-S6 complete")

    def load_grfar_model(self):
        self.grfar_model = GRFAR()
        self.grf_grid = self.grfar_model.grf_grid
        self.N_grf_grid = self.grf_grid.shape[0]
        print("S1: GRFAR model is loaded successfully!")

    def load_cost_valley(self):
        self.CV = CostValley()
        print("S2: Cost Valley is loaded successfully!")

    def load_rrtstar(self):
        self.rrtstar = RRTStarCV()
        print("S3: RRTStarCV is loaded successfully!")

    def load_rrthome(self):
        self.rrthome = RRTStarHome()
        print("S4: RRTStarHome is loaded successfully!")

    def load_straight_line_planner(self):
        self.straight_line_planner = StraightLinePathPlanner()
        print("S5: Straight line planner is loaded successfully!")

    def setup_AUV(self):
        self.auv = AUV()
        print("S6: AUV is setup successfully!")

    def load_current_location(self):
        self.x_current, self.y_current = np.loadtxt(FILEPATH + "Backup/current_location.txt", delimiter=", ")
        print("S7: Current location is loaded successfully!")

    def load_conditional_field(self):
        self.mu = np.load(FILEPATH + "Backup/mu.npy")
        self.Sigma = np.load(FILEPATH + "Backup/Sigma.npy")
        self.grfar_model.mu_cond = self.mu
        self.grfar_model.Sigma_cond = self.Sigma
        print("mu, Sigma is loaded successfully!")

    def run(self):
        # self.x_current = X_START
        # self.y_current = Y_START
        self.x_previous = self.x_current
        self.y_previous = self.y_current

        self.counter_waypoint = 0
        self.auv_data = []
        self.set_waypoint_to_xy(self.x_current, self.y_current)

        mu = self.grfar_model.mu_cond.flatten()
        Sigma = self.grfar_model.Sigma_cond
        self.CV.update_cost_valley(mu=mu, Sigma=Sigma, x_current=self.x_current, y_current=self.y_current,
                                   x_previous=self.x_previous, y_previous=self.y_previous)
        self.pool.apply_async(self.rrtstar.search_path_from_trees, args=(self.CV.cost_valley,
                                                                       self.CV.budget.polygon_budget_ellipse,
                                                                       self.CV.budget.line_budget_ellipse,
                                                                       self.x_current,
                                                                       self.y_current))
        # self.rrtstar.search_path_from_trees(cost_valley=self.CV.cost_valley,
        #                                     polygon_budget_ellipse=self.CV.budget.polygon_budget_ellipse,
        #                                     line_budget_ellipse=self.CV.budget.line_budget_ellipse,
        #                                     x_current=self.x_current, y_current=self.y_current)
        # self.x_next = self.rrtstar.x_next
        # self.y_next = self.rrtstar.y_next
        self.x_next, self.y_next = np.loadtxt(FILEPATH+"Waypoint/waypoint.txt", delimiter=', ')
        print("Waypoint is loaded successfully: ", self.x_next, self.y_next)

        t_start = time.time()
        self.t_ar1_start = time.time()
        while not rospy.is_shutdown():
            if self.auv.init:
                print("Waypoint step: ", self.counter_waypoint)
                if not self.straight_line:
                    self.x_next, self.y_next = np.loadtxt(FILEPATH + "Waypoint/waypoint.txt", delimiter=', ')
                    print("Next waypoint: ", self.x_next, self.y_next)
                t_end = time.time()

                self.auv_data.append([self.auv.vehicle_pos[0],
                                      self.auv.vehicle_pos[1],
                                      self.auv.vehicle_pos[2],
                                      self.auv.currentSalinity])

                self.auv.current_state = self.auv.auv_handler.getState()
                print("AUV state: ", self.auv.current_state)

                if ((t_end - t_start) / self.auv.max_submerged_time >= 1 and
                        (t_end - t_start) % self.auv.max_submerged_time >= 0):
                    print("Longer than 10 mins, need a long break")
                    self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=self.auv.min_popup_time,
                                           phone_number=self.auv.phone_number,
                                           iridium_dest=self.auv.iridium_destination)  # self.ada_state = "surfacing"
                    t_start = time.time()
                    self.popup = True

                if not self.popup:
                    if (self.auv.auv_handler.getState() == "waiting" and
                            rospy.get_time() -self.update_time > WAYPOINT_UPDATE_TIME):
                        print("Arrived the current location")
                        if self.gohome:
                            if np.sqrt((X_HOME - self.x_current) ** 2 + (Y_HOME - self.y_current) ** 2) <= TARGET_RADIUS:
                                self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=self.auv.min_popup_time,
                                                           phone_number=self.auv.phone_number,
                                                           iridium_dest=self.auv.iridium_destination)  # self.ada_state = "surfacing"
                                print("Mission complete! Congrates!")
                                rospy.signal_shutdown("Mission completed!!!")
                                break
                        self.x_previous = self.x_current
                        self.y_previous = self.y_current
                        self.x_current = self.x_next
                        self.y_current = self.y_next

                        lat_waypoint, lon_waypoint = xy2latlon(self.x_current, self.y_current, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
                        depth_waypoint = DEPTH_LAYER

                        if self.counter_waypoint >= NUM_STEPS:
                            self.auv.auv_handler.PopUp(sms=True, iridium=True, popup_duration=self.auv.min_popup_time,
                                                   phone_number=self.auv.phone_number,
                                                   iridium_dest=self.auv.iridium_destination)  # self.ada_state = "surfacing"
                            print("Mission complete! Congrates!")
                            rospy.signal_shutdown("Mission completed!!!")
                            break
                        else:
                            self.auv.auv_handler.setWaypoint(deg2rad(lat_waypoint), deg2rad(lon_waypoint), depth_waypoint,
                                                             speed=self.auv.speed)
                            print("Set waypoint successfully!")
                            self.update_time = rospy.get_time()

                        ind_assimilated, salinity_assimilated = self.assimilate_data(np.array(self.auv_data))
                        print("Sampled salinity: ", np.mean(salinity_assimilated))
                        self.t_ar1_end = time.time()
                        timestep = int((self.t_ar1_end - self.t_ar1_start)/TIME_AR1)
                        print("Timestep in AR1: ", timestep)
                        self.grfar_model.update_grfar_model(ind_assimilated, salinity_assimilated, timestep=timestep)
                        self.t_ar1_start = time.time()
                        mu = self.grfar_model.mu_cond.flatten()
                        Sigma = self.grfar_model.Sigma_cond
                        self.CV.update_cost_valley(mu=mu, Sigma=Sigma, x_current=self.x_current, y_current=self.y_current,
                                                   x_previous=self.x_previous, y_previous=self.y_previous)
                        self.gohome = self.CV.budget.gohome_alert

                        if not self.gohome:
                            self.pool.apply_async(self.rrtstar.search_path_from_trees,
                                                    args=(self.CV.cost_valley,
                                                         self.CV.budget.polygon_budget_ellipse,
                                                         self.CV.budget.line_budget_ellipse,
                                                         self.x_next,
                                                         self.y_next))
                            # self.rrtstar.search_path_from_trees(cost_valley=self.CV.cost_valley,
                            #                                     polygon_budget_ellipse=self.CV.budget.polygon_budget_ellipse,
                            #                                     line_budget_ellipse=self.CV.budget.line_budget_ellipse,
                            #                                     x_current=self.x_next, y_current=self.y_next)
                            # self.x_pioneer = self.rrtstar.x_next
                            # self.y_pioneer = self.rrtstar.y_next
                        else:
                            self.obstacle_in_the_way = self.is_obstacle_in_the_way(x1=self.x_next, y1=self.y_next,
                                                                                   x2=X_HOME, y2=Y_HOME)
                            if self.obstacle_in_the_way:
                                self.pool.apply_async(self.rrthome.search_path_from_trees,
                                                                args=(self.x_next,
                                                                     self.y_next,
                                                                     X_HOME,
                                                                     Y_HOME))
                                # self.rrthome.search_path_from_trees(x_current=self.x_next, y_current=self.y_next,
                                #                                     x_target=X_HOME, y_target=Y_HOME)
                                # self.x_pioneer = self.rrthome.x_next
                                # self.y_pioneer = self.rrthome.y_next
                            else:
                                # self.pool.apply_async(self.straight_line_planner.get_waypoint_from_straight_line,
                                #                                                 args=(self.x_next, self.y_next,
                                #                                                       X_HOME, Y_HOME))
                                self.straight_line = True
                                self.straight_line_planner.get_waypoint_from_straight_line(x_current=self.x_next,
                                                                                           y_current=self.y_next,
                                                                                           x_target=X_HOME,
                                                                                           y_target=Y_HOME)
                                self.x_next = self.straight_line_planner.x_next
                                self.y_next = self.straight_line_planner.y_next
                        self.counter_waypoint += 1
                else:
                    if (self.auv.auv_handler.getState() == "waiting" and
                            rospy.get_time() - self.update_time > WAYPOINT_UPDATE_TIME):
                        self.popup = False
                self.auv.last_state = self.auv.auv_handler.getState()
                self.auv.auv_handler.spin()
            self.auv.rate.sleep()

    def set_waypoint_to_xy(self, x, y):
        lat_waypoint, lon_waypoint = xy2latlon(x, y, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        self.auv.auv_handler.setWaypoint(deg2rad(lat_waypoint), deg2rad(lon_waypoint), .5, speed=self.auv.speed)
        print("Set waypoint successfully!")

    def is_obstacle_in_the_way(self, x1, y1, x2, y2):
        line = LineString([(x1, y1), (x2, y2)])
        if self.rrtstar.line_obstacle_shapely.intersects(line):
            return True
        else:
            return False

    def assimilate_data(self, dataset):
        print("dataset before filtering: ", dataset[:10, :])
        depth_dataset = np.abs(dataset[:, 2])
        ind_selected_depth_layer = np.where((depth_dataset >= MIN_DEPTH_FOR_DATA_ASSIMILATION) *
                                            (depth_dataset <= DEPTH_LAYER + DEPTH_TOLERANCE))[0]
        dataset = dataset[ind_selected_depth_layer, :]
        print("dataset after filtering: ", dataset[:10, :])
        t1 = time.time()
        dx = (vectorise(dataset[:, 0]) @ np.ones([1, self.N_grf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.grf_grid[:, 0]).T) ** 2
        dy = (vectorise(dataset[:, 1]) @ np.ones([1, self.N_grf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.grf_grid[:, 1]).T) ** 2
        dist = dx + dy
        ind_min_distance = np.argmin(dist, axis=1)
        t2 = time.time()
        ind_assimilated = np.unique(ind_min_distance)
        salinity_assimilated = np.zeros(len(ind_assimilated))
        for i in range(len(ind_assimilated)):
            ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
            salinity_assimilated[i] = np.mean(dataset[ind_selected, 3])
        print("Data assimilation takes: ", t2 - t1)
        self.auv_data = []
        print("Reset auv_data: ", self.auv_data)
        return vectorise(ind_assimilated), vectorise(salinity_assimilated)
    

if __name__ == "__main__":
    s = GOOGLE2Launcher()
    s.run()

