
"""
This script setup the AUV
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-24
"""

from Config.AdaframeConfig import *
from usr_func import *
from Config.Config import *


class AUV:

    def __init__(self):
        self.node_name = 'MASCOT'
        rospy.init_node(self.node_name, anonymous=True)
        self.rate = rospy.Rate(1)  # 1Hz
        self.auv_handler = AuvHandler(self.node_name, "MASCOT")

        rospy.Subscriber("/IMC/Out/Salinity", Salinity, self.SalinityCB)
        rospy.Subscriber("/Vehicle/Out/EstimatedState_filtered", EstimatedState, self.EstimatedStateCB)

        self.speed = 1.5  # m/s
        self.depth = 0.0  # meters
        self.last_state = "unavailable"
        self.rate.sleep()
        self.init = True
        self.currentTemperature = 0.0
        self.currentSalinity = 0.0
        self.vehicle_pos = [0, 0, 0]

        self.max_submerged_time = 600
        self.min_popup_time = 90

        self.sms_pub_ = rospy.Publisher("/IMC/In/Sms", Sms, queue_size = 10)
        self.phone_number = "+4792526858"
        self.iridium_destination = "manta-ntnu-1"

    def SalinityCB(self, msg):
        self.currentSalinity = msg.value.data_auv

    def EstimatedStateCB(self, msg):
        offset_north = msg.lat_auv.data_auv - deg2rad(LATITUDE_ORIGIN)
        offset_east = msg.lon_auv.data_auv - deg2rad(LONGITUDE_ORIGIN)
        N = offset_north * CIRCUMFERENCE / (2.0 * np.pi)
        E = offset_east * CIRCUMFERENCE * np.cos(deg2rad(LATITUDE_ORIGIN)) / (2.0 * np.pi)
        D = msg.depth_auv.data_auv
        self.vehicle_pos = [N, E, D]
