"""
AUV module communicates with the actual vehicle to send the updated waypoints and acquire
the sensor data through ROS-IMC bridge.
"""
from WGS import WGS
import rospy
from math import degrees
import numpy as np
from auv_handler import AuvHandler
from imc_ros_interface.msg import Temperature, Salinity, EstimatedState, Sms


class AUV:

    __speed = 1.5  # [m/s]
    __depth = .0
    __max_submerged_time = 600  # sec can be submerged.
    __min_popup_time = 90  # sec to be on the surface.
    #__phone_number = "+4740040327"
    __phone_number = "+351962901313"
    __iridium_destination = "manta-1"
    __currentSalinity = .0
    __vehicle_pos = [0, 0, 0]

    def __init__(self):
        self.node_name = 'AUV'
        rospy.init_node(self.node_name, anonymous=True)
        self.rate = rospy.Rate(1)  # 1Hz
        self.auv_handler = AuvHandler(self.node_name, "AUV")

        rospy.Subscriber("/IMC/Out/Salinity", Salinity, self.SalinityCB)
        rospy.Subscriber("/Vehicle/Out/EstimatedState_filtered", EstimatedState, self.EstimatedStateCB)

        self.last_state = "unavailable"
        self.rate.sleep()
        self.init = True
        self.sms_pub_ = rospy.Publisher("/IMC/In/Sms", Sms, queue_size = 10)

    def SalinityCB(self, msg):
        self.__currentSalinity = msg.value.data

    def EstimatedStateCB(self, msg):
        # lat_origin, lon_origin = WGS.get_origin()
        # circum = WGS.get_circumference()
        # offset_north = msg.lat.data - math.radians(lat_origin)
        # offset_east = msg.lon.data - math.radians(lon_origin)
        # N = offset_north * circum / (2.0 * np.pi)
        # E = offset_east * circum * np.cos(math.radians(lat_origin)) / (2.0 * np.pi)
        N, E = WGS.latlon2xy(degrees(msg.lat.data), degrees(msg.lon.data))
        D = msg.depth.data
        self.__vehicle_pos = [N, E, D]

    def send_SMS_mission_complete(self):
        print("Mission complete! will be sent via SMS")
        SMS = Sms()
        SMS.number.data = self.__phone_number
        SMS.timeout.data = 60
        SMS.contents.data = "Congrats, Mission complete! Now it is super bock time! "
        self.sms_pub_.publish(SMS)
        print("Finished SMS sending!")

    def get_vehicle_pos(self) -> list:
        """ Get the location of the vehicle in [N, E, D]. """
        return self.__vehicle_pos

    def get_salinity(self) -> float:
        """ Get the current salinity from the vehicle. """
        return self.__currentSalinity

    def get_speed(self) -> float:
        """ Get the speed of the vehicle. """
        return self.__speed

    def get_max_submerged_time(self) -> float:
        """ Get the maximum allowed submerged time. """
        return self.__max_submerged_time

    def get_min_popup_time(self) -> float:
        """ Get the minimum pop up time, or surfacing time to send SMS. """
        return self.__min_popup_time

    def get_phone_number(self) -> str:
        """ Get phone number. """
        return self.__phone_number

    def get_iridium(self) -> str:
        return self.__iridium_destination
