"""
T stores and measures the time used during the simulation study for rrtstar.
"""


class T:
    """ Time container. """
    def __init__(self):
        self.__t_total = .0
        self.__t_new_wp = .0
        self.__t_steering = .0
        self.__t_rewire = .0
        self.__t_trajectory = .0
        self.__t_ep1 = .0
        self.__t_ep2 = .0
        self.__t_ep3 = .0
        self.__t_ep4 = .0

    def set_t_total(self, value: float) -> None:
        """ Set the total consumed time. """
        self.__t_total = value

    def set_t_new_wp(self, value: float) -> None:
        """ Set the new waypoint time. """
        self.__t_new_wp = value

    def set_t_steering(self, value: float) -> None:
        """ Set the steering time. """
        self.__t_steering = value

    def set_t_rewire(self, value: float) -> None:
        """ Set the rewire time. """
        self.__t_rewire = value

    def set_t_trajectory(self, value: float) -> None:
        """ Set the trajectory time. """
        self.__t_trajectory = value

    def set_t_ep1(self, value: float) -> None:
        """ Set the tree expansion time. """
        self.__t_ep1 = value

    def set_t_ep2(self, value: float) -> None:
        """ Set the tree expansion time. """
        self.__t_ep2 = value

    def set_t_ep3(self, value: float) -> None:
        """ Set the tree expansion time. """
        self.__t_ep3 = value

    def set_t_ep4(self, value: float) -> None:
        """ Set the tree expansion time. """
        self.__t_ep4 = value

    def get_t_total(self) -> float:
        """ Get total consumed time. """
        return self.__t_total

    def get_t_new_wp(self) -> float:
        """ Get the time for generating new waypoint. """
        return self.__t_new_wp

    def get_t_steering(self) -> float:
        """ Get the steering time. """
        return self.__t_steering

    def get_t_rewire(self) -> float:
        """ Get the rewire time. """
        return self.__t_rewire

    def get_t_trajectory(self) -> float:
        """ Get the trajectory time. """
        return self.__t_trajectory

    def get_t_ep1(self) -> float:
        """ Get the tree expansion time. """
        return self.__t_ep1

    def get_t_ep2(self) -> float:
        """ Get the tree expansion time. """
        return self.__t_ep2

    def get_t_ep3(self) -> float:
        """ Get the tree expansion time. """
        return self.__t_ep3

    def get_t_ep4(self) -> float:
        """ Get the tree expansion time. """
        return self.__t_ep4


if __name__ == "__main__":
    t = T()

