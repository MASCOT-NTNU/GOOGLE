from unittest import TestCase
from Config import Config
from WGS import WGS
import matplotlib.pyplot as plt
from numpy import testing
import numpy as np
from usr_func.set_resume_state import set_resume_state
from usr_func.get_resume_state import get_resume_state


class TestConfig(TestCase):

    def setUp(self) -> None:
        self.c = Config()

    def test_set_home_location(self):
        loc_home = self.c.get_loc_home()
        loc = np.array([41.12677, -8.68574])
        x, y = WGS.latlon2xy(loc[0], loc[1])
        self.assertIsNone(testing.assert_array_equal(loc_home, np.array([x, y])))

    def test_starting_home_location(self):
        loc_home = self.c.get_loc_home()
        loc_start = self.c.get_loc_start()
        plg = self.c.get_polygon_operational_area()
        x, y = WGS.latlon2xy(plg[:, 0], plg[:, 1])
        plt.plot(y, x, 'r-.')
        # plt.plot(plg[:, 1], plg[:, 0], 'r-.')
        plt.plot(loc_start[1], loc_start[0], 'k.')
        plt.plot(loc_home[1], loc_home[0], 'b.')
        plt.show()

    def test_get_resume_state(self):
        # s1: check false state
        set_resume_state(False)
        self.assertFalse(get_resume_state())

        # s2: check true state
        set_resume_state(True)
        self.assertTrue(get_resume_state())

        # s3: check false again
        set_resume_state(False)
        self.assertFalse(get_resume_state())

        # s4: check true
        set_resume_state(True)
        self.assertTrue(get_resume_state())

        # s5: check false
        set_resume_state(False)
        self.assertFalse(get_resume_state())


