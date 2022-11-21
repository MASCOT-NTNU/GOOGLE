""" Unit test for Field
This module tests the field object.
"""

from unittest import TestCase
from Field import Field
from usr_func.is_list_empty import is_list_empty
from numpy import testing
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np
from numpy import testing


class TestField(TestCase):
    """ Common test class for the waypoint graph module
    """

    def run_test_empty_grids(self):
        """ Test if it generates empty waypoint. """
        actual_len = len(self.grid)
        min = 0
        self.assertGreater(actual_len, min, "Waypoints are empty! Test is not passed!")

    def run_test_illegal_grids(self):
        """ Test if any waypoints are not within the border polygon or colliding with obstacles. """
        pb = Polygon(self.polygon_border)
        pos = []
        if not is_list_empty(self.polygon_obstacle):
            for po in self.polygon_obstacle:
                pos.append(Polygon(po))
        s = True

        for i in range(len(self.grid)):
            p = Point(self.grid[i, :2])
            in_border = pb.contains(p)
            in_obs = False
            for po in pos:
                if po.contains(p):
                    in_obs = True
                    break
            if in_obs or not in_border:
                s = False
                break
        self.assertTrue(s)

    def run_test_get_locations_from_ind(self):
        # c1: empty ind
        wp = self.f.get_location_from_ind([])
        self.assertEqual(wp.shape[0], 0)
        # c2: one ind
        ids = 10
        wp = self.f.get_location_from_ind(ids).reshape(-1, 2)
        self.assertEqual(wp.shape[0], 1)
        # c3: multiple inds
        ids = [11, 13, 15]
        wp = self.f.get_location_from_ind(ids)
        self.assertEqual(wp.shape[0], len(ids))

    def run_test_get_ind_from_locations(self):
        """ Test waypoint interpolation works. Given random location, it should return indices for the nearest locations. """
        # c1: empty wp
        ind = self.f.get_ind_from_location([])
        self.assertIsNone(ind)

        xmin, ymin = map(np.amin, [self.grid[:, 0], self.grid[:, 1]])
        xmax, ymax = map(np.amax, [self.grid[:, 0], self.grid[:, 1]])

        # c2: one wp
        # s1: generate random waypoint
        xr = np.random.uniform(xmin, xmax)
        yr = np.random.uniform(ymin, ymax)
        wp = np.array([xr, yr])

        # s2: get indice from function
        ind = self.f.get_ind_from_location(wp)
        self.assertIsNotNone(ind)

        # s3: get waypoint from indice
        wr = self.grid[ind]

        # s4: compute distance between nearest waypoint and rest
        d = cdist(wr.reshape(1, -1), wp.reshape(1, -1))
        da = cdist(self.grid, wp.reshape(1, -1))

        # s5: see if it is nearest waypoint
        self.assertTrue(d, da.min())

        # c3: more than one wp
        # t = np.random.randint(0, len(self.waypoints))
        t = 10
        xr = np.random.uniform(xmin, xmax, t)
        yr = np.random.uniform(ymin, ymax, t)
        wp = np.stack((xr, yr), axis=1)

        ind = self.f.get_ind_from_location(wp)
        self.assertIsNotNone(ind)
        wr = self.grid[ind]
        d = np.diag(cdist(wr, wp))
        da = cdist(self.grid, wp)
        self.assertTrue(np.all(d == np.amin(da, axis=0)))

        plt.plot(self.grid[:, 0], self.grid[:, 1], 'k.', alpha=.1)
        for i in range(len(wp)):
            plt.plot([wp[i, 0], wr[i, 0]], [wp[i, 1], wr[i, 1]], 'r.-')
            # plt.plot(wr[i, 0], wr[i, 1], '.', alpha=.3)
        plt.show()

    def run_test_border_contains(self):
        x, y = 1e6, 1e6
        b = self.f.border_contains(np.array([x, y]))
        self.assertFalse(b)
        x, y = self.grid[0, :]
        b = self.f.border_contains(np.array([x, y]))
        self.assertTrue(b)

    def run_test_obstacles_contain(self):
        x, y = 1e6, 1e6
        o = self.f.obstacles_contain(np.array([x, y]))
        self.assertFalse(o)
        x, y = self.polygon_obtacles[0, :]
        o = self.f.border_contains(np.array([x, y]))
        self.assertTrue(o)

    def run_test_border_in_the_way(self):
        x1, y1 = 0, 0
        x2, y2 = 100, 100
        c = self.f.is_border_in_the_way(np.array([x1, y1]), np.array([x2, y2]))
        self.assertTrue(c)
        x1, y1 = -20, -20
        x2, y2 = -100, -100
        c = self.f.is_border_in_the_way(np.array([x1, y1]), np.array([x2, y2]))
        self.assertFalse(c)

    def run_test_obstacles_in_the_way(self):
        # c1: border along polygon obstacle
        x1, y1 = self.polygon_obstacle[0][0, :]
        x2, y2 = self.polygon_obstacle[0][-1, :]
        c = self.f.is_obstacle_in_the_way(np.array([x1, y1]), np.array([x2, y2]))
        self.assertTrue(c)

        # c2: no collision detection.
        x1, y1 = -20, -20
        x2, y2 = -100, -100
        c = self.f.is_obstacle_in_the_way(np.array([x1, y1]), np.array([x2, y2]))
        self.assertFalse(c)


class TC2(TestField):

    def setUp(self) -> None:
        """ setup parameters """
        self.f = Field()
        self.grid = self.f.get_grid()
        self.polygon_border = self.f.get_polygon_border()
        self.polygon_obstacle = self.f.get_polygon_obstacles()

    def test_all(self):
        self.run_test_illegal_grids()
        self.run_test_get_ind_from_locations()
        self.run_test_empty_grids()
        self.run_test_get_locations_from_ind()
        self.run_test_border_contains()
        self.run_test_border_in_the_way()
        self.run_test_obstacles_in_the_way()


if __name__ == "__main__":

    # T2: test setup 2
    tc2 = TC2()
    tc2.test_all()




