""" Unit test for Field
This module tests the field object.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-08-22
"""
from unittest import TestCase
from Field import Field
from Config import Config
from shapely.geometry import Polygon, Point
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import numpy as np


class TestField(TestCase):
    """ Common test class for the waypoint graph module
    """
    def setUp(self) -> None:
        """ setup parameters """
        self.f = Field()
        self.c = Config()
        self.grid = self.f.get_grid()
        self.polygon_border = self.c.get_polygon_border()
        self.polygon_obstacle = self.c.get_polygon_obstacle()

    def test_empty_grids(self):
        """ Test if it generates empty waypoint. """
        actual_len = len(self.grid)
        min = 0
        self.assertGreater(actual_len, min, "Waypoints are empty! Test is not passed!")

    def test_illegal_grids(self):
        """ Test if any waypoints are not within the border polygon or colliding with obstacles. """
        pb = Polygon(self.polygon_border)
        po = Polygon(self.polygon_obstacle)
        s = True

        for i in range(len(self.grid)):
            p = Point(self.grid[i, :2])
            in_border = pb.contains(p)
            in_obs = po.contains(p)
            if in_obs or not in_border:
                s = False
                break
        self.assertTrue(s)

    def test_get_locations_from_ind(self):
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

    def test_get_ind_from_locations(self):
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

        plt.plot(self.grid[:, 1], self.grid[:, 0], 'k.', alpha=.1)
        plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'k-.')
        plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'k-.')
        for i in range(len(wp)):
            plt.plot([wp[i, 1], wr[i, 1]], [wp[i, 0], wr[i, 0]], 'r.-')
            # plt.plot(wr[i, 0], wr[i, 1], '.', alpha=.3)
        plt.gca().set_aspect('equal')
        plt.show()

    def test_border_contains(self):
        x, y = 1e6, 1e6
        b = self.f.border_contains(np.array([x, y]))
        self.assertFalse(b)
        x, y = self.grid[0, :]
        b = self.f.border_contains(np.array([x, y]))
        self.assertTrue(b)

    def test_border_in_the_way(self):
        """ Test if border is colliding with path. """
        # c1: far away
        x1, y1 = 0, 0
        x2, y2 = 5000, 0
        c = self.f.is_border_in_the_way(np.array([x1, y1]), np.array([x2, y2]))
        self.assertTrue(c)

        # c2: close
        x1, y1 = 0, 0
        x2, y2 = 5000, 10
        c = self.f.is_border_in_the_way(np.array([x1, y1]), np.array([x2, y2]))
        self.assertTrue(c)

        # c3: not colliding
        x1, y1 = 0, 0
        x2, y2 = 1000, 1000
        c = self.f.is_border_in_the_way(np.array([x1, y1]), np.array([x2, y2]))
        self.assertFalse(c)

    def test_get_neighbours(self):
        # c1: get one neighbour
        N = len(self.grid)
        ind = np.random.randint(0, N, 1)[0]
        loc_now = self.f.get_location_from_ind(ind)
        indn = self.f.get_neighbour_indices(ind)
        nd = self.f.get_neighbour_distance()
        for idn in indn:
            loc = self.f.get_location_from_ind(idn)
            dist = np.sqrt((loc[0] - loc_now[0])**2 +
                           (loc[1] - loc_now[1])**2)
            self.assertLess(dist, nd + .01*nd)

        # c2: multiple test.
        for i in range(100):
            ind = np.random.randint(0, N, 1)[0]
            loc_now = self.f.get_location_from_ind(ind)
            indn = self.f.get_neighbour_indices(ind)
            nd = self.f.get_neighbour_distance()
            for idn in indn:
                loc = self.f.get_location_from_ind(idn)
                dist = np.sqrt((loc[0] - loc_now[0]) ** 2 +
                               (loc[1] - loc_now[1]) ** 2)
                self.assertLess(dist, nd + .01*nd)

        # c3: multiple neighbour test.
        # loc = np.array([6000, 8000])
        # ind = self.f.get_ind_from_location(loc)
        # indn = self.f.get_neighbour_indices(ind)
        # indnn = self.f.get_neighbour_indices(indn)
        # indnnn = self.f.get_neighbour_indices(indnn)
        # plt.plot(self.grid[:, 1], self.grid[:, 0], 'k.', alpha=.1)
        # plt.plot(self.grid[indnnn, 1], self.grid[indnnn, 0], 'y.')
        # plt.plot(self.grid[indnn, 1], self.grid[indnn, 0], 'g.')
        # plt.plot(self.grid[indn, 1], self.grid[indn, 0], 'r.')
        # plt.plot(self.grid[ind, 1], self.grid[ind, 0], 'b.')
        # plt.show()

