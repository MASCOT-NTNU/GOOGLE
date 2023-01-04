""" Unit test for TreeNode

This module tests the planner object.

"""

from unittest import TestCase
from Planner.RRTSCV.TreeNode import TreeNode
import numpy as np


class TestPlanner(TestCase):
    """ Common test class for the waypoint graph module
    """

    def setUp(self) -> None:
        loc = np.array([.0, .0])
        self.tn = TreeNode(loc)

    def test_set_location(self):
        x, y = self.tn.get_location()
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)
        x, y = 1, 1
        loc = np.array([x, y])
        self.tn.set_location(loc)
        xx, yy = self.tn.get_location()
        self.assertEqual(x, xx)
        self.assertEqual(y, yy)

    def test_cost(self):
        c = self.tn.get_cost()
        self.assertEqual(c, .0)
        c = 10
        self.tn.set_cost(c)
        self.assertEqual(self.tn.get_cost(), c)

    def test_parent(self):
        self.assertIsNone(self.tn.get_parent())
        loc = np.array([.0, .0])
        p2 = TreeNode(loc)
        self.tn.set_parent(p2)
        p = self.tn.get_parent()
        self.assertEqual(p, p2)

