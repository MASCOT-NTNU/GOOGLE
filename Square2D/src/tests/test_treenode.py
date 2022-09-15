""" Unit test for planner

This module tests the planner object.

"""

from unittest import TestCase
from Planner.TreeNode import TreeNode


class TestPlanner(TestCase):
    """ Common test class for the waypoint graph module
    """

    def setUp(self) -> None:
        self.tn = TreeNode()

    def test_set_location(self):
        x, y = self.tn.get_location()
        self.assertEqual(x, 0)
        self.assertEqual(y, 0)

    def test_nothing(self):
        print("hello")

