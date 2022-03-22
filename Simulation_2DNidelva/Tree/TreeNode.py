"""
This script only contains the tree node
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""


class TreeNode:

    def __init__(self, location=None, parent=None, cost=0, knowledge=None):
        self.location = location
        self.parent = parent
        self.cost = cost
        self.knowledge = knowledge


