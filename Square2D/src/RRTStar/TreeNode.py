"""
TreeNode is the basis for tree expansion during RRT* exploration.
"""
import numpy as np


class TreeNode:

    __x = .0
    __y = .0
    __cost = .0
    __parent = None

    def __init__(self, loc: np.ndarray, cost=.0, parent=None):
        self.__x, self.__y = loc
        self.__cost = cost
        self.__parent = parent

    def set_location(self, loc: np.ndarray) -> None:
        """ Set location for the new tree node. """
        self.__x, self.__y = loc

    def set_cost(self, value: float) -> None:
        self.__cost = value

    def set_parent(self, parent: 'TreeNode') -> None:
        self.__parent = parent

    def get_location(self) -> np.ndarray:
        """ Return the location associated with the tree node. """
        return np.array([self.__x, self.__y])

    def get_cost(self) -> float:
        """ Get cost associated with the tree node. """
        return self.__cost

    def get_parent(self):
        """ Return the parent node of the tree node. """
        return self.__parent

    @staticmethod
    def get_distance_between_nodes(n1, n2):
        dx = n1.__x - n2.__x
        dy = n1.__y - n2.__y
        dist = np.sqrt(dx ** 2 + dy ** 2)
        return dist


if __name__ == "__main__":
    t = TreeNode(np.array([.0, .0]))



