"""
TreeNode is the basis for tree expansion during RRT* exploration.
"""
import numpy as np


class TreeNode:

    __x = .0
    __y = .0
    __cost = .0
    __parent = None

    def __init__(self, loc: np.ndarray, cost=0, parent=None) -> None:
        self.__x, self.__y = loc
        self.__cost = cost
        self.__parent = parent

    def set_location(self, loc: np.ndarray) -> None:
        """ Set location for the new tree node. """
        self.__x, self.__y = loc

    def set_cost(self, value: float) -> None:
        """ Set cost associated with tree node. """
        self.__cost = value

    def set_parent(self, parent: 'TreeNode') -> None:
        """ Set parent of the current tree node. """
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
    def get_distance_between_nodes(n1: 'TreeNode', n2: 'TreeNode') -> float:
        dist = np.sqrt((n1.__x - n2.__x)**2 +
                       (n1.__y - n2.__y)**2)
        return dist


if __name__ == "__main__":
    t = TreeNode()
    print("h")



