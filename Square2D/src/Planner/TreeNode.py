"""
TreeNode is the basis for tree expansion during RRT* exploration.
"""


class TreeNode:

    __x = .0
    __y = .0
    __cost = .0
    __parent = None

    def set_location(self, x: float, y: float) -> None:
        """ Set location for the new tree node. """
        self.__x = x
        self.__y = y

    def get_location(self) -> tuple:
        """ Return the location associated with the tree node. """
        return self.__x, self.__y

    def get_cost(self) -> float:
        """ Get cost associated with the tree node. """
        return self.__cost

    def get_parent(self):
        """ Return the parent node of the tree node. """
        return self.__parent


if __name__ == "__main__":
    t = TreeNode()
    print("h")



