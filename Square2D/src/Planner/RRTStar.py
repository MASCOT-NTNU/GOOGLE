"""
RRTStar object produces the possible tree generation in the constrained field.
It employs RRT as the building block, and the cost associated with each tree branch is used to
determine the final tree discretization.
"""


class RRTStar:

    __tree = None
    __trajectory = None
    __maximum_iteration = 1000
    __neighbour_radius = .0

    def __init__(self):
        pass

    def expand_trees(self) -> None:
        # s1: append starting node

        # s2: loop all iterations:
        # s21: generate new random location within square.
            # if sampling rate is smaller, then target.
            # else generate new location
                # if budget is playing a role, within budget_ellipse_a
                # else: random location.
        # s3: find nearest node
        # s4: find next node (rewire)
        # s5: check collision, if yes, then no append and jump over.
        # s6: rewire tree
        # s7: check collision condition again,
        # s8: check arrival
            # yes, just append parent
            # else, just append regular nodes.
        pass

    def set_maximum_iteration(self, value: int) -> None:
        """ Set maximum iteration allowed to be value. """
        self.__maximum_iteration = value

    def get_maximum_iteration(self) -> int:
        """ Return maximum iteration allowed for the tree expansion. """
        return self.__maximum_iteration




if __name__ == "__main__":
    rt = RRTStar()
