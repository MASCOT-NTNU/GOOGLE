"""
RRTStar object produces the possible tree generation in the constrained field.
It employs RRT as the building block, and the cost associated with each tree branch is used to
determine the final tree discretization.
"""
from Planner.RRTSCV.TreeNode import TreeNode
from Field import Field
from Config import Config
from CostValley.CostValley import CostValley
import numpy as np
import os
from time import time
from shapely.geometry import Polygon, Point, LineString


class RRTStarCV:
    """ RRT* CV planning strategy """

    def __init__(self, neighbour_distance: float = 120, weight_eibv: float = 1., weight_ivr: float = 1.,
                 sigma: float = .1, nugget: float = .01, budget_mode: bool = False,
                 approximate_eibv: bool = False, fast_eibv: bool = True) -> None:
        """
        Initialize the planner.
        """
        self.__budget_mode = budget_mode
        self.__config = Config()
        self.__field = Field(neighbour_distance=neighbour_distance)

        """ Load pre-generated random indices and locations to speed up the computation. """
        self.__filepath = os.getcwd() + "/Planner/RRTSCV/"
        self.__random_locations = np.load(self.__filepath + "RRT_Random_Locations.npy")
        self.__goal_indices = np.load(self.__filepath + "Goal_indices.npy")
        self.__N_random_locations = len(self.__random_locations)

        """ Cost valley """
        self.__cost_valley = CostValley(weight_eibv=weight_eibv, weight_ivr=weight_ivr, sigma=sigma, nugget=nugget,
                                        budget_mode=budget_mode, approximate_eibv=approximate_eibv, fast_eibv=fast_eibv)

        # loc
        self.__loc_start = np.array([1000, 1000])
        self.__loc_target = np.array([1000, 1000])
        self.__loc_new = np.array([1000, 1000])

        # tree
        self.__nodes = []  # all nodes in the tree.
        self.__trajectory = np.empty([0, 2])  # to save trajectory.
        self.__cost_trajectory = .0  # cost along the trajectory.
        self.__distance_trajectory = .0  # distance along the trajectory.
        self.__goal_sampling_rate = .01
        self.__max_expansion_iteration = 1700  # TODO: to run simulation and see if it is able to converage
        self.__stepsize = self.__field.get_neighbour_distance() * 1  # hard-coded values, need to be checked.
        self.__home_radius = self.__stepsize * .8
        self.__rrtstar_neighbour_radius = self.__stepsize * 1.12

        # polygons and lines
        self.__polygon_border_shapely = self.__config.get_polygon_border_shapely()
        self.__line_border_shapely = self.__config.get_line_border_shapely()

        self.__polygon_obstacle_shapely = self.__config.get_polygon_obstacle_shapely()
        self.__line_obstacle_shapely = self.__config.get_line_obstacle_shapely()

        self.__polygon_ellipse_shapely = None  # budget
        self.__line_ellipse_shapely = None

        # nodes
        self.__starting_node = TreeNode(self.__loc_start)
        self.__target_node = TreeNode(self.__loc_target)
        self.__nearest_node = None
        self.__new_node = None
        self.__neighbour_nodes = []

        # field
        self.__xlim, self.__ylim = self.__field.get_border_limits()

        # budget
        self.__Budget = self.__cost_valley.get_Budget()

    def get_next_waypoint(self, loc_start: np.ndarray, loc_target: np.ndarray) -> np.ndarray:
        """
        Get the next waypoint according to RRT* path planning philosophy.
        :param loc_start: current location np.array([x, y])
        :param loc_target: minimum cost location, np.array([x, y])
        :param cost_valley: cost valley contains the cost field.
        :return next waypoint: np.array([x, y])
        """
        t_start = time()
        # s0: clean all nodes
        self.__nodes = []

        # s1: set starting location and target location in rrt*.
        self.__loc_start = loc_start
        self.__loc_target = loc_target
        self.__starting_node = TreeNode(self.__loc_start)
        self.__target_node = TreeNode(self.__loc_target)

        # s12: update budget properties.
        if self.__budget_mode:
            self.__polygon_ellipse_shapely = self.__Budget.get_polygon_ellipse()
            self.__line_ellipse_shapely = self.__Budget.get_line_ellipse()

        # s2: expand the trees.
        self.__expand_trees()

        # s3: get shortest trajectory.
        self.__get_shortest_trajectory()

        # s4: get the next possible waypoint out of trajectory.
        path_mc = np.array(self.__trajectory)
        if len(path_mc) <= 2:
            loc_next = self.__loc_target
        else:
            loc_next = path_mc[-2, :]
        angle = np.math.atan2(loc_next[0] - loc_start[0],
                              loc_next[1] - loc_start[1])
        y = loc_start[1] + self.__stepsize * np.cos(angle)
        x = loc_start[0] + self.__stepsize * np.sin(angle)
        wp_next = np.array([x, y])

        # s5: final check legal condition, if not produce a random next location.
        if not self.is_location_legal(wp_next) or not self.is_path_legal(loc_start, wp_next):
            angles = np.linspace(0, 2 * np.pi, 60)
            for angle in angles:
                x_next = loc_start[0] + self.__stepsize * np.cos(angle)
                y_next = loc_start[1] + self.__stepsize * np.sin(angle)
                ln = np.array([x_next, y_next])
                if self.is_location_legal(ln) and self.is_path_legal(loc_start, ln):
                    wp_next = ln
                    break
        t_end = time()
        print("RRT* time: ", t_end - t_start, "s")
        return wp_next

    def __expand_trees(self):
        # start by appending the starting node to the nodes list.
        self.__nodes.append(self.__starting_node)

        # s0: select a chunk of random locations.
        ind_selected = np.random.randint(0, self.__N_random_locations, self.__max_expansion_iteration)
        x_random = self.__random_locations[ind_selected, 0]
        y_random = self.__random_locations[ind_selected, 1]
        goal_indices = self.__goal_indices[ind_selected]

        for i in range(self.__max_expansion_iteration):
            # print("tree: ", i)

            # s1: get new location.
            if goal_indices[i] <= self.__goal_sampling_rate:
                self.__loc_new = self.__loc_target
            else:
                self.__loc_new = np.array([x_random[i], y_random[i]])

            # s2: get nearest node to the current location.
            self.__get_nearest_node()

            # s3: steer new location to get the nearest tree node to this new location.
            if TreeNode.get_distance_between_nodes(self.__nearest_node, self.__new_node) > self.__stepsize:
                xn, yn = self.__nearest_node.get_location()
                angle = np.math.atan2(self.__loc_new[0] - xn,
                                      self.__loc_new[1] - yn)
                y = yn + self.__stepsize * np.cos(angle)
                x = xn + self.__stepsize * np.sin(angle)
                loc = np.array([x, y])
                self.__new_node = TreeNode(loc, parent=self.__nearest_node)

            # s4: check if it is colliding.
            if not self.is_location_legal(self.__new_node.get_location()):
                continue

            # s5: rewire trees in the neighbourhood.
            self.__rewire_trees()

            # s6: check path possibility.
            if not self.is_path_legal(self.__nearest_node.get_location(),
                                      self.__new_node.get_location()):
                continue

            # s7: check connection to the goal node.
            if self.__isarrived():
                self.__target_node.set_parent(self.__new_node)
                self.__target_node.set_cost(self.__get_cost_between_nodes(self.__target_node, self.__new_node))
            else:
                self.__nodes.append(self.__new_node)

    def __get_nearest_node(self) -> None:
        """ Return nearest node in the tree graph, only use distance. """
        dist = []
        self.__new_node = TreeNode(self.__loc_new)
        for node in self.__nodes:
            dist.append(TreeNode.get_distance_between_nodes(node, self.__new_node))
        self.__nearest_node = self.__nodes[dist.index(min(dist))]
        self.__new_node.set_parent(self.__nearest_node)

    def __rewire_trees(self):
        # s1: find cheapest node.
        self.__get_neighbour_nodes()
        for node in self.__neighbour_nodes:
            if (self.__get_cost_between_nodes(node, self.__new_node) <
                    self.__get_cost_between_nodes(self.__nearest_node, self.__new_node)):
                self.__nearest_node = node

            self.__new_node.set_parent(self.__nearest_node)
            self.__new_node.set_cost(self.__get_cost_between_nodes(self.__nearest_node, self.__new_node))

        # s2: update other nodes.
        for node in self.__neighbour_nodes:
            cost_current_neighbour = self.__get_cost_between_nodes(self.__new_node, node)
            if cost_current_neighbour < node.get_cost():
                node.set_cost(cost_current_neighbour)
                node.set_parent(self.__new_node)

    def __get_neighbour_nodes(self):
        distance_between_nodes = []
        for node in self.__nodes:
            distance_between_nodes.append(TreeNode.get_distance_between_nodes(node, self.__new_node))
        ind_neighbours = np.where(np.array(distance_between_nodes) <= self.__rrtstar_neighbour_radius)[0]
        self.__neighbour_nodes = []
        for idx in ind_neighbours:
            self.__neighbour_nodes.append(self.__nodes[idx])

    def __get_cost_between_nodes(self, n1: 'TreeNode', n2: 'TreeNode') -> float:
        """ Get cost between nodes. """
        cost_distance = TreeNode.get_distance_between_nodes(n1, n2) / self.__stepsize
        cost_costvalley = self.__cost_valley.get_cost_along_path(n1.get_location(), n2.get_location())
        cost = n1.get_cost() + cost_distance + cost_costvalley
        # cost = n1.get_cost() + cost_costvalley
        return cost

    def __isarrived(self) -> bool:
        dist = TreeNode.get_distance_between_nodes(self.__new_node, self.__target_node)
        if dist < self.__home_radius:
            return True
        else:
            return False

    def __get_shortest_trajectory(self):
        wp_old = self.__target_node.get_location().reshape(1, -1)
        self.__trajectory = np.empty([0, 2])
        self.__trajectory = np.append(self.__trajectory, wp_old, axis=0)
        self.__cost_trajectory = self.__target_node.get_cost()

        pointer_node = self.__target_node
        cnt = 0
        while pointer_node.get_parent() is not None:
            cnt += 1
            node = pointer_node.get_parent()
            wp_new = pointer_node.get_location().reshape(1, -1)
            self.__trajectory = np.append(self.__trajectory, wp_new, axis=0)
            self.__distance_trajectory += np.sqrt((wp_new[0, 0] - wp_old[0, 0])**2 +
                                                  (wp_new[0, 1] - wp_new[0, 1])**2)

            pointer_node = node
            wp_old = wp_new

            if cnt > self.__max_expansion_iteration:
                break

        wp_new = self.__starting_node.get_location().reshape(1, -1)
        self.__trajectory = np.append(self.__trajectory, wp_new, axis=0)
        self.__cost_trajectory += self.__starting_node.get_cost()
        self.__distance_trajectory += np.sqrt((wp_new[0, 0] - wp_old[0, 0]) ** 2 +
                                              (wp_new[0, 1] - wp_old[0, 1]) ** 2)

    def get_tree_nodes(self) -> list:
        """ Return all the tree nodes. """
        return self.__nodes

    def get_trajectory(self) -> np.ndarray:
        """ Return the trajectory from the starting location to the target location. """
        return self.__trajectory

    def get_cost_along_trajectory(self) -> float:
        """ Get the cost along the desired trajectory. """
        return self.__cost_trajectory

    def get_distance_along_trajectory(self) -> float:
        return self.__distance_trajectory

    def is_location_legal(self, loc: np.ndarray) -> bool:
        x, y = loc
        point = Point(x, y)
        islegal = True
        if self.__budget_mode:
            if self.__polygon_obstacle_shapely.contains(point) or not self.__polygon_ellipse_shapely.contains(point):
                islegal = False
        else:
            if self.__polygon_obstacle_shapely.contains(point):
                islegal = False
        return islegal

    def is_path_legal(self, loc1: np.ndarray, loc2: np.ndarray) -> bool:
        x1, y1 = loc1
        x2, y2 = loc2
        line = LineString([(x1, y1), (x2, y2)])
        islegal = True
        c1 = self.__line_border_shapely.intersects(line)  # TODO: tricky to detect, since cannot have points on border.
        c2 = self.__line_obstacle_shapely.intersects(line)
        if self.__budget_mode:
            c3 = self.__line_ellipse_shapely.intersects(line)
            if c1 or c2 or c3:
                islegal = False
        else:
            if c1 or c2:
                islegal = False
        return islegal

    def get_CostValley(self) -> 'CostValley':
        return self.__cost_valley

    def set_goal_sampling_rate(self, value: float) -> None:
        """ Set the goal sampling rate. """
        self.__goal_sampling_rate = value

    def set_stepsize(self, value: float) -> None:
        """ Set the step size of the trees. """
        self.__stepsize = value

    def set_max_expansion_iteraions(self, value: int) -> None:
        """ Set the maximum expansion itersions. """
        self.__max_expansion_iteration = value

    def set_rrtstar_neighbour_radius(self, value: float) -> None:
        """ Set the neighbour radius for tree searching. """
        self.__rrtstar_neighbour_radius = value

    def set_home_radius(self, value: float) -> None:
        """ Set the home radius for path convergence. """
        self.__home_radius = value

    def get_goal_sampling_rate(self) -> float:
        """ Get the goal sampling rate. """
        return self.__goal_sampling_rate

    def get_stepsize(self) -> float:
        """ Get the step size of the trees. """
        return self.__stepsize

    def get_max_expansion_iteraions(self) -> int:
        """ Get the maximum expansion itersions. """
        return self.__max_expansion_iteration

    def get_rrtstar_neighbour_radius(self) -> float:
        """ Get the neighbour radius for tree searching. """
        return self.__rrtstar_neighbour_radius

    def get_home_radius(self) -> float:
        """ Get the home radius for path convergence. """
        return self.__home_radius


if __name__ == "__main__":
    t = RRTStarCV()



