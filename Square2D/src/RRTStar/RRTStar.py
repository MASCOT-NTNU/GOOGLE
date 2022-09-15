"""
RRTStar object produces the possible tree generation in the constrained field.
It employs RRT as the building block, and the cost associated with each tree branch is used to
determine the final tree discretization.
"""

from RRTStar.TreeNode import TreeNode
from Field import Field
import numpy as np
from shapely.geometry import Polygon, GeometryCollection, Point, LineString

# TODO: delete
from Visualiser.TreePlotter import TreePlotter
import matplotlib.pyplot as plt
import time


class RRTStar:
    # loc
    __loc_start = np.array([.01, .01])
    __loc_target = np.array([.99, .99])
    __loc_new = np.array([.01, .01])

    # tree
    __nodes = []  # all nodes in the tree.
    __trajectory = np.empty([0, 2])  # to save trajectory.
    __goal_sampling_rate = .01
    __max_expansion_iteration = 2000
    __distance_tolerance = .05
    __step_size = .1
    __distance_rewire_neighbour = .12

    # polygons and lines
    __polygon_border = Field.get_polygon_border()
    __polygon_obstacle = Field.get_polygon_obstacles()[0]
    __polygon_border_shapely = Polygon(__polygon_border)
    __polygon_obstacle_shapely = Polygon(__polygon_obstacle)
    __line_border_shapely = LineString(__polygon_border)
    __line_obstacle_shapely = LineString(__polygon_obstacle)
    __polygon_ellipse_shapely = None
    __line_ellipse_shapely = None

    # nodes
    __starting_node = TreeNode(__loc_start)
    __target_node = TreeNode(__loc_target)
    __nearest_node = None
    __new_node = None
    __neighbour_nodes = []

    # field
    __xlim, __ylim = Field.get_border_limits()

    # budget
    __Budget = False

    # cost valley
    __cost_valley = None

    # plot
    polygon_border = np.append(__polygon_border, __polygon_border[0, :].reshape(1, -1), axis=0)
    polygon_obstacle = np.append(__polygon_obstacle, __polygon_obstacle[0, :].reshape(1, -1), axis=0)
    cnt_plot = 0

    def get_next_waypoint(self, loc_start: np.ndarray, loc_target: np.ndarray, cost_valley) -> np.ndarray:
        """
        Get the next waypoint according to RRT* path planning philosophy.
        :param loc_start:
        :param loc_target:
        :param cost_valley: cost valley contains the cost field.
        :return next waypoint: np.array([x, y])
        """
        # s0: clean all nodes
        self.__nodes = []
        self.__trajectory = np.empty([0, 2])

        # s1: set starting location and target location in rrt*.
        self.__loc_start = loc_start
        self.__loc_target = loc_target
        self.__cost_valley = cost_valley
        self.__Budget = self.__cost_valley.get_Budget()
        self.__starting_node = TreeNode(self.__loc_start)
        self.__target_node = TreeNode(self.__loc_target)

        # update polygons from budget.
        self.__polygon_ellipse_shapely = self.__Budget.get_polygon_ellipse()
        self.__line_ellipse_shapely = self.__Budget.get_line_ellipse()

        # TODO: delete
        self.tp = TreePlotter()

        # s2: expand the trees.
        self.__expand_trees()

        # s3: get possible trajectory
        self.__get_shortest_trajectory()

        # s4: get the next possible waypoint out of trajectory.
        path_mc = np.array(self.__trajectory)
        if len(path_mc) <= 2:
            loc_next = self.__loc_target
        else:
            loc_next = path_mc[-2, :]
        angle = np.math.atan2(loc_next[1] - loc_start[1],
                              loc_next[0] - loc_start[0])
        x = loc_start[0] + self.__step_size * np.cos(angle)
        y = loc_start[1] + self.__step_size * np.sin(angle)
        wp_next = np.array([x, y])

        # # s5: final check legal condition, if not produce a random next location.
        # if not self.is_location_legal(wp_next) or not self.is_path_legal(loc_start, wp_next):
        #     angles = np.linspace(0, 2 * np.pi, 60)
        #     for angle in angles:
        #         x_next = loc_start[0] + self.__step_size * np.cos(angle)
        #         y_next = loc_start[1] + self.__step_size * np.sin(angle)
        #         ln = np.array([x_next, y_next])
        #         if self.is_location_legal(ln) and self.is_path_legal(loc_start, ln):
        #             wp_next = ln
        #             break
        return wp_next

    def __expand_trees(self):
        # t1 = time.time()
        self.__nodes.append(self.__starting_node)
        for i in range(self.__max_expansion_iteration):
            # if i % 100 == 0:
            print(i)
            # s1: convert this new location to a new node.
            self.__get_new_node()

            # s2: check if it is colliding.
            if not self.is_location_legal(self.__new_node.get_location()):
                continue

            # s3: rewire trees in the neighbourhood.
            self.__rewire_trees()

            # s4: check path possibility.
            if not self.is_path_legal(self.__nearest_node.get_location(),
                                      self.__new_node.get_location()):
                continue

            # s5: check connection to the goal node.
            if self.__isarrived():
                self.__target_node.set_parent(self.__new_node)
            else:
                self.__nodes.append(self.__new_node)

            self.tp.update_trees(self.__nodes)

            """ plot section """
            # plt.figure(figsize=(10, 10))
            # self.tp.plot_tree()
            # plt.plot(0, 0, 'r.', markersize=20)
            # plt.plot(0, 1, 'k*', markersize=20)
            # plt.plot(self.polygon_border[:, 0], self.polygon_border[:, 1], 'r-.')
            # plt.plot(self.polygon_obstacle[:, 0], self.polygon_obstacle[:, 1], 'r-.')
            # if self.__isarrived():
            #     # update new traj
            #     self.__get_shortest_trajectory()
            # traj = self.get_trajectory()
            # plt.plot(traj[:, 0], traj[:, 1], 'k-', linewidth=4)
            # plt.xlabel("x")
            # plt.ylabel("y")
            # plt.savefig("/Users/yaoling/Downloads/trees/rrts/P_{:03d}.png".format(self.cnt_plot))
            # self.cnt_plot += 1
            # plt.close("all")
            """ End of plot. """
        # t2 = time.time()
        # print("Tree expansion takes: ", t2 - t1)

    def __get_new_node(self):
        # s1: get new location within boundary.
        self.__get_new_location()

        # s2: steer new location to get the nearest tree node to this new location.
        self.__get_nearest_node()

        # s3: steer the new node.
        if TreeNode.get_distance_between_nodes(self.__nearest_node, self.__new_node) > self.__step_size:
            xn, yn = self.__nearest_node.get_location()
            angle = np.math.atan2(self.__loc_new[1] - yn,
                                  self.__loc_new[0] - xn)
            x = xn + self.__step_size * np.cos(angle)
            y = yn + self.__step_size * np.sin(angle)
            loc = np.array([x, y])
            self.__new_node = TreeNode(loc, parent=self.__nearest_node)

    def __get_new_location(self):
        """ Get new location within the operational area. """
        if np.random.rand() <= self.__goal_sampling_rate:
            self.__loc_new = self.__loc_target
        else:
            # if not self.__Budget.get_border_warning():
            """ s1: when border warning is not on, just sample from everywhere within the border. """
            x = np.random.uniform(self.__xlim[0], self.__xlim[1])
            y = np.random.uniform(self.__ylim[0], self.__ylim[1])
            self.__loc_new = np.array([x, y])
            # else:
            #     """ s2: when time to shrink polygon ellipse, just sample from inside the ellipse polygon and lim. """
            #     plg = self.__get_shrunk_border()
            #     plgs = Polygon(plg)
            #     xmin = np.amin(plg[:, 0])
            #     xmax = np.amax(plg[:, 0])
            #     ymin = np.amin(plg[:, 1])
            #     ymax = np.amax(plg[:, 1])
            #     while True:
            #         x = np.random.uniform(xmin, xmax)
            #         y = np.random.uniform(ymin, ymax)
            #         point = Point(x, y)
            #         if plgs.contains(point):
            #             self.__loc_new = np.array([x, y])
            #             break

    def __get_shrunk_border(self):
        """ Return updated shrunk border. """
        plg_box = Polygon(Field.get_polygon_border())
        plg_ellipse = Polygon(self.__Budget.get_polygon_ellipse())
        inters = [plg_ellipse.intersection(plg_box)]
        reg = GeometryCollection(inters)
        op = reg.geoms
        x = op[0].exterior.xy[0]
        y = op[0].exterior.xy[1]
        plg = np.vstack((x, y)).T
        self.__polygon_border = plg
        self.__polygon_border_shapely = Polygon(plg)
        self.__line_border_shapely = LineString(plg)
        return plg

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
            if Field.is_obstacle_in_the_way(node.get_location(), self.__new_node.get_location()):
                distance_between_nodes.append(np.inf)
            else:
                distance_between_nodes.append(TreeNode.get_distance_between_nodes(node, self.__new_node))
        ind_neighbours = np.where(np.array(distance_between_nodes) <= self.__distance_rewire_neighbour)[0]
        self.__neighbour_nodes = []
        for idx in ind_neighbours:  # TODO: check properly
            self.__neighbour_nodes.append(self.__nodes[idx])

    def __get_cost_between_nodes(self, n1: 'TreeNode', n2: 'TreeNode') -> float:
        """ Get cost between nodes. """
        # if self.is_path_legal(n1.get_location(), n2.get_location()):
        cost_distance = TreeNode.get_distance_between_nodes(n1, n2)
        cost_costvalley = self.__cost_valley.get_cost_along_path(n1.get_location(), n2.get_location())
        cost = n1.get_cost() + cost_distance + cost_costvalley
        # cost = n1.get_cost() + cost_distance
        # cost = .0
        # else:
        #     cost = np.inf
        return cost

    def __isarrived(self) -> bool:
        dist = TreeNode.get_distance_between_nodes(self.__new_node, self.__target_node)
        if dist < self.__distance_tolerance:
            return True
        else:
            return False

    def __get_shortest_trajectory(self):
        self.__trajectory = np.empty([0, 2])
        self.__trajectory = np.append(self.__trajectory, self.__target_node.get_location().reshape(1, -1), axis=0)
        pointer_node = self.__target_node
        cnt = 0
        while pointer_node.get_parent() is not None:
            cnt += 1
            node = pointer_node.get_parent()
            self.__trajectory = np.append(self.__trajectory, pointer_node.get_location().reshape(1, -1), axis=0)
            pointer_node = node
            if cnt > self.__max_expansion_iteration:
                break
        self.__trajectory = np.append(self.__trajectory, self.__starting_node.get_location().reshape(1, -1), axis=0)

    def __get_border_limits(self):
        return self.__xlim, self.__ylim

    def get_nodes(self):
        return self.__nodes

    def get_trajectory(self):
        return self.__trajectory

    def is_location_legal(self, loc: np.ndarray) -> bool:
        x, y = loc
        point = Point(x, y)
        islegal = True
        if self.__polygon_obstacle_shapely.contains(point) or not self.__polygon_ellipse_shapely.contains(point):
            islegal = False
        return islegal

    def is_path_legal(self, loc1: np.ndarray, loc2: np.ndarray) -> bool:
        x1, y1 = loc1
        x2, y2 = loc2
        line = LineString([(x1, y1), (x2, y2)])
        return not self.__polygon_obstacle_shapely.intersects(line)
        # islegal = True
        # c1 = self.__line_border_shapely.intersects(line)  # TODO: tricky to detect, since cannot have points on border.
        # c2 = self.__line_obstacle_shapely.intersects(line)
        # c3 = self.__line_ellipse_shapely.intersects(line)
        # if c1 or c2 or c3:
        #     islegal = False
        # return islegal


if __name__ == "__main__":
    t = RRTStar()



