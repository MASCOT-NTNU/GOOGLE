"""
This script creates the strategies using RRT*
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""

from usr_func import *
from GOOGLE.Simulation_2DSquare.Tree.TreeNode import TreeNode
from GOOGLE.Simulation_2DSquare.Config.Config import *
from GOOGLE.Simulation_2DSquare.Tree.Location import *


class RRTStar:

    def __init__(self, knowledge=None):
        self.knowledge = knowledge
        self.nodes = []
        self.trajectory = []
        self.maximum_iteration = self.knowledge.maximum_iteration
        self.distance_neighbour_radar = self.knowledge.distance_neighbour_radar

        self.starting_node = TreeNode(location=self.knowledge.starting_location, parent=None, 
                                      cost=0, knowledge=self.knowledge)
        self.ending_node = TreeNode(location=self.knowledge.ending_location, parent=None, 
                                    cost=0, knowledge=self.knowledge)
        self.goal_node = TreeNode(location=self.knowledge.goal_location, parent=None, 
                                  cost=0, knowledge=self.knowledge)

        self.counter_fig = 0

    def get_next_waypoint(self):
        self.expand_trees()
        self.get_shortest_trajectory()
        if len(self.trajectory) <= 2:
            self.maximum_iteration = MAXITER_HARD
            self.expand_trees()
            self.get_shortest_trajectory()
        self.path_minimum_cost = np.array(self.trajectory)
        if len(self.path_minimum_cost) <= 2:
            next_location = self.knowledge.ending_location
        else:
            next_location = Location(self.path_minimum_cost[-2, 0], self.path_minimum_cost[-2, 1])
        angle = np.math.atan2(next_location.y - self.knowledge.current_location.y,
                              next_location.x - self.knowledge.current_location.x)
        # if get_distance_between_locations(self.knowledge.current_location, next_location) < self.knowledge.step_size:
        x = self.knowledge.current_location.x + self.knowledge.step_size * np.cos(angle)
        y = self.knowledge.current_location.y + self.knowledge.step_size * np.sin(angle)
        next_location = Location(x, y)
        return next_location

    def expand_trees(self):
        self.nodes.append(self.starting_node)
        for i in range(self.maximum_iteration):
            if np.random.rand() <= self.knowledge.goal_sample_rate:
                new_location = self.knowledge.ending_location
            else:
                if self.knowledge.budget_ellipse_b < BUDGET_ELLIPSE_B_MARGIN_Tree:
                    new_location = self.get_new_location_within_budget_ellipse()
                else:
                    new_location = self.get_new_location()

            nearest_node = self.get_nearest_node(self.nodes, new_location)
            next_node = self.get_next_node(nearest_node, new_location)

            if self.is_location_within_obstacles(next_node):
                continue

            next_node, nearest_node = self.rewire_tree(next_node, nearest_node)

            if self.is_path_intersects_with_obstacles(nearest_node, next_node):
                continue

            if self.isarrived(next_node):
                self.ending_node.parent = next_node
            else:
                self.nodes.append(next_node)
            pass

    @staticmethod
    def get_new_location():
        x = np.random.uniform(XLIM[0], XLIM[1])
        y = np.random.uniform(YLIM[0], YLIM[1])
        location = Location(x, y)
        return location

    def get_new_location_within_budget_ellipse(self):
        # t1 = time.time()
        theta = np.random.uniform(0, 2 * np.pi)
        module = np.sqrt(np.random.rand())
        x_usr = self.knowledge.budget_ellipse_a * module * np.cos(theta)
        y_usr = self.knowledge.budget_ellipse_b * module * np.sin(theta)
        x_wgs = (self.knowledge.budget_middle_location.x +
                 x_usr * np.cos(self.knowledge.budget_ellipse_angle) -
                 y_usr * np.sin(self.knowledge.budget_ellipse_angle))
        y_wgs = (self.knowledge.budget_middle_location.y +
                 x_usr * np.sin(self.knowledge.budget_ellipse_angle) +
                 y_usr * np.cos(self.knowledge.budget_ellipse_angle))
        # t2 = time.time()
        # print("Generating location takes: ", t2 - t1)
        return Location(x_wgs, y_wgs)

    def get_nearest_node(self, nodes, location):
        dist = []
        node_new = TreeNode(location)
        for node in nodes:
            dist.append(self.get_distance_between_nodes(node, node_new))
        return nodes[dist.index(min(dist))]

    def get_next_node(self, node, location):
        node_temp = TreeNode(location)
        if self.get_distance_between_nodes(node, node_temp) <= self.knowledge.step_size:
            return TreeNode(location=location, parent=node, knowledge=self.knowledge)
        else:
            angle = np.math.atan2(location.y - node.location.y, location.x - node.location.x)
            x = node.location.x + self.knowledge.step_size * np.cos(angle)
            y = node.location.y + self.knowledge.step_size * np.sin(angle)
            location_next = Location(x, y)
        return TreeNode(location=location_next, parent=node, knowledge=self.knowledge)

    @staticmethod
    def get_distance_between_nodes(node1, node2):
        dist_x = node1.location.x - node2.location.x
        dist_y = node1.location.y - node2.location.y
        dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
        return dist

    def rewire_tree(self, node_current, node_nearest):
        ind_neighbour_nodes = self.get_neighbour_nodes(node_current)

        for i in range(len(ind_neighbour_nodes)):
            node_neighbour = self.nodes[ind_neighbour_nodes[i]]
            if self.get_cost_between_nodes(node_neighbour, node_current) < \
                    self.get_cost_between_nodes(node_nearest, node_current):
                node_nearest = node_neighbour

            node_current.parent = node_nearest
            node_current.cost = self.get_cost_between_nodes(node_nearest, node_current)

        for i in range(len(ind_neighbour_nodes)):
            node_neighbour = self.nodes[ind_neighbour_nodes[i]]
            cost_current_neighbour = self.get_cost_between_nodes(node_current, node_neighbour)
            if cost_current_neighbour < node_neighbour.cost:
                node_neighbour.cost = cost_current_neighbour
                node_neighbour.parent = node_current
        return node_current, node_nearest

    def get_neighbour_nodes(self, node_current):
        distance_between_nodes = []
        for i in range(len(self.nodes)):
            if self.is_path_intersects_with_obstacles(self.nodes[i], node_current):
                distance_between_nodes.append(np.inf)
            else:
                distance_between_nodes.append(self.get_distance_between_nodes(self.nodes[i], node_current))
        ind_neighbours = np.where(np.array(distance_between_nodes) <= self.distance_neighbour_radar)[0]
        return ind_neighbours

    def get_cost_between_nodes(self, node1, node2):
        cost = (node1.cost +
                self.get_distance_between_nodes(node1, node2) +
                self.get_cost_from_cost_valley(node1, node2))
                # self.get_cost_from_cost_valley(node1, node2))
        return cost

    def get_cost_from_cost_valley(self, node1, node2):
        indF1 = get_ind_at_location2d_xy(self.knowledge.grid, node1.location)
        indF2 = get_ind_at_location2d_xy(self.knowledge.grid, node2.location)
        cost1 = self.knowledge.cost_valley[indF1]
        cost2 = self.knowledge.cost_valley[indF2]
        cost_total = ((cost1 + cost2) / 2 * self.get_distance_between_nodes(node1, node2))
        return cost_total

    def isarrived(self, node):
        dist = self.get_distance_between_nodes(self.ending_node, node)
        if dist < DISTANCE_TOLERANCE:
            return True
        else:
            return False

    '''
    Collision detection
    '''
    def is_location_within_obstacles(self, node):
        point = Point(node.location.x, node.location.y)
        within = False
        for i in range(len(self.knowledge.polygon_obstacles_shapely)):
            if self.knowledge.polygon_obstacles_shapely[i].contains(point):
                within = True
        return within

    def is_path_intersects_with_obstacles(self, node1, node2):
        line = LineString([(node1.location.x, node1.location.y),
                           (node2.location.x, node2.location.y)])
        intersect = False
        for i in range(len(self.knowledge.polygon_obstacles_shapely)):
            if self.knowledge.polygon_obstacles_shapely[i].intersects(line):
                intersect = True
        return intersect
    '''
    End of collision detection
    '''

    def get_shortest_trajectory(self):
        self.trajectory.append([self.ending_node.location.x, self.ending_node.location.y])
        pointer_node = self.ending_node
        while pointer_node.parent is not None:
            node = pointer_node.parent
            self.trajectory.append([node.location.x, node.location.y])
            pointer_node = node
        # self.trajectory = np.array(self.trajectory)

    def plot_tree(self):
        # plt.figure()
        if np.any(self.knowledge.polygon_obstacles):
            for i in range(len(self.knowledge.polygon_obstacles)):
                obstacle = np.append(self.knowledge.polygon_obstacles[i],
                                     self.knowledge.polygon_obstacles[i][0, :].reshape(1, -1), axis=0)
                plt.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')

        for node in self.nodes:
            if node.parent is not None:
                plt.plot([node.location.x, node.parent.location.x],
                         [node.location.y, node.parent.location.y], "-g")

        trajectory = np.array(self.trajectory)
        plt.plot(trajectory[:, 0], trajectory[:, 1], "-r")
        plt.plot(self.knowledge.starting_location.x, self.knowledge.starting_location.y, 'kv', ms=10)
        plt.plot(self.knowledge.ending_location.x, self.knowledge.ending_location.y, 'bx', ms=10)
        # middle_location = self.get_middle_location(self.starting_node, self.goal_node)
        # ellipse = Ellipse(xy=(middle_location.x, middle_location.y), width=2*self.budget_ellipse_a,
        #                   height=2*self.budget_ellipse_b, angle=math.degrees(self.budget_ellipse_angle),
        #                   edgecolor='r', fc='None', lw=2)
        # plt.gca().add_patch(ellipse)
        # plt.grid()
        # plt.title("rrt*")
        # plt.savefig(FIGPATH + "T_{:04d}.png".format(self.counter_fig))
        # plt.show()


