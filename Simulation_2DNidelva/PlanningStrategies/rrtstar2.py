"""
This script creates the strategies using RRT*
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""

from usr_func import *
from GOOGLE.Simulation_2DNidelva.Tree.TreeNode import TreeNode
from GOOGLE.Simulation_2DNidelva.Config.Config import *
from GOOGLE.Simulation_2DNidelva.Tree.Location import *


class RRTStar:

    def __init__(self, knowledge=None):
        self.nodes = []
        self.trajectory = []
        self.maxiter = MAXITER_EASY
        self.radius_neighbour = RADIUS_NEIGHBOUR
        self.knowledge = knowledge

        self.obstacles = np.array(OBSTACLES)
        self.polygon_obstacles = []
        self.set_obstacles()

        self.starting_node = TreeNode(self.knowledge.starting_location, None, 0, self.knowledge)
        self.ending_node = TreeNode(self.knowledge.ending_location, None, 0, self.knowledge)
        self.goal_node = TreeNode(self.knowledge.goal_location, None, 0, self.knowledge)

        self.counter_fig = 0

    def expand_trees(self):
        self.nodes.append(self.starting_node)
        for i in range(self.maxiter):
            if np.random.rand() <= self.knowledge.goal_sample_rate:
                new_location = self.knowledge.ending_location
            else:
                if self.knowledge.kernel.budget_ellipse_b < BUDGET_ELLIPSE_B_MARGIN_Tree:
                    # print("Here comes new sampling distribution!")
                    new_location = self.get_new_location_within_budget_ellipse()
                    self.radius_neighbour = RADIUS_NEIGHBOUR
                else:
                    new_location = self.get_new_location()

            nearest_node = self.get_nearest_node(self.nodes, new_location)
            next_node = self.get_next_node(nearest_node, new_location)

            if self.isWithin(next_node):
                continue

            next_node, nearest_node = self.rewire_tree(next_node, nearest_node)

            if self.isIntersect(nearest_node, next_node):
                continue

            if self.isarrived(next_node):
                self.ending_node.parent = next_node
            else:
                self.nodes.append(next_node)

    @staticmethod
    def get_new_location():
        x = np.random.uniform(XLIM[0], XLIM[1])
        y = np.random.uniform(YLIM[0], YLIM[1])
        location = Location(x, y)
        return location

    def get_bigger_box(self):
        self.box_lat_min, self.box_lon_min, self.box_depth_min = map(np.amin, [self.config.polygon_within[:, 0],
                                                                               self.config.polygon_within[:, 1],
                                                                               self.config.depth])
        self.box_lat_max, self.box_lon_max, self.box_depth_max = map(np.amax, [self.config.polygon_within[:, 0],
                                                                               self.config.polygon_within[:, 1],
                                                                               self.config.depth])

    def get_new_location(self):
        while True:
            lat = np.random.uniform(self.box_lat_min, self.box_lat_max)
            lon = np.random.uniform(self.box_lon_min, self.box_lon_max)
            if self.isWithin((lat, lon)):
                depth = np.random.uniform(self.box_depth_min, self.box_depth_max)
                location = Location(lat, lon, depth)
                return location

    def get_nearest_node(self, nodes, location):
        dist = []
        node_new = TreeNode(location)
        for node in nodes:
            dist.append(self.get_distance_between_nodes(node, node_new))
        return nodes[dist.index(min(dist))]

    @staticmethod
    def get_distance_between_nodes(node1, node2):
        dist_x, dist_y = latlon2xy(node1.location.lat, node1.location.lon, node2.location.lat, node2.location.lon)
        dist_z = node1.location.depth - node2.location.depth
        dist = np.sqrt(dist_x ** 2 + dist_y ** 2 + dist_z ** 2)
        return dist

    def get_next_node(self, node, location):
        x, y = latlon2xy(location.lat, location.lon, node.location.lat, node.location.lon)
        angle_lateral = np.math.atan2(x, y)
        y_new = self.config.step_lateral * np.cos(angle_lateral)
        x_new = self.config.step_lateral * np.sin(angle_lateral)
        angle_vertical = np.math.atan2(location.depth - node.location.depth, np.sqrt(x ** 2 + y ** 2))
        z_new = self.config.step_vertical * np.sin(angle_vertical)
        lat_new, lon_new = xy2latlon(x_new, y_new, node.location.lat, node.location.lon)
        depth_new = node.location.depth + z_new
        location_next = Location(lat_new, lon_new, depth_new)
        return TreeNode(location_next, node)

    def get_new_location_within_budget_ellipse(self):
        # t1 = time.time()
        theta = np.random.uniform(0, 2 * np.pi)
        module = np.sqrt(np.random.rand())
        x_usr = self.knowledge.kernel.budget_ellipse_a * module * np.cos(theta)
        y_usr = self.knowledge.kernel.budget_ellipse_b * module * np.sin(theta)
        x_wgs = (self.knowledge.kernel.budget_middle_location.x +
                 x_usr * np.cos(self.knowledge.kernel.budget_ellipse_angle) -
                 y_usr * np.sin(self.knowledge.kernel.budget_ellipse_angle))
        y_wgs = (self.knowledge.kernel.budget_middle_location.y +
                 x_usr * np.sin(self.knowledge.kernel.budget_ellipse_angle) +
                 y_usr * np.cos(self.knowledge.kernel.budget_ellipse_angle))
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
            return TreeNode(location, node, knowledge=self.knowledge)
        else:
            angle = np.math.atan2(location.y - node.location.y, location.x - node.location.x)
            x = node.location.x + self.knowledge.step_size * np.cos(angle)
            y = node.location.y + self.knowledge.step_size * np.sin(angle)
            location_next = Location(x, y)
        return TreeNode(location_next, node, knowledge=self.knowledge)

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
            if self.isIntersect(self.nodes[i], node_current):
                distance_between_nodes.append(np.inf)
            else:
                distance_between_nodes.append(self.get_distance_between_nodes(self.nodes[i], node_current))
        ind_neighbours = np.where(np.array(distance_between_nodes) <= self.radius_neighbour)[0]
        return ind_neighbours

    def get_cost_between_nodes(self, node1, node2):
        cost = (node1.cost +
                self.get_distance_between_nodes(node1, node2) +
                self.get_cost_from_cost_valley(node1, node2))
        return cost

    def get_cost_from_cost_valley(self, node1, node2):
        F1 = self.knowledge.kernel.get_ind_F(node1.location)
        F2 = self.knowledge.kernel.get_ind_F(node2.location)
        cost1 = self.knowledge.kernel.cost_valley[F1]
        cost2 = self.knowledge.kernel.cost_valley[F2]
        cost_total = ((cost1 + cost2) / 2 * self.get_distance_between_nodes(node1, node2))
        return cost_total

    def isarrived(self, node):
        dist = self.get_distance_between_nodes(self.ending_node, node)
        if dist < DISTANCE_TOLERANCE:
            return True
        else:
            return False

    def set_obstacles(self):
        for i in range(self.obstacles.shape[0]):
            self.polygon_obstacles.append(Polygon(list(map(tuple, self.obstacles[i]))))

    '''
    Collision detection
    '''
    def isWithin(self, node):
        point = Point(node.location.x, node.location.y)
        within = False
        for i in range(len(self.polygon_obstacles)):
            if self.polygon_obstacles[i].contains(point):
                within = True
        return within

    def isIntersect(self, node1, node2):
        line = LineString([(node1.location.x, node1.location.y),
                           (node2.location.x, node2.location.y)])
        intersect = False
        for i in range(len(self.polygon_obstacles)):
            if self.polygon_obstacles[i].intersects(line):
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
        self.trajectory = np.array(self.trajectory)

    def plot_tree(self):
        # plt.figure()
        if np.any(self.obstacles):
            for i in range(len(self.obstacles)):
                obstacle = np.append(self.obstacles[i], self.obstacles[i][0, :].reshape(1, -1), axis=0)
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

