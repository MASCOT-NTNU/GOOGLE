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
# FIGPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/fig/Sim_Nidelva/rrtstar/"


class RRTStar:

    def __init__(self, knowledge=None):
        self.knowledge = knowledge
        self.nodes = []
        self.trajectory = []

        self.starting_node = TreeNode(self.knowledge.starting_location, None, 0, self.knowledge)
        self.ending_node = TreeNode(self.knowledge.ending_location, None, 0, self.knowledge)
        self.goal_node = TreeNode(self.knowledge.goal_location, None, 0, self.knowledge)

        self.counter_fig = 0
        self.get_bigger_box()

    def expand_trees(self):
        self.nodes.append(self.starting_node)
        t1 = time.time()
        for i in range(self.knowledge.maximum_iteration):
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

            if self.is_node_within_obstacle(next_node):
                continue

            next_node, nearest_node = self.rewire_tree(next_node, nearest_node)

            if self.is_path_intersect_with_obstacles(nearest_node, next_node):
                continue

            if self.isarrived(next_node):
                self.ending_node.parent = next_node
            else:
                self.nodes.append(next_node)
        t2 = time.time()
        print("Finished tree expansion, time consumed: ", t2 - t1)

    def get_bigger_box(self):
        self.box_lat_min, self.box_lon_min = map(np.amin, [self.knowledge.polygon_border[:, 0],
                                                           self.knowledge.polygon_border[:, 1]])
        self.box_lat_max, self.box_lon_max = map(np.amax, [self.knowledge.polygon_border[:, 0],
                                                           self.knowledge.polygon_border[:, 1]])

    def get_new_location(self):
        while True:
            lat = np.random.uniform(self.box_lat_min, self.box_lat_max)
            lon = np.random.uniform(self.box_lon_min, self.box_lon_max)
            if self.is_location_within_border(Location(lat, lon)):
                location = Location(lat, lon)
                return location

    def get_new_location_within_budget_ellipse(self):
        # t1 = time.time()
        theta = np.random.uniform(0, 2 * np.pi)
        module = np.sqrt(np.random.rand())
        y_usr = self.knowledge.kernel.budget_ellipse_a * module * np.cos(theta)
        x_usr = self.knowledge.kernel.budget_ellipse_b * module * np.sin(theta)
        y_wgs = (self.knowledge.kernel.budget_middle_location.y +
                 y_usr * np.cos(self.knowledge.kernel.budget_ellipse_angle) -
                 x_usr * np.sin(self.knowledge.kernel.budget_ellipse_angle))
        x_wgs = (self.knowledge.kernel.budget_middle_location.x +
                 y_usr * np.sin(self.knowledge.kernel.budget_ellipse_angle) +
                 x_usr * np.cos(self.knowledge.kernel.budget_ellipse_angle))
        lat, lon = xy2latlon(x_wgs, y_wgs, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        # t2 = time.time()
        # print("Generating location takes: ", t2 - t1)
        return Location(lat, lon)

    def get_nearest_node(self, nodes, location):
        dist = []
        node_new = TreeNode(location)
        for node in nodes:
            dist.append(self.get_distance_between_nodes(node, node_new))
        return nodes[dist.index(min(dist))]

    def get_next_node(self, node, location):
        node_temp = TreeNode(location)
        if self.get_distance_between_nodes(node, node_temp) <= self.knowledge.step_size:
            location_next = location
        else:
            x, y = latlon2xy(location.lat, location.lon, node.location.lat, node.location.lon)
            angle = np.math.atan2(x, y)
            y_new = self.knowledge.step_size * np.cos(angle)
            x_new = self.knowledge.step_size * np.sin(angle)
            lat_new, lon_new = xy2latlon(x_new, y_new, node.location.lat, node.location.lon)
            location_next = Location(lat_new, lon_new)
        return TreeNode(location_next, node, knowledge=self.knowledge)

    @staticmethod
    def get_distance_between_nodes(node1, node2):
        dist_x, dist_y = latlon2xy(node1.location.lat, node1.location.lon,
                                   node2.location.lat, node2.location.lon)
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
            if self.is_path_intersect_with_obstacles(self.nodes[i], node_current):
                distance_between_nodes.append(np.inf)
            else:
                distance_between_nodes.append(self.get_distance_between_nodes(self.nodes[i], node_current))
        ind_neighbours = np.where(np.array(distance_between_nodes) <= self.knowledge.distance_neighbour_radar)[0]
        return ind_neighbours

    def get_cost_between_nodes(self, node1, node2):
        cost = (node1.cost +
                self.get_distance_between_nodes(node1, node2))
        return cost

    def isarrived(self, node):
        dist = self.get_distance_between_nodes(self.ending_node, node)
        if dist < DISTANCE_TOLERANCE:
            return True
        else:
            return False

    '''
    Collision detection
    '''
    def is_location_within_border(self, location):
        point = Point(location.lat, location.lon)
        return self.knowledge.polygon_border_path.contains(point)

    def is_node_within_obstacle(self, node):
        point = Point(node.location.lat, node.location.lon)
        return self.knowledge.polygon_obstacle_path.contains(point)

    def is_path_intersect_with_obstacles(self, node1, node2):
        line = LineString([(node1.location.lat, node1.location.lon),
                           (node2.location.lat, node2.location.lon)])
        intersect = False
        if self.knowledge.polygon_obstacle_path.intersects(line) or self.knowledge.borderline_path.intersects(line):
            intersect = True
        return intersect
    '''
    End of collision detection
    '''

    def get_shortest_trajectory(self):
        self.trajectory.append([self.ending_node.location.lat,
                                self.ending_node.location.lon])
        pointer_node = self.ending_node
        while pointer_node.parent is not None:
            node = pointer_node.parent
            self.trajectory.append([node.location.lat,
                                    node.location.lon])
            pointer_node = node
        self.trajectory = np.array(self.trajectory)

    def plot_tree(self):
        # plt.figure()
        # x, y = latlon2xy(self.knowledge.polygon_border[:, 0], self.knowledge.polygon_border[:, 1],
        #                  LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        plt.plot(self.knowledge.polygon_border[:, 1], self.knowledge.polygon_border[:, 0], 'k-', linewidth=1)
        # x, y = latlon2xy(self.knowledge.polygon_obstacle[:, 0], self.knowledge.polygon_obstacle[:, 1],
        #                  LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        plt.plot(self.knowledge.polygon_obstacle[:, 1], self.knowledge.polygon_obstacle[:, 0], 'k-', linewidth=1)

        for node in self.nodes:
            if node.parent is not None:
                plt.plot([node.location.lon, node.parent.location.lon],
                         [node.location.lat, node.parent.location.lat], "-g")

        trajectory = np.array(self.trajectory)
        # x, y = latlon2xy(trajectory[:, 0], trajectory[:, 1], LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        plt.plot(trajectory[:, 1], trajectory[:, 0], "-r")

        plt.plot(self.knowledge.starting_location.lon, self.knowledge.starting_location.lat, 'kv', ms=10)
        plt.plot(self.knowledge.ending_location.lon, self.knowledge.ending_location.lat, 'bx', ms=10)
        # middle_location = self.get_middle_location(self.starting_node, self.goal_node)
        # ellipse = Ellipse(xy=(middle_location.x, middle_location.y), width=2*self.budget_ellipse_a,
        #                   height=2*self.budget_ellipse_b, angle=math.degrees(self.budget_ellipse_angle),
        #                   edgecolor='r', fc='None', lw=2)
        # plt.gca().add_patch(ellipse)
        plt.grid(which='minor', alpha=0.2)
        plt.title("rrt*")
        # plt.savefig(FIGPATH + "T_{:04d}.png".format(self.counter_fig))
        # plt.show()



