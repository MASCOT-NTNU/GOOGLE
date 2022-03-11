"""
This script creates the strategies using RRT*
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-10
"""
import numpy as np

from usr_func import *
from GOOGLE.Simulation_Nidelva.Tree.TreeNode import TreeNode
from GOOGLE.Simulation_Nidelva.Config.Config import *
from GOOGLE.Simulation_Nidelva.Tree.Location import *
FIGPATH = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/fig/Sim_Nidelva/rrtstar/"


class RRTStar:

    def __init__(self, knowledge=None):
        self.knowledge = knowledge
        # self.obstacles = np.array(OBSTACLES)
        # self.polygon_obstacles = []
        # self.set_obstacles()
        self.nodes = []
        self.trajectory = []
        self.starting_node = TreeNode(self.knowledge.starting_location, None, 0, self.knowledge)
        self.ending_node = TreeNode(self.knowledge.ending_location, None, 0, self.knowledge)
        self.goal_node = TreeNode(self.knowledge.goal_location, None, 0, self.knowledge)
        # self.get_budget_field()
        self.counter_fig = 0
        self.get_bigger_box()

    def expand_trees(self):
        self.nodes.append(self.starting_node)
        self.arrived_signal = False
        for i in range(self.knowledge.maximum_iteration):
            print("Iteration: ", i)
            if np.random.rand() <= self.knowledge.goal_sample_rate:
                new_location = self.knowledge.ending_location
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
                self.arrived_signal = True
                self.ending_node.parent = next_node
                self.trajectory = []
                self.get_shortest_trajectory()
            else:
                self.nodes.append(next_node)

            # == 2D plot
            plt.plot(self.knowledge.polygon_border[:, 1], self.knowledge.polygon_border[:, 0], 'k-', linewidth=1)
            plt.plot(self.knowledge.polygon_obstacle[:, 1], self.knowledge.polygon_obstacle[:, 0], 'k-', linewidth=1)

            for node in self.nodes:
                if node.parent is not None:
                    plt.plot([node.location.lon, node.parent.location.lon],
                             [node.location.lat, node.parent.location.lat], "-g")

            if self.arrived_signal:
                trajectory = np.array(self.trajectory)
                plt.plot(trajectory[:, 1], trajectory[:, 0], "-r")
            plt.plot(self.knowledge.starting_location.lon, self.knowledge.starting_location.lat, 'kv', ms=10)
            plt.plot(self.knowledge.ending_location.lon, self.knowledge.ending_location.lat, 'bx', ms=10)
            plt.grid()
            plt.title("rrt*")
            plt.savefig(FIGPATH + "P_{:04d}.png".format(i))
            plt.close("all")


            # == 3D plot
            # fig = go.Figure(data=[go.Scatter3d(
            #     x=self.knowledge.polygon_border[:, 1],
            #     y=self.knowledge.polygon_border[:, 0],
            #     z=np.zeros_like(self.knowledge.polygon_border[:, 0]),
            #     mode='lines',
            #     line=dict(
            #         width=2,
            #         color='darkblue',
            #     )
            # )])
            #
            # fig.add_trace(go.Scatter3d(
            #     x=self.knowledge.polygon_obstacle[:, 1],
            #     y=self.knowledge.polygon_obstacle[:, 0],
            #     z=np.zeros_like(self.knowledge.polygon_obstacle[:, 0]),
            #     mode='lines',
            #     line=dict(
            #         width=2,
            #         color='darkblue',
            #     )
            # ))
            #
            # for node in self.nodes:
            #     if node.parent is not None:
            #         # plt.plot([node.location.lon, node.parent.location.lon],
            #         #          [node.location.lat, node.parent.location.lat], "-g")
            #         fig.add_trace(go.Scatter3d(
            #             x=[node.location.lon, node.parent.location.lon],
            #             y=[node.location.lat, node.parent.location.lat],
            #             z=[node.location.depth, node.parent.location.depth],
            #             mode='lines',
            #             line=dict(
            #                 width=1,
            #                 color='green',
            #             )
            #         ))
            #
            # if self.arrived_signal:
            #     trajectory = np.array(self.trajectory)
            #     fig.add_trace(go.Scatter3d(
            #         x=trajectory[:, 1],
            #         y=trajectory[:, 0],
            #         z=trajectory[:, 2],
            #         mode='lines',
            #         line=dict(
            #             width=1,
            #             color='red',
            #         )
            #     ))
            #
            # fig.add_trace(go.Scatter3d(
            #     x=[self.knowledge.starting_location.lon],
            #     y=[self.knowledge.starting_location.lat],
            #     z=[self.knowledge.starting_location.depth],
            #     mode='markers',
            #     marker=dict(
            #         size = 20,
            #         color='blue',
            #     )
            # ))
            #
            # fig.add_trace(go.Scatter3d(
            #     x=[self.knowledge.ending_location.lon],
            #     y=[self.knowledge.ending_location.lat],
            #     z=[self.knowledge.ending_location.depth],
            #     mode='markers',
            #     marker=dict(
            #         size = 20,
            #         color='black',
            #     )
            # ))
            # # fig.update_scenes(xaxis_visible=False, yaxis_visible=False,zaxis_visible=False)
            # plotly.offline.plot(fig, filename=FIGPATH+"P_{:03d}.html".format(i), auto_open=False)


    def get_bigger_box(self):
        self.box_lat_min, self.box_lon_min, self.box_depth_min = map(np.amin, [self.knowledge.polygon_border[:, 0],
                                                                               self.knowledge.polygon_border[:, 1],
                                                                               self.knowledge.depth])
        self.box_lat_max, self.box_lon_max, self.box_depth_max = map(np.amax, [self.knowledge.polygon_border[:, 0],
                                                                               self.knowledge.polygon_border[:, 1],
                                                                               self.knowledge.depth])

    def get_new_location(self):
        while True:
            lat = np.random.uniform(self.box_lat_min, self.box_lat_max)
            lon = np.random.uniform(self.box_lon_min, self.box_lon_max)
            if self.is_location_within_border(Location(lat, lon)):
                depth = np.random.uniform(self.box_depth_min, self.box_depth_max)
                location = Location(lat, lon, depth)
                return location

    def get_nearest_node(self, nodes, location):
        dist = []
        node_new = TreeNode(location)
        for node in nodes:
            dist.append(self.get_distance_between_nodes(node, node_new))
        return nodes[dist.index(min(dist))]

    def get_next_node(self, node, location):
        node_temp = TreeNode(location)
        if self.get_distance_between_nodes(node, node_temp) <= self.knowledge.step_size_total:
            location_next = location
        else:
            x, y = latlon2xy(location.lat, location.lon, node.location.lat, node.location.lon)
            angle_lateral = np.math.atan2(x, y)
            y_new = self.knowledge.step_size_lateral * np.cos(angle_lateral)
            x_new = self.knowledge.step_size_lateral * np.sin(angle_lateral)
            angle_vertical = np.math.atan2(location.depth - node.location.depth, np.sqrt(x ** 2 + y ** 2))
            z_new = self.knowledge.step_size_vertical * np.sin(angle_vertical)
            lat_new, lon_new = xy2latlon(x_new, y_new, node.location.lat, node.location.lon)
            depth_new = node.location.depth + z_new
            location_next = Location(lat_new, lon_new, depth_new)

        return TreeNode(location_next, node, knowledge=self.knowledge)

    @staticmethod
    def get_distance_between_nodes(node1, node2):
        dist_x, dist_y = latlon2xy(node1.location.lat, node1.location.lon, node2.location.lat, node2.location.lon)
        dist_z = node1.location.depth - node2.location.depth
        dist = np.sqrt(dist_x ** 2 + dist_y ** 2 + dist_z ** 2)
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
        ind_neighbours = np.where(np.array(distance_between_nodes) <= RADIUS_NEIGHBOUR)[0]
        return ind_neighbours

    def get_cost_between_nodes(self, node1, node2):
        cost = (node1.cost +
                self.get_distance_between_nodes(node1, node2))
                # self.get_cost_according_to_budget(node1, node2) +
                # self.get_reward_between_nodes(node1, node2))
                # self.get_cost_along_path(node1, node2))
               # RRTStar.get_cost_along_path(node1.location, node2.location)
        # print("Cost: ", cost)
        return cost

    # def get_reward_between_nodes(self, node1, node2):
    #     self.get_eibv_at_node(node1)
    #     self.get_eibv_at_node(node2)
    #     reward = ((node1.knowledge.EIBV + node2.knowledge.EIBV) / 2 *
    #               self.get_distance_between_nodes(node1, node2))
    #     return reward
    #
    # def get_eibv_at_node(self, node):
    #     node.knowledge.F = self.knowledge.kernel.get_ind_F(node.location)
    #     node.knowledge.EIBV = self.knowledge.kernel.eibv[node.knowledge.F]
    #
    # def get_cost_according_to_budget(self, node1, node2):
    #     cost1 = self.get_cost_from_budget_field(node1)
    #     cost2 = self.get_cost_from_budget_field(node2)
    #     cost_total = ((cost1 + cost2) / 2 *
    #                   self.get_distance_between_nodes(node1, node2))
    #     return cost_total
    #
    # def get_cost_from_budget_field(self, node):
    #     x_wgs = node.location.x - self.budget_middle_location.x
    #     y_wgs = node.location.y - self.budget_middle_location.y
    #     x_usr = (x_wgs * np.cos(self.budget_ellipse_angle) +
    #              y_wgs * np.sin(self.budget_ellipse_angle))
    #     y_usr = (- x_wgs * np.sin(self.budget_ellipse_angle) +
    #              y_wgs * np.cos(self.budget_ellipse_angle))
    #
    #     if (x_usr / self.budget_ellipse_a) ** 2 + (y_usr / self.budget_ellipse_b) ** 2 <= 1:
    #         return 0
    #     else:
    #         return np.inf
    #
    # def get_budget_field(self):
    #     print("Starting location: ", self.starting_node.location.x, self.starting_node.location.y)
    #     print("Ending location: ", self.goal_node.location.x, self.goal_node.location.y)
    #     self.budget_middle_location = self.get_middle_location(self.starting_node, self.goal_node)
    #     self.budget_ellipse_angle = self.get_angle_between_locations(self.starting_node, self.goal_node)
    #
    #     self.budget_ellipse_a = self.starting_node.knowledge.budget / 2
    #     self.budget_ellipse_c = self.get_distance_between_nodes(self.starting_node, self.goal_node) / 2
    #     self.budget_ellipse_b = np.sqrt(self.budget_ellipse_a ** 2 - self.budget_ellipse_c ** 2)
    #     print("a: ", self.budget_ellipse_a, "b: ", self.budget_ellipse_b, "c: ", self.budget_ellipse_c)
    #
    # @staticmethod
    # def get_middle_location(node1, node2):
    #     x_middle = (node1.location.x + node2.location.x) / 2
    #     y_middle = (node1.location.y + node2.location.y) / 2
    #     return Location(x_middle, y_middle)
    #
    # @staticmethod
    # def get_angle_between_locations(node1, node2):
    #     delta_y = node2.location.y - node1.location.y
    #     delta_x = node2.location.x - node1.location.x
    #     angle = np.math.atan2(delta_y, delta_x)
    #     return angle

    # @staticmethod
    # def get_cost_along_path(node1, node2):
    #     N = 10
    #     x = np.linspace(node1.location.x, node2.location.x, N)
    #     y = np.linspace(node1.location.y, node2.location.y, N)
    #     cost = []
    #     for i in range(N):
    #         cost.append(RRTStar.get_cost_from_field(Location(x[i], y[i])))
    #     cost_total = np.trapz(cost) / N
    #     # cost_total = np.sum(cost) / RRTStar.get_distance_between_nodes(TreeNode(location1), TreeNode(location2))
    #     # print("cost total: ", cost_total)
    #     return cost_total
    #
    # @staticmethod
    # def get_cost_from_field(location):
    #     # return 0
    #     # return 2**2 * np.exp(-((location.x - .5) ** 2 + (location.y - .5) ** 2) / .09)
    #     # return (1 - np.exp(-((location.x - .25) ** 2 + (location.y - .75) ** 2) / .09) +
    #     #         2 ** 2 * np.exp(-((location.x - .25) ** 2 + (location.y - .25) ** 2) / .09))
    #     if ((location.x - .5)/ .4) ** 2 + ((location.y - .5)/ .2) ** 2 <= 1:
    #         return 0
    #     else:
    #         return 40
    #     # return (location.x ** 2 / )

    def isarrived(self, node):
        dist = self.get_distance_between_nodes(self.ending_node, node)
        if dist < DISTANCE_TOLERANCE:
            return True
        else:
            return False

    '''
    Collision detection
    '''
    # def isWithin(self, node):
    #     point = Point(node.location.x, node.location.y)
    #     within = False
    #     for i in range(len(self.polygon_obstacles)):
    #         if self.polygon_obstacles[i].contains(point):
    #             within = True
    #     return within

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
        if self.knowledge.polygon_obstacle_path.intersects(line) or self.knowledge.polygon_border_path.intersects(line):
            intersect = True
        return intersect
    '''
    End of collision detection
    '''

    def get_shortest_trajectory(self):
        self.trajectory.append([self.ending_node.location.lat,
                                self.ending_node.location.lon,
                                self.ending_node.location.depth])
        pointer_node = self.ending_node
        while pointer_node.parent is not None:
            node = pointer_node.parent
            self.trajectory.append([node.location.lat,
                                    node.location.lon,
                                    node.location.depth])
            pointer_node = node
        self.trajectory = np.array(self.trajectory)

    def plot_tree(self):
        # plt.figure()
        plt.plot(self.knowledge.polygon_border[:, 1], self.knowledge.polygon_border[:, 0], 'k-', linewidth=1)
        plt.plot(self.knowledge.polygon_obstacle[:, 1], self.knowledge.polygon_obstacle[:, 0], 'k-', linewidth=1)

        for node in self.nodes:
            if node.parent is not None:
                plt.plot([node.location.lon, node.parent.location.lon],
                         [node.location.lat, node.parent.location.lat], "-g")

        trajectory = np.array(self.trajectory)
        plt.plot(trajectory[:, 1], trajectory[:, 0], "-r")

        plt.plot(self.knowledge.starting_location.lon, self.knowledge.starting_location.lat, 'kv', ms=10)
        plt.plot(self.knowledge.ending_location.lon, self.knowledge.ending_location.lat, 'bx', ms=10)
        # middle_location = self.get_middle_location(self.starting_node, self.goal_node)
        # ellipse = Ellipse(xy=(middle_location.x, middle_location.y), width=2*self.budget_ellipse_a,
        #                   height=2*self.budget_ellipse_b, angle=math.degrees(self.budget_ellipse_angle),
        #                   edgecolor='r', fc='None', lw=2)
        # plt.gca().add_patch(ellipse)
        plt.grid()
        plt.title("rrt*")
        # plt.savefig(FIGPATH + "T_{:04d}.png".format(self.counter_fig))
        # plt.show()


