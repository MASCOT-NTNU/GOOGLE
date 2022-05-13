"""
This script tests the rrt* algorithm for collision avoidance
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-25
"""

from usr_func import *
from shapely.geometry import Point, Polygon, LineString


class Location:

    def __init__(self, lat=None, lon=None, depth=None):
        self.lat = lat
        self.lon = lon
        self.depth = depth


class TreeNode:

    def __init__(self, location=None, parent=None, cost=None):
        self.location = location
        self.parent = parent
        self.cost = cost


class RRTConfig:

    def __init__(self, polygon_within=None, polygon_without=None, depth=None, starting_location=None,
                 ending_location=None, goal_sample_rate=None, step_lateral=None, step_vertical=None,
                 maximum_num=1000, neighbour_radius=None, distance_tolerance=None):
        self.polygon_within = polygon_within
        self.polygon_without = polygon_without
        self.polygon_within_path = mplPath.Path(self.polygon_within)
        self.polygon_without_path = Polygon(self.polygon_without)
        self.depth = depth
        self.starting_location = starting_location
        self.ending_location = ending_location
        self.goal_sample_rate = goal_sample_rate
        self.step_lateral = step_lateral
        self.step_vertical = step_vertical
        self.maximum_num = maximum_num
        self.neighbour_radius = neighbour_radius
        self.distance_tolerance = distance_tolerance


class RRTStar:

    def __init__(self, config=None):
        self.config = config
        self.path = []
        self.starting_node = TreeNode(self.config.starting_location, None, 0)
        self.ending_node = TreeNode(self.config.ending_location, None, 0)
        self.nodes = []

        self.get_bigger_box()
        self.expand_trees()

    def expand_trees(self):
        self.nodes.append(self.starting_node)
        self.counter = 0
        for i in range(self.config.maximum_num):
            print("Iteration: ", i)
            if np.random.rand() <= self.config.goal_sample_rate:
                new_location = self.config.ending_location
            else:
                new_location = self.get_new_location()

            nearest_node = self.get_nearest_node(self.nodes, new_location)
            next_node = self.get_next_node(nearest_node, new_location)
            next_node, nearest_node = self.rewire_tree(next_node, nearest_node)

            if self.iscollided(next_node):
                continue

            if self.isarrived(next_node):
                self.ending_node.parent = next_node
                self.nodes.append(self.ending_node)
                break
            else:
                self.nodes.append(next_node)
            pass
        pass

    def get_bigger_box(self):
        self.box_lat_min, self.box_lon_min, self.box_depth_min = map(np.amin, [self.config.polygon_border_xy[:, 0],
                                                                               self.config.polygon_border_xy[:, 1],
                                                                               self.config.depth])
        self.box_lat_max, self.box_lon_max, self.box_depth_max = map(np.amax, [self.config.polygon_border_xy[:, 0],
                                                                               self.config.polygon_border_xy[:, 1],
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

    def rewire_tree(self, node_new, node_nearest):
        for i in range(len(self.nodes)):
            if RRTStar.get_distance_between_nodes(self.nodes[i], node_new) <= self.config.distance_neighbour_radar:
                if self.nodes[i].cost + RRTStar.get_distance_between_nodes(self.nodes[i], node_new) < \
                        node_nearest.cost + RRTStar.get_distance_between_nodes(node_nearest, node_new):
                    node_nearest = self.nodes[i]
        node_new.cost = node_nearest.cost + RRTStar.get_distance_between_nodes(node_nearest, node_new)
        node_new.parent = node_nearest
        return node_new, node_nearest

    def isarrived(self, node):
        dist = self.get_distance_between_nodes(self.ending_node, node)
        if dist < self.config.distance_tolerance:
            return True
        else:
            return False

    def iscollided(self, node):
        #TODO: check collision detection
        point = Point(node.location.lat, node.location.lon)
        line = LineString([(node.parent.location.lat, node.parent.location.lon),
                           (node.location.lat, node.location.lon)])
        collision = False
        if self.config.polygon_obstacles_shapely.contains(point) or self.config.polygon_obstacles_shapely.intersects(line):
            collision = True
        return collision

    def isWithin(self, location):
        return self.config.polygon_border_shapely.contains_point(location)

    def get_shortest_path(self):
        print("here comes the path finding")
        self.path.append([self.ending_node.location.lat, self.ending_node.location.lon])
        pointer_node = self.ending_node
        counter = 0
        while pointer_node.parent is not None:
            print("counter: ", counter)
            counter += 1
            node = pointer_node.parent
            self.path.append([node.location.lat, node.location.lon])
            pointer_node = node

        self.path.append([self.starting_node.location.lat, self.starting_node.location.lon])

        print("Finished path reorganising")

        dist = 0
        if len(self.path) > 2:
            path = np.array(self.path)
            lat_prev, lon_prev = path[0, :]
            for i in range(len(path)):
                lat_now, lon_now = path[i, :]
                dist_x, dist_y = latlon2xy(lat_now, lon_now, lat_prev, lon_prev)
                dist += np.sqrt(dist_x ** 2 + dist_y ** 2)
                lat_prev, lon_prev = lat_now, lon_now
        print("Distance travelled: ", dist, "m")

    def plot_tree(self):

        plt.figure(figsize=(8, 8))
        plt.clf()
        plt.plot(self.config.polygon_obstacle_xy[:, 1], self.config.polygon_obstacle_xy[:, 0], 'k-', linewidth=4)
        plt.plot(self.config.polygon_border_xy[:, 1], self.config.polygon_border_xy[:, 0], 'k-', linewidth=4)

        for node in self.nodes:
            if node.parent is not None:
                plt.plot([node.location.lon, node.parent.location.lon],
                         [node.location.lat, node.parent.location.lat], "-g")
        path = np.array(self.path)
        plt.plot(path[:, 1], path[:, 0], "-b")
        plt.plot(self.config.starting_location.lon, self.config.starting_location.lat, 'k*', ms=10)
        plt.plot(self.config.ending_location.lon, self.config.ending_location.lat, 'g*', ms=10)
        plt.grid()
        plt.show()

