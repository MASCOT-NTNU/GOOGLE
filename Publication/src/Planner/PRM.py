"""
PRM conducts the path planning using probabilistic road map philosophy. It selects the minimum cost path between
the starting location and the end location.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import copy
from shapely.geometry import Point, Polygon, LineString


class Node:

    def __init__(self, loc=None, cost=None):
        self.x, self.y = loc
        self.cost = cost
        self.parent = None
        self.neighbours = []

class PRM:

    nodes = []
    # obstacles = np.array(OBSTACLES)
    polygon_obstacles = []

    def __init__(self, loc_start, loc_end, border, obstacle):
        self.loc_start = loc_start
        self.loc_end = loc_end
        self.plg_border = border
        self.plg_obstacle = obstacle
        self.plg_border_shapely = Polygon(self.plg_border)
        self.plg_obstacle_shapely = Polygon(self.plg_obstacle)
        self.num_nodes = 1000
        self.num_neighbours = 10
        self.xlim = [np.amin(border[:, 0]), np.amax(border[:, 0])]
        self.ylim = [np.amin(border[:, 1]), np.amax(border[:, 1])]
        self.path = []

    def get_road_map(self):
        # s1: initialise nodes
        self.starting_node = Node(self.loc_start)
        self.ending_node = Node(self.loc_end)

        # s2: get random locations
        self.nodes.append(self.starting_node)
        counter_nodes = 0
        while counter_nodes < self.num_nodes:
            new_location = self.get_new_location()
            if not self.inRedZone(new_location):
                self.nodes.append(Node(new_location))
                counter_nodes += 1
        self.nodes.append(self.ending_node)

        # s3: get road maps
        for i in range(len(self.nodes)):
            dist = []
            node_now = self.nodes[i]
            for j in range(len(self.nodes)):
                node_next = self.nodes[j]
                dist.append(PRM.get_distance_between_nodes(node_now, node_next))
            ind_sort = np.argsort(dist)
            # print(ind_sort[:self.config.num_neighbours])
            for k in range(self.num_neighbours):
                node_neighbour = self.nodes[ind_sort[:self.num_neighbours][k]]
                if not self.iscollided(node_now, node_neighbour):
                    node_now.neighbours.append(node_neighbour)

    # @staticmethod
    def get_new_location(self):
        while True:
            x = np.random.uniform(self.xlim[0], self.xlim[1])
            y = np.random.uniform(self.ylim[0], self.ylim[1])
            point = Point(x, y)
            if self.plg_border_shapely.contains(point):
                return [x, y]

    def inRedZone(self, location):
        x, y = location
        point = Point(x, y)
        collision = False
        for i in range(len(self.polygon_obstacles)):
            if self.polygon_obstacles[i].contains(point):
                collision = True
        return collision

    @staticmethod
    def get_distance_between_nodes(n1, n2):
        dist_x = n1.x - n2.x
        dist_y = n1.y - n2.y
        dist = np.sqrt(dist_x**2 + dist_y**2)
        return dist

    # def set_obstacles(self):
    #     for i in range(self.obstacles.shape[0]):
    #         self.polygon_obstacles.append(Polygon(list(map(tuple, self.obstacles[i]))))

    def iscollided(self, n1, n2):
        line = LineString([(n1.x, n1.y),
                           (n2.x, n2.y)])
        collision = False
        for i in range(len(self.polygon_obstacles)):
            if self.polygon_obstacles[i].intersects(line):
                collision = True
        return collision

    def get_shortest_path_using_dijkstra(self):
        self.unvisited_nodes = []
        for node in self.nodes:
            node.cost = np.inf
            node.parent = None
            self.unvisited_nodes.append(node)

        current_node = self.unvisited_nodes[0]
        current_node.cost = 0
        pointer_node = current_node

        while self.unvisited_nodes:
            ind_min_cost = PRM.get_ind_min_cost(self.unvisited_nodes)
            current_node = self.unvisited_nodes[ind_min_cost]

            for neighbour_node in current_node.neighbours:
                if neighbour_node in self.unvisited_nodes:
                    cost = current_node.cost + PRM.get_distance_between_nodes(current_node, neighbour_node)
                    if cost < neighbour_node.cost:
                        neighbour_node.cost = cost
                        neighbour_node.parent = current_node
            pointer_node = current_node
            self.unvisited_nodes.pop(ind_min_cost)

        self.path.append([pointer_node.x, pointer_node.y])

        while pointer_node.parent is not None:
            node = pointer_node.parent
            self.path.append([node.location.x, node.location.y])
            pointer_node = node

        self.path.append([self.starting_node.x, self.starting_node.y])

    @staticmethod
    def get_ind_min_cost(nodes):
        cost = []
        for node in nodes:
            cost.append(node.cost)
        return cost.index(min(cost))

    def plot_prm(self):
        plt.clf()
        for i in range(self.obstacles.shape[0]):
            obstacle = np.append(self.obstacles[i], self.obstacles[i][0, :].reshape(1, -1), axis=0)
            plt.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')

        for node in self.nodes:
            if node.neighbours is not None:
                for i in range(len(node.neighbours)):
                    plt.plot([node.location.x, node.neighbours[i].location.x],
                             [node.location.y, node.neighbours[i].location.y], "-g")
        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], "-r")
        plt.grid()
        plt.title("prm")
        plt.show()


if __name__ == "__main__":
    starting_loc = [0, 0]
    ending_loc = [1, 1]

    MAXNUM = 100
    XLIM = [0, 1]
    YLIM = [0, 1]
    GOAL_SAMPLE_RATE = .01
    STEP = .1
    RADIUS_NEIGHBOUR = .15
    DISTANCE_TOLERANCE = .11
    # OBSTACLES = [[[.1, .1], [.2, .1], [.2, .2], [.1, .2]],
    #              [[.4, .4], [.6, .5], [.5, .6], [.3, .4]],
    #              [[.8, .8], [.95, .8], [.95, .95], [.8, .95]]]
    OBSTACLES = [[[.1, .0], [.2, .0], [.2, .5], [.1, .5]],
                 [[.0, .6], [.6, .6], [.6, 1.], [.0, 1.]],
                 [[.8, .0], [1., .0], [1., .9], [.8, .9]],
                 [[.3, .1], [.4, .1], [.4, .6], [.3, .6]],
                 [[.5, .0], [.6, .0], [.6, .4], [.5, .4]]]

    FIGPATH = os.getcwd() + "/../../fig/prm/"

    prm = PRM(starting_loc, ending_loc, np.array([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]), np.array(OBSTACLES))
    prm.set_obstacles()
    prm.get_all_random_nodes()
    prm.get_road_maps()
    prm.get_shortest_path_using_dijkstra()
    # prm.get_shortest_path_using_astar()
    prm.plot_prm()
    pass




