from shapely.geometry import Polygon, Point
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import os
from unittest import TestCase
from WaypointGraph import WaypointGraph
from matplotlib.gridspec import GridSpec


class Node:

    def __init__(self, loc, parent=None):
        self.x, self.y = loc
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0


class AStar:
    DISTANCE_TOLERANCE_TARGET = .075
    DISTANCE_TOLERANCE = .02
    def __init__(self, plg_border=None, plg_obstacle=None):
        self.plg_border = plg_border
        self.plg_obstacle = plg_obstacle
        self.plg_border_shapely = Polygon(self.plg_border)
        self.plg_obstacle_shapely = Polygon(self.plg_obstacle)
        self.max_iter = 2000
        self.cnt = 0
        self.stepsize = .05
        self.open_list = []
        self.closed_list = []
        self.path = []
        self.arrival = False
        self.figpath = os.getcwd() + "/../../fig/astar/"

    def search_path(self, loc_start, loc_end):

        angles = np.arange(0, 360, 60)

        # s1: initialise nodes
        self.start_node = Node(loc_start, None)
        self.start_node.g = self.start_node.h = self.start_node.f = 0
        self.end_node = Node(loc_end, None)
        self.end_node.g = self.end_node.h = self.end_node.f = 0
        self.open_list = []
        self.closed_list = []

        # s2: append open list
        self.open_list.append(self.start_node)

        # s3: loop open list
        while len(self.open_list) > 0:
            # s31: find smallest cost node and append this to closed list and remove it from open list.
            node_now = self.get_min_cost_node()
            self.closed_list.append(node_now)

            if self.is_arrived(node_now):
                pointer = node_now
                while pointer is not None:
                    self.path.append([pointer.x, pointer.y])
                    pointer = pointer.parent

            # s32: produce children and then start allocating locations.
            children = []
            for angle in angles:
                ang = math.radians(angle)
                xn = node_now.x + np.cos(ang) * self.stepsize
                yn = node_now.y + np.sin(ang) * self.stepsize
                loc_n = [xn, yn]
                if self.is_within_obstacle(loc_n) or not self.is_within_border(loc_n):
                    continue
                node_new = Node(loc_n, node_now)
                children.append(node_new)

            # s33: loop through all children to filter illegal points.
            for child in children:
                if self.is_node_in_list(child, self.closed_list):
                    continue

                child.g = node_now.g + self.stepsize
                child.h = self.get_distance_between_nodes(child, self.end_node)
                child.f = child.g + child.h

                if self.is_node_in_list(child, self.open_list):
                    # print("skip")
                    continue

                self.open_list.append(child)

            fig = plt.figure(figsize=(10, 10))
            gs = GridSpec(nrows=1, ncols=1)

            def plf():
                plt.plot(self.plg_border[:, 0], self.plg_border[:, 1], 'r-.')
                plt.plot(self.plg_obstacle[:, 0], self.plg_obstacle[:, 1], 'r-.')
                plt.plot(self.start_node.x, self.start_node.y, 'k.', markersize=25)
                plt.plot(self.end_node.x, self.end_node.y, 'b*', markersize=25)
                if self.arrival:
                    pa = np.array(self.path)
                    plt.plot(pa[:, 0], pa[:, 1], 'r.-', markersize=20)

            ax = fig.add_subplot(gs[0])
            plf()
            for item in self.open_list:
                ax.plot(item.x, item.y, 'c.', alpha=.2, markersize=20)
            for item in self.closed_list:
                ax.plot(item.x, item.y, 'k.', alpha=.2, markersize=20)
            plt.plot(node_now.x, node_now.y, 'g.', markersize=20)
            for child in children:
                ax.plot(child.x, child.y, 'r.', alpha=.2, markersize=20)

            plt.savefig(self.figpath + "P_{:03d}.png".format(self.cnt))
            plt.close("all")

            print("cnt: ", self.cnt)
            self.cnt += 1
            if self.cnt > self.max_iter:
                print("Cannot converge")
                break

            if self.arrival:
                print("Arrived")
                break
            pass
        pass

    def get_min_cost_node(self):
        min_node = self.open_list[0]
        for node in self.open_list:
            if node.f < min_node.f:
                min_node = node
        self.open_list.remove(min_node)
        return min_node

    def is_arrived(self, node):
        dist = self.get_distance_between_nodes(node, self.end_node)
        if dist <= AStar.DISTANCE_TOLERANCE_TARGET:
            self.arrival = True
            return True
        else:
            return False

    def is_node_in_list(self, node, l):
        for e in l:
            dist = self.get_distance_between_nodes(node, e)
            if dist <= AStar.DISTANCE_TOLERANCE:
                return True
        return False

    def is_within_border(self, loc):
        point = Point(loc[0], loc[1])
        return self.plg_border_shapely.contains(point)

    def is_within_obstacle(self, loc):
        point = Point(loc[0], loc[1])
        return self.plg_obstacle_shapely.contains(point)

    @staticmethod
    def get_distance_between_nodes(n1, n2):
        return np.sqrt((n1.x - n2.x)**2 + (n1.y - n2.y)**2)


class TestAstar(TestCase):
    def setUp(self) -> None:
        self.plg_border = np.array([[0, 0],
                               [0, 1],
                               [1, 1],
                               [1, 0],
                               [0, 0]])

        # self.plg_obstacle = np.array([[.5, .5],
        #                               [.51, .5],
        #                               [.51, .51],
        #                               [.5, .51],
        #                               [.5, .5]])
        # self.plg_obstacle = np.array([[.25, .25],
        #                          [.65, .25],
        #                          [.65, .65],
        #                          [.25, .65],
        #                          [.25, .25]])
        self.plg_obstacle = np.array([[.21, .21],
                                      [.41, .21],
                                      [.41, .61],
                                      [.61, .61],
                                      [.61, .21],
                                      [.81, .21],
                                      [.81, .81],
                                      [.21, .81],
                                      [.21, .21]])
        self.wp = WaypointGraph()
        self.wp.set_polygon_border(self.plg_border)
        self.wp.set_polygon_obstacles([self.plg_obstacle])
        self.wp.set_depth_layers([0])
        self.wp.set_neighbour_distance(.05)
        self.wp.construct_waypoints()
        self.wp.construct_hash_neighbours()
        self.waypoint = self.wp.get_waypoints()

        self.astar = AStar(self.plg_border, self.plg_obstacle)

    def test_astar(self):
        loc_start = [.01, .01]
        loc_end = [.99, .99]
        self.astar.search_path(loc_start, loc_end)




