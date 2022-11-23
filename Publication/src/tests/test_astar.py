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

    def __init__(self, ind, parent=None):
        self.ind = ind
        self.parent = parent
        self.g = 0
        self.h = 0
        self.f = 0


def is_exist(value, list):
    return value in list


def astar(wp: 'WaypointGraph', ind_start, ind_end, border, obstacle):
    figpath = os.getcwd() + "/../../fig/astar/"

    start_node = Node(ind_start, None)
    start_node.cost = 0
    end_node = Node(ind_end, None)
    end_node.cost = 0

    stepsize = wp.get_neighbour_distance()
    maximum_iter = 300
    cnt = 0

    open_list = []
    closed_list = []
    ind_open = []
    ind_closed = []

    ind_open = []
    ind_closed = []

    waypoints = wp.get_waypoints()
    open_list.append(start_node)
    ind_open.append(ind_start)
    arrival = False

    while len(ind_open) > 0:
        # print(ind_open)

        node_now = open_list[0]
        ind_now = 0
        for i in range(len(ind_open)):
            if open_list[i].f < node_now.f:
                node_now = open_list[i]
                ind_now = i
        # for index, item in enumerate(open_list):
        #     if item.f < node_now.f:
        #         node_now = item
        #         ind_now = index

        # print("open before: ", open_list)
        closed_list.append(node_now)
        ind_closed.append(ind_open[ind_now])

        open_list.pop(ind_now)
        ind_open.pop(ind_now)
        # print("open after: ", open_list)

        # print("closed: ", closed_list)


        if node_now.ind == end_node.ind:
            arrival = True
            path = []
            pointer = node_now
            while pointer is not None:
                path.append([pointer.ind])
                pointer = pointer.parent
            # return path[::-1]

        if not arrival:
            children = []
            ind_neighbours = wp.get_ind_neighbours(node_now.ind)
            for idn in ind_neighbours:
                node_new = Node(idn, node_now)
                children.append(node_new)

            for child in children:
                if is_exist(child.ind, ind_closed):
                    continue
                # s1: compute g
                child.g = node_now.g + stepsize

                # s2: compute h
                wp1 = wp.get_waypoint_from_ind(child.ind)
                wp2 = wp.get_waypoint_from_ind(end_node.ind)
                # print("wp1: ", wp1)
                # print("wp2: ", wp2)
                child.h = (wp1[0] - wp2[0])**2 + (wp1[1] - wp2[1])**2

                # s3: compute f
                child.f = child.g + child.h

                if is_exist(child.ind, ind_open):
                    continue
                # for open_node in open_list:
                #     if open_node == child and child.g > open_node.g:
                #     # print(np.sqrt((open_node.x - child.x)**2 + (open_node.y - child.y)**2))
                # #     if open_node.ind == child.ind and child.g > open_node.g:
                #         continue

                open_list.append(child)
                ind_open.append(child.ind)

        fig = plt.figure(figsize=(35, 15))
        gs = GridSpec(nrows=1, ncols=3)

        def plf():
            # plt.plot(waypoints[:, 0], waypoints[:, 1], 'g.', alpha=.1, markersize=20)
            plt.plot(border[:, 0], border[:, 1], 'r-.')
            plt.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')
            loc = wp.get_waypoint_from_ind(ind_start)
            plt.plot(loc[0], loc[1], 'k.', markersize=25)
            loc = wp.get_waypoint_from_ind(ind_end)
            plt.plot(loc[0], loc[1], 'b*', markersize=25)
            if arrival:
                for p in path:
                    plt.plot(waypoints[p, 0], waypoints[p, 1], 'r.', markersize=20)

        ax = fig.add_subplot(gs[0])
        plf()
        for item in open_list:
            loc = wp.get_waypoint_from_ind(item.ind)
            ax.plot(loc[0], loc[1], 'c.', alpha=.2, markersize=20)
        loc = wp.get_waypoint_from_ind(node_now.ind)
        plt.plot(loc[0], loc[1], 'g.', markersize=20)


        ax = fig.add_subplot(gs[1])
        plf()
        for item in closed_list:
            loc = wp.get_waypoint_from_ind(item.ind)
            ax.plot(loc[0], loc[1], 'k.', alpha=.2, markersize=20)
        loc = wp.get_waypoint_from_ind(node_now.ind)
        plt.plot(loc[0], loc[1], 'g.', markersize=20)

        ax = fig.add_subplot(gs[2])
        plf()
        for child in children:
            loc = wp.get_waypoint_from_ind(child.ind)
            ax.plot(loc[0], loc[1], 'r.', alpha=.2, markersize=20)
        loc = wp.get_waypoint_from_ind(node_now.ind)
        plt.plot(loc[0], loc[1], 'g.', markersize=20)

        plt.savefig(figpath + "P_{:03d}.png".format(cnt))
        plt.close("all")

        print("cnt: ", cnt)
        cnt += 1
        if cnt > maximum_iter:
            print("Cannot converge")
            break

        if arrival:
            print("Arrived")
            break


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
        # plt.plot(self.waypoint[:, 1], self.waypoint[:, 0], 'k.')
        # plt.show()

    def test_astar(self):
        # pass
        loc_start = np.array([.1, .1, 0])
        loc_end = np.array([.8, .99, 0])
        ids = self.wp.get_ind_from_waypoint(loc_start)
        ide = self.wp.get_ind_from_waypoint(loc_end)
        # wps = self.wp.get_waypoint_from_ind(ids)
        # wpe = self.wp.get_waypoint_from_ind(ide)
        astar(self.wp, ind_start=ids, ind_end=ide,
              border=self.plg_border, obstacle=self.plg_obstacle)






