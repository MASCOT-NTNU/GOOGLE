"""
This script generates regular grid points within a polygon boundary
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-23
"""

'''
Drawback of this algorithm is when the field gets larger, it will produce overlapping nodes, or nodes that 
are closely located with each other, see also the updated version using HexagonalGrid2D.py
'''


import matplotlib.path as mplPath  # used to determine whether a point is inside the grid or not
from GOOGLE.Field.Grid.gridNode import gridNode
from usr_func import *
import warnings


class GridGenerator:
    '''
    Generate grid within 2D dimension
    '''
    def __init__(self, polygon=None, depth=None, distance_neighbour=0, no_children=6, points_allowed = 1000):
        if polygon is None:
            raise ValueError("Polygon is not valid, please check it again")
        if depth is None:
            raise ValueError("Depth is not valid, please check it again")
        if distance_neighbour == 0:
            raise ValueError("Neighbour distance cannot be 0, please check it again")
        if no_children != 6:
            warnings.warn("Grid to be generated may not be regular")
        self.polygon = polygon
        self.depth = depth
        self.distance_neighbour = distance_neighbour
        self.points_allowed = points_allowed
        self.counter_grid = 0
        self.grid = []
        self.angle_neighbour = deg2rad(np.arange(no_children) * 360 / no_children)  # angles for polygon
        self.loc_start = [self.polygon[0, 0], self.polygon[0, 1]]
        self.polygon_path = mplPath.Path(self.polygon)

        # TODO: think of a better way of building grid
        self.get_rectangular_boundary()
        self.polygon_path = self.rectangular_boundary_path

        self.traverseField()
        self.getCoordinates()

    def get_rectangular_boundary(self):
        lat_min, lon_min = map(np.amin, [self.polygon[:, 0], self.polygon[:, 1]])
        lat_max, lon_max = map(np.amax, [self.polygon[:, 0], self.polygon[:, 1]])
        self.rectangular_boundary = np.array([[lat_min, lon_min],
                                              [lat_min, lon_max],
                                              [lat_max, lon_max],
                                              [lat_max, lon_min]])
        # self.rectangular_boundary_path = mplPath.Path(self.rectangular_boundary)

    def traverseField(self):
        lat_new, lon_new = self.getNewLocations(self.loc_start)
        start_node = []
        for i in range(len(self.angle_neighbour)):
            if self.polygon_path.contains_point((lat_new[i], lon_new[i])):
                start_node.append([lat_new[i], lon_new[i]])
                self.grid.append([lat_new[i], lon_new[i]])
                self.counter_grid = self.counter_grid + 1

        gridNode_start = gridNode(len(start_node), start_node, self.loc_start)
        allGridWithinField = self.traverseChildrenNodes(gridNode_start)
        self.grid = np.array(self.grid)
        if len(self.grid) > self.points_allowed:
            # print("{:d} grid points are generated, only {:d} waypoints are selected!".format(len(self.grid), self.points_allowed))
            self.grid = self.grid[:self.points_allowed, :]
        else:
            pass
            # print("{:d} grid points are generated, all are selected!".format(len(self.grid)))
        # print("grid shape: ", self.grid.shape)

    def getNewLocations(self, loc):
        '''
        get new locations around the current location
        '''
        lat_new, lon_new = xy2latlon(self.distance_neighbour * np.sin(self.angle_neighbour),
                                     self.distance_neighbour * np.cos(self.angle_neighbour), loc[0], loc[1])
        return lat_new, lon_new

    def traverseChildrenNodes(self, grid_node):
        if self.counter_grid > self.points_allowed:
            return gridNode(0, [], grid_node.grid_loc)
        for i in range(grid_node.subGrid_len):
            subsubGrid = []
            length_new = 0
            lat_subsubGrid, lon_subsubGrid = self.getNewLocations(grid_node.subGrid_loc[i])
            for j in range(len(self.angle_neighbour)):
                if self.polygon_path.contains_point((lat_subsubGrid[j], lon_subsubGrid[j])):
                    testRevisit = self.revisit([lat_subsubGrid[j], lon_subsubGrid[j]])
                    if not testRevisit[0]:
                        subsubGrid.append([lat_subsubGrid[j], lon_subsubGrid[j]])
                        self.grid.append([lat_subsubGrid[j], lon_subsubGrid[j]])
                        self.counter_grid = self.counter_grid + 1
                        length_new = length_new + 1
            if len(subsubGrid) > 0:
                subGrid = gridNode(len(subsubGrid), subsubGrid, grid_node.subGrid_loc[i])
                self.traverseChildrenNodes(subGrid)
            else:
                return gridNode(0, [], grid_node.subGrid_loc[i])
        return gridNode(0, [], grid_node.grid_loc)

    def revisit(self, loc):
        '''
        function determines whether it revisits the points it already has
        '''
        temp = np.array(self.grid)
        if len(self.grid) > 0:
            dist_min = np.min(np.sqrt((temp[:, 0] - loc[0]) ** 2 + (temp[:, 1] - loc[1]) ** 2))
            ind = np.argmin(np.sqrt((temp[:, 0] - loc[0]) ** 2 + (temp[:, 1] - loc[1]) ** 2))
            if dist_min <= .00001:
                return [True, ind]
            else:
                return [False, []]
        else:
            return [False, []]

    def getCoordinates(self):
        coordinates = []
        for i in range(self.grid.shape[0]):
            for j in range(len(self.depth)):
                coordinates.append([self.grid[i, 0], self.grid[i, 1], self.depth[j]])
        self.coordinates = np.array(coordinates)
        # print("Coordinates are built successfully! Coordinates: ", self.coordinates.shape)


