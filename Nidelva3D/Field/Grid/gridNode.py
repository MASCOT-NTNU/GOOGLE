"""
This contains the grid node
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-23
"""


class gridNode:
    '''
    generate node for each grid
    '''
    grid_loc = None
    subGrid_len = 0
    subGrid_loc = []

    def __init__(self, subGrids_len, subGrids_loc, grid_loc):
        if subGrids_len < 0:
            raise ValueError("There are no sub nodes")
        if not isinstance(subGrids_loc, list):
            raise TypeError("Sub nodes need to be a 2D list")
        if not grid_loc:
            raise ValueError("Current location is empty, please check")
        self.subGrid_len = subGrids_len
        self.subGrid_loc = subGrids_loc
        self.grid_loc = grid_loc


