"""
Interpolates values in 2d grid discretization.
"""
import numpy as np
from usr_func.vectorize import vectorize
from scipy.interpolate import griddata


def interpolate_2d(x, y, nx, ny, value, interpolation_method="linear"):
    '''
    Use griddata to interpolate to a finer results
    '''
    xmin, ymin = map(np.amin, [x, y])
    xmax, ymax = map(np.amax, [x, y])
    points = np.hstack((vectorize(x), vectorize(y)))
    xv = np.linspace(xmin, xmax, nx)
    yv = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(xv, yv)
    grid_value = griddata(points, value, (grid_x, grid_y), method=interpolation_method)
    return grid_x, grid_y, grid_value