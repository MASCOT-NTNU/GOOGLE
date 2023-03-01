"""
Interpolate 2D data to a finer grid using griddata.
"""
import numpy as np
from usr_func.vectorize import vectorize
from scipy.interpolate import griddata


def interpolate_2d(x, y, nx, ny, value, interpolation_method="linear") -> tuple:
    """
    Args:
        x (np.ndarray): x coordinates of the data points.
        y (np.ndarray): y coordinates of the data points.
        nx (int): Number of points to be refined in the x direction.
        ny (int): Number of points to be refined in the y direction.
        value (np.ndarray): Values of the data points.
        interpolation_method (str): Interpolation method. Default is "linear".

    Returns:
        tuple: x coordinates of the refined grid, y coordinates of the refined grid, and values of the refined grid.

    Examples:
        >>> x = np.array([0, 1, 2, 3, 4, 5])
        >>> y = np.array([0, 1, 2, 3, 4, 5])
        >>> nx = 100
        >>> ny = 100
        >>> value = np.array([0, 1, 2, 3, 4, 5])
        >>> grid_x, grid_y, grid_value = interpolate_2d(x, y, nx, ny, value)
        >>> grid_x.shape
        (100, 100)
        >>> grid_y.shape
        (100, 100)
        >>> grid_value.shape
        (100, 100)

    """
    xmin, ymin = map(np.amin, [x, y])
    xmax, ymax = map(np.amax, [x, y])
    points = np.hstack((vectorize(x), vectorize(y)))
    xv = np.linspace(xmin, xmax, nx)
    yv = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(xv, yv)
    grid_value = griddata(points, value, (grid_x, grid_y), method=interpolation_method)
    return grid_x, grid_y, grid_value