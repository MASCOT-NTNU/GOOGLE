"""
This function interpolates 3d values.
"""
import numpy as np
from usr_func.interpolate_2d import interpolate_2d
from usr_func.refill_nan_values import refill_nan_values


def interpolate_3d(x: np.ndarray, y: np.ndarray, z: np.ndarray, value: np.ndarray) -> tuple:
    '''
    Interpolates values for 3d grid by interpolate on 2d layers and combine them together
    '''
    z_layer = np.unique(z)
    grid = []
    values = []
    nx = 50
    ny = 50
    nz = len(z_layer)
    for i in range(nz):
        ind_layer = np.where(z == z_layer[i])[0]
        grid_x, grid_y, grid_value = interpolate_2d(x[ind_layer], y[ind_layer], nx=nx, ny=ny,
                                                    value=value[ind_layer], interpolation_method="cubic")
        grid_value = refill_nan_values(grid_value)
        for j in range(grid_x.shape[0]):
            for k in range(grid_x.shape[1]):
                grid.append([grid_x[j, k], grid_y[j, k], z_layer[i]])
                values.append(grid_value[j, k])

    grid = np.array(grid)
    values = np.array(values)
    return grid, values