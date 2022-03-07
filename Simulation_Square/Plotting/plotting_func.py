"""
This script contains all the necessary plotting functions
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""


from usr_func import *
from GOOGLE.Simulation_Square.Config.Config import *


def plotf_vector(grid, values, title, alpha=None, cmap="Paired"):
    x = grid[:, 0]
    y = grid[:, 1]
    nx = 100
    ny = 100

    xmin, ymin = map(np.amin, [x, y])
    xmax, ymax = map(np.amax, [x, y])

    xv = np.linspace(xmin, xmax, nx)
    yv = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(xv, yv)

    grid_values = griddata(grid, values, (grid_x, grid_y))
    # plt.figure()
    plt.scatter(grid_x, grid_y, c=grid_values, cmap=cmap, alpha=alpha)
    plt.xlim(XLIM)
    plt.ylim(YLIM)
    plt.colorbar()
    plt.title(title)
    # plt.show()


def plotf_budget_radar(centre, radius):
    plt.plot(centre[0], centre[1], 'yp')
    radar = plt.Circle((centre[0], centre[1]), radius, color='b', fill=False)
    plt.gca().add_patch(radar)
    if radius >= 1:
        plt.xlim([radius * XLIM[0], radius * XLIM[1]])
        plt.ylim([radius * YLIM[0], radius * YLIM[1]])


def plotf_trajectory(trajectory):
    path = []
    for location in trajectory:
        path.append([location.x, location.y])
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'k.-')
    plt.plot(path[:, 0], path[:, 1], 'k-')


def plotf_matrix(values, title):
    # grid_values = griddata(grid, values, (grid_x, grid_y))
    plt.figure()
    plt.imshow(values, cmap="Paired", extent=(XLIM[0], XLIM[1], YLIM[0], YLIM[1]), origin="lower")
    plt.colorbar()
    plt.title(title)
    plt.show()

