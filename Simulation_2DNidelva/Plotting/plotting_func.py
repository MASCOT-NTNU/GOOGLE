"""
This script contains all the necessary plotting functions
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""
import numpy as np

from usr_func import *
from GOOGLE.Simulation_2DNidelva.Config.Config import *


def plotf_vector(grid, values, title=None, alpha=None, cmap="Paired", cbar_title='test', colorbar=True,
                 vmin=None, vmax=None, ticks=None, kernel=None, stepsize=None, threshold=None, self=None):
    lat = grid[:, 0]
    lon = grid[:, 1]

    triangulated = tri.Triangulation(lon, lat)
    lon_triangulated = lon[triangulated.triangles].mean(axis=1)
    lat_triangulated = lat[triangulated.triangles].mean(axis=1)

    ind_mask = []
    for i in range(len(lon_triangulated)):
        ind_mask.append(is_masked(lat_triangulated[i], lon_triangulated[i], kernel))
    triangulated.set_mask(ind_mask)
    refiner = tri.UniformTriRefiner(triangulated)
    triangulated_refined, value_refined = refiner.refine_field(values.flatten(), subdiv=3)

    ax = plt.gca()
    # ax.triplot(triangulated, lw=0.5, color='white')
    if vmin and vmax:
        levels = np.arange(vmin, vmax, stepsize)
    else:
        levels = None

    if np.any(levels):
        if threshold:
            linewidths = np.ones_like(levels) * .3
            dist = np.abs(threshold - levels)
            ind = np.where(dist == np.amin(dist))[0]
            linewidths[ind] = 3
        else:
            linewidths = np.ones_like(levels) * .3
        contourplot = ax.tricontourf(triangulated_refined, value_refined, levels=levels, cmap=cmap)
        ax.tricontour(triangulated_refined, value_refined, levels=levels, linewidths=linewidths)
    else:
        contourplot = ax.tricontourf(triangulated_refined, value_refined, cmap=cmap)
        ax.tricontour(triangulated_refined, value_refined, vmin=vmin, vmax=vmax)

    if colorbar:
        cbar = plt.colorbar(contourplot, ax=ax, ticks=ticks)
        cbar.ax.set_title(cbar_title)
    # plt.grid()
    plt.title(title)

    plt.plot(self.kernel.polygon_border[:, 1], self.kernel.polygon_border[:, 0], 'k-', linewidth=1)
    plt.plot(self.kernel.polygon_obstacle[:, 1], self.kernel.polygon_obstacle[:, 0], 'k-', linewidth=1)

    # plt.show()

def plotf_vector_scatter(grid, values, title=None, alpha=None, cmap="Paired", cbar_title='test', colorbar=True,
                 vmin=None, vmax=None, ticks=None):
    # x = grid[:, 0]
    # y = grid[:, 1]
    # nx = 100
    # ny = 100
    #
    # xmin, ymin = map(np.amin, [x, y])
    # xmax, ymax = map(np.amax, [x, y])
    #
    # xv = np.linspace(xmin, xmax, nx)
    # yv = np.linspace(ymin, ymax, ny)
    # grid_x, grid_y = np.meshgrid(xv, yv)
    #
    # grid_values = griddata(grid, values, (grid_x, grid_y))
    # plt.figure()
    plt.scatter(grid[:, 1], grid[:, 0], c=values, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    # plt.scatter(grid_x, grid_y, c=grid_values, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    # plt.xlim(XLIM)
    # plt.ylim(YLIM)
    if colorbar:
        cbar = plt.colorbar(ticks=ticks)
        cbar.ax.set_title(cbar_title)
    plt.title(title)
    # plt.show()


def is_masked(lat, lon, kernel):
    point = Point(lat, lon)
    masked = False
    if kernel.polygon_obstacle_path.contains(point) or not kernel.polygon_border_path.contains(point):
        masked = True
    return masked


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
        path.append([location.lon, location.lat])
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



