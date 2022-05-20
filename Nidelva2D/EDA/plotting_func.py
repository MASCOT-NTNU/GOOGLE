"""
This script contains all the necessary plotting functions
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""
import numpy as np

from usr_func import *
from GOOGLE.Nidelva2D.Config.Config import *


def plotf_vector(lon, lat, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10), cbar_title='test', colorbar=True,
                 vmin=None, vmax=None, ticks=None, stepsize=None, threshold=None, polygon_border=None, polygon_obstacle=None,
                 xlabel=None, ylabel=None):

    triangulated = tri.Triangulation(lon, lat)
    lon_triangulated = lon[triangulated.triangles].mean(axis=1)
    lat_triangulated = lat[triangulated.triangles].mean(axis=1)

    ind_mask = []
    for i in range(len(lon_triangulated)):
        ind_mask.append(is_masked(lat_triangulated[i], lon_triangulated[i], Polygon(polygon_border),
                                  Polygon(polygon_obstacle)))
    triangulated.set_mask(ind_mask)
    refiner = tri.UniformTriRefiner(triangulated)
    triangulated_refined, value_refined = refiner.refine_field(values.flatten(), subdiv=3)

    ax = plt.gca()
    # ax.triplot(triangulated, lw=0.5, color='white')
    if np.any([vmin, vmax]):
        levels = np.arange(vmin, vmax, stepsize)
    else:
        levels = None

    # print("levels: ", levels)

    if np.any(levels):
        linewidths = np.ones_like(levels) * .3
        colors = len(levels) * ['black']
        if threshold:
            dist = np.abs(threshold - levels)
            ind = np.where(dist == np.amin(dist))[0]
            linewidths[ind] = 3
            colors[ind[0]] = 'red'
        contourplot = ax.tricontourf(triangulated_refined, value_refined, levels=levels, cmap=cmap, alpha=alpha)
        ax.tricontour(triangulated_refined, value_refined, levels=levels, linewidths=linewidths, colors=colors,
                      alpha=alpha)
    else:
        contourplot = ax.tricontourf(triangulated_refined, value_refined, cmap=cmap, alpha=alpha)
        ax.tricontour(triangulated_refined, value_refined, vmin=vmin, vmax=vmax, alpha=alpha)

    if colorbar:
        cbar = plt.colorbar(contourplot, ax=ax, ticks=ticks)
        cbar.ax.set_title(cbar_title)
    # plt.plot(knowledge.polygon_border_xy[:, 1], knowledge.polygon_border_xy[:, 0], 'k-', linewidth=1)
    # plt.plot(knowledge.polygon_obstacle_xy[:, 1], knowledge.polygon_obstacle_xy[:, 0], 'k-', linewidth=1)
    # plt.plot(knowledge.starting_location.y, knowledge.starting_location.x, 'kv', ms=10)
    # plt.plot(knowledge.goal_location.y, knowledge.goal_location.x, 'rv', ms=10)
    plt.xlim([np.amin(lon), np.amax(lon)])
    plt.ylim([np.amin(lat), np.amax(lat)])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'k-.', lw=2)
    plt.plot(polygon_obstacle[:, 1], polygon_obstacle[:, 0], 'k-.', lw=2)

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


def is_masked(lat, lon, polygon_border=None, polygon_obstacle=None):
    point = Point(lat, lon)
    masked = False
    if polygon_obstacle.contains(point) or not polygon_border.contains(point):
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
        path.append([location.y, location.x])
    path = np.array(path)
    plt.plot(path[:, 0], path[:, 1], 'y.-')
    plt.plot(path[:, 0], path[:, 1], 'y-')


def plotf_matrix(values, title):
    # grid_values = griddata(grid, values, (grid_x, grid_y))
    plt.figure()
    plt.imshow(values, cmap="Paired", extent=(XLIM[0], XLIM[1], YLIM[0], YLIM[1]), origin="lower")
    plt.colorbar()
    plt.title(title)
    plt.show()




