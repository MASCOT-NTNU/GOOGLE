"""
Plotting func to interpolate the scattered dots.
"""
from matplotlib import tri
from matplotlib.cm import get_cmap
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20

from Field import Field
field = Field()


def plotf_vector(x, y, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
                 cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
                 stepsize=None, threshold=None, polygon_border=None,
                 polygon_obstacle=None, xlabel=None, ylabel=None):
    """

    :param x:
    :param y:
    :param values:
    :param title:
    :param alpha:
    :param cmap:
    :param cbar_title:
    :param colorbar:
    :param vmin:
    :param vmax:
    :param ticks:
    :param stepsize:
    :param threshold:
    :param polygon_border: shapely polygon object.
    :param polygon_obstacle: list of shapely objects containing polygons of obstacles.
    :param xlabel:
    :param ylabel:
    :return:
    """
    triangulated = tri.Triangulation(x, y)
    x_triangulated = x[triangulated.triangles].mean(axis=1)
    y_triangulated = y[triangulated.triangles].mean(axis=1)

    ind_mask = []
    for i in range(len(x_triangulated)):
        ind_mask.append(is_masked(x_triangulated[i], y_triangulated[i]))
    triangulated.set_mask(ind_mask)
    refiner = tri.UniformTriRefiner(triangulated)
    triangulated_refined, value_refined = refiner.refine_field(values.flatten(), subdiv=3)

    ax = plt.gca()
    # ax.triplot(triangulated, lw=0.5, color='white')
    if np.any([vmin, vmax]):
        levels = np.arange(vmin, vmax, stepsize)
    else:
        levels = None
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
    plt.xlim([np.amin(x), np.amax(x)])
    plt.ylim([np.amin(y), np.amax(y)])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if np.any(polygon_border):
        plt.plot(polygon_border[:, 0], polygon_border[:, 1], 'k-.', lw=2)
        for i in range(len(polygon_obstacle)):
            plt.plot(polygon_obstacle[i][:, 0], polygon_obstacle[i][:, 1], 'k-.', lw=2)
    return ax


def is_masked(x, y):
    """
    :param x:
    :param y:
    :return:
    """
    loc = np.array([x, y])
    masked = False
    if field.obstacles_contain(loc) or not field.border_contains(loc):
        masked = True
    return masked

