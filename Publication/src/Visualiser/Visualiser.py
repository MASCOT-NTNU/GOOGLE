"""
Plotting func to interpolate the scattered dots.
"""
from matplotlib import tri
from matplotlib.cm import get_cmap
from shapely.geometry import Polygon, Point
import matplotlib.pyplot as plt
from matplotlib import transforms
import numpy as np
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20

from Field import Field
field = Field()


def plotf_vector(xplot, yplot, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
                 cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
                 stepsize=None, threshold=None, polygon_border=None,
                 polygon_obstacle=None, xlabel=None, ylabel=None):
    """
    NED system has an opposite coordinate system for plotting.
    """
    triangulated = tri.Triangulation(yplot, xplot)
    xplot_triangulated = xplot[triangulated.triangles].mean(axis=1)
    yplot_triangulated = yplot[triangulated.triangles].mean(axis=1)

    ind_mask = []
    for i in range(len(xplot_triangulated)):
        ind_mask.append(is_masked(yplot_triangulated[i], xplot_triangulated[i]))
    triangulated.set_mask(ind_mask)
    refiner = tri.UniformTriRefiner(triangulated)
    triangulated_refined, value_refined = refiner.refine_field(values.flatten(), subdiv=3)

    xre_plot = triangulated_refined.x
    yre_plot = triangulated_refined.y

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
        contourplot = ax.tricontourf(yre_plot, xre_plot, value_refined, levels=levels, cmap=cmap, alpha=alpha)
        ax.tricontour(yre_plot, xre_plot, value_refined, levels=levels, linewidths=linewidths, colors=colors,
                      alpha=alpha)
    else:
        contourplot = ax.tricontourf(yre_plot, xre_plot, value_refined, cmap=cmap, alpha=alpha)
        ax.tricontour(yre_plot, xre_plot, value_refined, vmin=vmin, vmax=vmax, alpha=alpha)

    if colorbar:
        cbar = plt.colorbar(contourplot, ax=ax, ticks=ticks)
        cbar.ax.set_title(cbar_title)
    # plt.xlim([np.amin(lon), np.amax(lon)])
    # plt.ylim([np.amin(lat), np.amax(lat)])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if np.any(polygon_border):
        plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'k-.', lw=2)

    return ax


def is_masked(lon, lat):
    """
    :param lon:
    :param lat:
    :return:
    """
    loc = np.array([lon, lat])
    masked = False
    if not field.border_contains(loc):
        masked = True
    return masked


