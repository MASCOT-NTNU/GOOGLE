"""
This script contains all the necessary plotting functions
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-07
"""


from usr_func import *
from GOOGLE.Simulation_2DSquare.Config.Config import *


def plotf_vector(grid, values, title=None, alpha=None, cmap="Paired", cbar_title='test', colorbar=True,
                 vmin=None, vmax=None, ticks=None):
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
    plt.scatter(grid_x, grid_y, c=grid_values, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
    plt.xlim(XLIM)
    plt.ylim(YLIM)
    if colorbar:
        cbar = plt.colorbar(ticks=ticks)
        cbar.ax.set_title(cbar_title)
    plt.title(title)
    # plt.show()


def plotf_vector_triangulated(grid, values, title=None, alpha=None, cmap="Paired", cbar_title='test', colorbar=True,
                 vmin=None, vmax=None, ticks=None, knowledge=None, stepsize=None, threshold=None):
    x = grid[:, 0]
    y = grid[:, 1]

    triangulated = tri.Triangulation(x, y)
    x_triangulated = x[triangulated.triangles].mean(axis=1)
    y_triangulated = y[triangulated.triangles].mean(axis=1)

    ind_mask = []
    for i in range(len(x_triangulated)):
        ind_mask.append(is_masked(x_triangulated[i], y_triangulated[i], knowledge))
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
        # print(colors)
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
    # plt.grid()
    plt.title(title)

    plt.plot(knowledge.polygon_border[:, 0], knowledge.polygon_border[:, 1], 'k-', linewidth=1)
    for i in range(len(knowledge.polygon_obstacles)):
        plt.plot(knowledge.polygon_obstacles[i][:, 0], knowledge.polygon_obstacles[i][:, 1], 'k-', linewidth=1)
    plt.plot(knowledge.starting_location.x, knowledge.starting_location.y, 'kv', ms=10)
    plt.plot(knowledge.goal_location.x, knowledge.goal_location.y, 'rv', ms=10)
    plt.xlim([np.amin(x), np.amax(x)])
    plt.ylim([np.amin(y), np.amax(y)])


def is_masked(x, y, knowledge):
    point = Point(x, y)
    masked = False
    if is_within_obstacles(point, knowledge) or not knowledge.polygon_border_shapely.contains(point):
        masked = True
    return masked

def is_within_obstacles(point, knowledge):
    within = False
    for i in range(len(knowledge.polygon_obstacles_shapely)):
        if knowledge.polygon_obstacles_shapely[i].contains(point):
            within = True
    return within

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
    plt.plot(path[:, 0], path[:, 1], 'y.-')
    plt.plot(path[:, 0], path[:, 1], 'y-')


def plotf_matrix(values, title):
    # grid_values = griddata(grid, values, (grid_x, grid_y))
    plt.figure()
    plt.imshow(values, cmap="Paired", extent=(XLIM[0], XLIM[1], YLIM[0], YLIM[1]), origin="lower")
    plt.colorbar()
    plt.title(title)
    plt.show()

