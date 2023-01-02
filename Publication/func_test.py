
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


#
# def plotf_vector(xplot, yplot, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
#                  cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
#                  stepsize=None, threshold=None, polygon_border=None,
#                  polygon_obstacle=None, xlabel=None, ylabel=None):
#     """
#     NED system has an opposite coordinate system for plotting.
#     """
#     triangulated = tri.Triangulation(yplot, xplot)
#     xplot_triangulated = xplot[triangulated.triangles].mean(axis=1)
#     yplot_triangulated = yplot[triangulated.triangles].mean(axis=1)
#
#     ind_mask = []
#     for i in range(len(xplot_triangulated)):
#         ind_mask.append(is_masked(yplot_triangulated[i], xplot_triangulated[i]))
#     triangulated.set_mask(ind_mask)
#     refiner = tri.UniformTriRefiner(triangulated)
#     triangulated_refined, value_refined = refiner.refine_field(values.flatten(), subdiv=3)
#
#     xre_plot = triangulated_refined.x
#     yre_plot = triangulated_refined.y
#
#     ax = plt.gca()
#     # ax.triplot(triangulated, lw=0.5, color='white')
#     if np.any([vmin, vmax]):
#         levels = np.arange(vmin, vmax, stepsize)
#     else:
#         levels = None
#     if np.any(levels):
#         linewidths = np.ones_like(levels) * .3
#         colors = len(levels) * ['black']
#         if threshold:
#             dist = np.abs(threshold - levels)
#             ind = np.where(dist == np.amin(dist))[0]
#             linewidths[ind] = 3
#             colors[ind[0]] = 'red'
#         contourplot = ax.tricontourf(yre_plot, xre_plot, value_refined, levels=levels, cmap=cmap, alpha=alpha)
#         ax.tricontour(yre_plot, xre_plot, value_refined, levels=levels, linewidths=linewidths, colors=colors,
#                       alpha=alpha)
#     else:
#         contourplot = ax.tricontourf(yre_plot, xre_plot, value_refined, cmap=cmap, alpha=alpha)
#         ax.tricontour(yre_plot, xre_plot, value_refined, vmin=vmin, vmax=vmax, alpha=alpha)
#
#     if colorbar:
#         cbar = plt.colorbar(contourplot, ax=ax, ticks=ticks)
#         cbar.ax.set_title(cbar_title)
#     # plt.xlim([np.amin(lon), np.amax(lon)])
#     # plt.ylim([np.amin(lat), np.amax(lat)])
#     plt.title(title)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     if np.any(polygon_border):
#         plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'k-.', lw=2)
#
#     return ax