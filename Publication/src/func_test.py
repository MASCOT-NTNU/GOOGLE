from matplotlib import tri
from matplotlib.cm import get_cmap

# def is_masked(value) -> bool:
#     """
#     :param xgrid:
#     :param ygrid:
#     :return:
#     """
#     masked = False
#     if value == np.inf
#         masked = True
#     return masked

def plotf_vector(xplot, yplot, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
                 cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
                 xlabel=None, ylabel=None):
    """ Note for triangulation:
    - Maybe sometimes it cannot triangulate based on one axis, but changing to another axis might work.
    - So then the final output needs to be carefully treated so that it has the correct visualisation.
    - Also note, the floating point number can cause issues as well.
    """
    """ To show threshold as a red line, then vmin, vmax, stepsize, threshold needs to have values. """
    triangulated = tri.Triangulation(xplot, yplot)
    x_triangulated = xplot[triangulated.triangles].mean(axis=1)
    y_triangulated = yplot[triangulated.triangles].mean(axis=1)

#     ind_mask = []
#     for i in range(len(x_triangulated)):
#         ind_mask.append(is_masked(values[i]))

    # triangulated.set_mask(ind_mask)
    refiner = tri.UniformTriRefiner(triangulated)
    triangulated_refined, value_refined = refiner.refine_field(values.flatten(), subdiv=3)

    """ extract new x and y, refined ones. """
    # xre_plot = triangulated_refined.x
    # yre_plot = triangulated_refined.y
    ax = plt.gca()
    # ax.triplot(triangulated, lw=0.5, color='white')
    contourplot = ax.tricontourf(triangulated_refined, value_refined, cmap=cmap, alpha=alpha)
    ax.tricontour(triangulated_refined, value_refined, vmin=vmin, vmax=vmax, alpha=alpha)
    # contourplot = ax.tricontourf(yre_plot, xre_plot, value_refined, cmap=cmap, alpha=alpha)
    # ax.tricontour(yre_plot, xre_plot, value_refined, vmin=vmin, vmax=vmax, alpha=alpha)

    if colorbar:
        cbar = plt.colorbar(contourplot, ax=ax, ticks=ticks)
        cbar.ax.set_title(cbar_title)

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return ax, value_refined
