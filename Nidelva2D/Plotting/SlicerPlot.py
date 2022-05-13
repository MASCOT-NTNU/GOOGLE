"""
This script plots the 3D data onto 2D slices
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-06
"""

from usr_func import *
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.gridspec import GridSpec


def scatter_to_high_resolution(x, y, value, interpolation_method="linear"):
    xmin, ymin = map(np.amin, [x, y])
    xmax, ymax = map(np.amax, [x, y])
    points = np.hstack((vectorise(x), vectorise(y)))
    grid_x, grid_y = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    grid_value = griddata(points, value, (grid_x, grid_y), method=interpolation_method)
    return grid_x, grid_y, grid_value


def organise_plot(number_of_plots):
    if not isinstance(number_of_plots, int):
        raise TypeError("Number of plots must be integer, " + number_of_plots + " is not a valid number, please check")
    if number_of_plots <= 3:
        number_of_rows = 1
        number_of_columns = number_of_plots
    elif number_of_plots <= 6:
        number_of_rows = 2
        number_of_columns = np.ceil(number_of_plots / 2)
    elif number_of_plots <= 9:
        number_of_rows = 3
        number_of_columns = np.ceil(number_of_plots / 3)
    elif number_of_plots <= 16:
        number_of_rows = 4
        number_of_columns = np.ceil(number_of_plots / 4)

    return int(number_of_rows), int(number_of_columns)


class SlicerPlot:

    def __init__(self, coordinates, value, vmin=28, vmax=30):
        if coordinates is None:
            raise ValueError("")
        if value is None:
            raise ValueError("")
        self.coordinates = coordinates
        self.value = value
        self.vmin = vmin
        self.vmax = vmax
        self.plot()

    def plot(self):
        lat = self.coordinates[:, 0]
        lon = self.coordinates[:, 1]
        depth = self.coordinates[:, 2]
        depth_layer = np.unique(depth)
        number_of_plots = len(depth_layer)

        number_of_rows, number_of_columns = organise_plot(number_of_plots)
        fig = plt.figure(figsize=(12 * number_of_columns, 8 * number_of_rows))
        gs = GridSpec(number_of_rows, number_of_columns, figure=fig)

        for i in range(len(depth_layer)):
            ind_depth = np.where(depth == depth_layer[i])[0]

            ax = fig.add_subplot(gs[i])
            im = ax.scatter(lon[ind_depth], lat[ind_depth], c=self.value[ind_depth], cmap="Paired", vmin=self.vmin, vmax=self.vmax)
            grid_x, grid_y, grid_value = scatter_to_high_resolution(lon[ind_depth], lat[ind_depth], self.value[ind_depth])
            im = ax.scatter(grid_x, grid_y, c=grid_value, cmap="Paired", vmin=self.vmin, vmax=self.vmax)
            ax.gridGenerator()
            # im = ax.scatter(lon[ind_depth], lat[ind_depth], c = vectorise(mu_prior["salinity"])[ind_depth], cmap = "Paired", vmin = 28, vmax = 30)
            # grid_x, grid_y, grid_value = scatter_to_high_resolution(lon[ind_depth], lat[ind_depth], vectorise(mu_prior["salinity"])[ind_depth])
            # im = ax.scatter(grid_x, grid_y, c=grid_value, cmap="Paired", vmin=28, vmax=30)
        plt.colorbar(im)
        plt.show()
        # ax = fig.add_subplot(gs[0])
        # ax.plot(10.4, 63.45, 'kx')
        # plt.show()






