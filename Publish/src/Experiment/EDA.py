"""
EDA module handles the explorative data analysis.
"""
from Experiment.AUV import AUV
from Field import Field
from GRF.GRF import GRF
from WGS import WGS
from Config import Config
from Visualiser.ValleyPlotter import ValleyPlotter
from Planner.RRTSCV.RRTStarCV import RRTStarCV
from Visualiser.TreePlotter import TreePlotter
from CostValley.Budget import Budget
from usr_func.checkfolder import checkfolder
from usr_func.interpolate_2d import interpolate_2d
from scipy.stats import norm
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.pyplot import get_cmap
from matplotlib.gridspec import GridSpec
from matplotlib import tri
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20


class EDA:
    def __init__(self) -> None:
        """
        The initialisation of EDA tries to replicate the experiment using data collected during the field test.

        Methodology:
            1. Set up the same grid as the field test.
            2. Set up the same AUV.
            3. Set up the same GRF with the same parameters.
            4. Set up the figure paths.

        Returns:
            None

        """
        self.config = Config()
        self.auv = AUV()
        self.grf = GRF(sigma=1.5, nugget=.4)

        # s0, get grid
        self.grid = self.grf.grid

        # s1, get field
        self.field = self.grf.field

        # s2, get polygons for plotting
        self.polygon_border = self.config.get_polygon_border()
        self.polygon_obstacle = self.config.get_polygon_obstacle()

        # s3, set up figpath
        self.figpath = os.getcwd() + "/../../../../OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Docs/fig/Paper/EDA/"
        checkfolder(self.figpath)
        # print(os.listdir(self.figpath))

    def refine_grid(self, resolution: float = 32.) -> None:
        """
        Refine the grid to a higher resolution.

        Methodology:
            1. Interpolate the field to a higher resolution using griddata.
            2. Plot the interpolated field.

        Args:
            resolution (float, optional): The resolution of the refined grid. Defaults to 32..

        Returns:
            None

        """
        field = Field(neighbour_distance=resolution)
        grid = field.get_grid()
        v = griddata(self.grid, self.grf.get_mu(), (grid[:, 0], grid[:, 1]), method="cubic")

        plt.scatter(self.grid[:, 1], self.grid[:, 0], c=self.grf.get_mu(), cmap=get_cmap("BrBG", 10),
                    vmin=10, vmax=33)
        plt.colorbar()
        plt.show()
        plt.scatter(grid[:, 1], grid[:, 0], c=v, cmap=get_cmap("BrBG", 10),
                    vmin=10, vmax=33)
        plt.colorbar()
        plt.show()
        # return ind_refined

    def save_prior(self) -> None:
        """
        Save the prior distribution of the GRF to a csv file.

        Methodology:
            1. Get the threshold of the GRF.
            2. Get the grid of the field.
            3. Get the mean and standard deviation of the GRF.
            4. Interpolate the mean, standard deviation and exceedance probability to the grid.
            5. Save the interpolated data to a csv file.

        Returns:
            None

        """
        threshold = self.grf.get_threshold()
        field = Field(neighbour_distance=32)
        grid = field.get_grid()
        filepath = "./csv/EDA/recap/"
        mu = self.grf.get_mu()
        std = np.sqrt(np.diag(self.grf.get_covariance_matrix()))

        v_mu = griddata(self.grid, mu.flatten(), (grid[:, 0], grid[:, 1]), method="cubic")
        v_std = griddata(self.grid, std, (grid[:, 0], grid[:, 1]), method="cubic")
        ep = norm.cdf(threshold, mu.flatten(), std)
        v_ep = griddata(self.grid, ep, (grid[:, 0], grid[:, 1]), method="cubic")
        lat, lon = WGS.xy2latlon(grid[:, 0], grid[:, 1])
        dd = np.stack((lat, lon, v_mu, v_std, v_ep), axis=1)  # xp, yp refers to xplot, yplot, which are not grid
        ddf = pd.DataFrame(dd, columns=['lat', 'lon', 'mu', 'std', 'ep'])
        ddf.to_csv(filepath + "prior.csv", index=False)

    def get_trees_on_cost_valley(self) -> None:
        """
        Plot rrt* trees on the 3D cost valley with different weights.

        Methodology:
            1. Set up the RRT* planner with different weights.
            2. Set up the cost valley.
            3. Set up the tree plotter.
            4. Set up the valley plotter.
            5. Set up the start and end location.
            6. Plot the trees on the cost valley.

        Returns:
            None

        """
        sigma = .3
        nugget = .01
        self.rrtstar = RRTStarCV(weight_eibv=.5, weight_ivr=.5, sigma=sigma, nugget=nugget, budget_mode=True)
        self.tp = TreePlotter()
        self.cv = self.rrtstar.get_CostValley()
        costvalley = self.cv.get_cost_field()
        vp = ValleyPlotter(self.grid)

        loc_now = np.array([3000, 1000])
        self.cv.update_cost_valley(loc_now=loc_now)
        loc_end = self.cv.get_minimum_cost_location()
        wp = self.rrtstar.get_next_waypoint(loc_now, loc_end)
        nodes = self.rrtstar.get_tree_nodes()
        traj = self.rrtstar.get_trajectory()
        vp.plot_trees_on_3d_valley(costvalley, nodes=nodes, cv=self.cv, traj=traj, wp_now=loc_now, wp_next=loc_end,
                                   filename=self.figpath + "rrt_cv.html", title="RRTStar and Cost valley illustration")

    def get_3d_cost_valley(self) -> None:
        """
        Plot the 3D cost valley.

        Methodology:
            1. Set up the cost valley.
            2. Set up the valley plotter.
            3. Plot the 3D cost valley.

        Returns:
            None

        """
        # s0, create a small demonstration GRF field.
        grf = GRF(sigma=.3, nugget=.1)
        vp = ValleyPlotter(self.grid)

        # s1, get the ei field (eibv, ivr)
        eibv, ivr = grf.get_ei_field()
        vp.plot_3d_valley(eibv, filename=self.figpath + "eibv.html", title="Cost valley illustration, EIBV component",
                          vmin=.0)
        vp.plot_3d_valley(ivr, filename=self.figpath + "eibv.html", title="Cost valley illustration, IVR component")

        # s2, check budget

        # self.b = Budget(self.grid)
        # loc = np.array([3000, 1000])
        # bf = self.b.get_budget_field(loc[0], loc[1])
        # # vp.plot_3d_valley(bf, filename=self.figpath + "budget.html", vmin=-.1, vmax=6.)
        # loc = np.array([2000, 0])
        # bf = self.b.get_budget_field(loc[0], loc[1])
        # vp.plot_3d_valley(bf, filename=self.figpath + "budget.html", vmin=-.1, vmax=6.)

        cv = .5 * eibv + .5 * ivr
        vp.plot_3d_valley(cv, filename=self.figpath + "cv.html", title="Cost valley illustration, total")
        # loc = np.array([2000, -1000])
        # bf = self.b.get_budget_field(loc[0], loc[1])
        # vp.plot_3d_valley(bf, filename=self.figpath + "budget.html")
        # loc = np.array([2500, -1500])
        # bf = self.b.get_budget_field(loc[0], loc[1])
        # vp.plot_3d_valley(bf, filename=self.figpath + "budget.html")

        # plt.scatter(grid[:, 1], grid[:, 0], c=v_eibv, cmap=get_cmap("RdBu", 10), vmin=0, vmax=1);
        # plt.colorbar();
        # plt.show()
        # plt.scatter(grid[:, 1], grid[:, 0], c=v_ivr, cmap=get_cmap("RdBu", 10), vmin=0, vmax=1);
        # plt.colorbar();
        # plt.show()
        # plt.scatter(grid[:, 1], grid[:, 0], c=v_eibv+v_ivr, cmap=get_cmap("RdBu", 10), vmin=0, vmax=2);
        # plt.colorbar();
        # plt.show()
        eibv, ivr
        pass

    def get_fields4gis(self) -> None:
        """
        Plot the fields for GIS.

        Methodology:
            1. Set up the GRF field.
            2. Set up the valley plotter.
            3. Plot the fields for GIS.

        Returns:
            None

        """
        field = Field(neighbour_distance=32)
        grid = field.get_grid()

        filepath = "./csv/EDA/recap/"
        checkfolder(filepath)
        checkfolder(filepath + "../traj/")
        checkfolder(filepath + "../indices/")
        step_auv = 170  # samples between two waypoints.
        df = self.auv.get_dataset()
        n_samples = len(df)
        threshold = self.grf.get_threshold()
        sigma = self.grf.get_sigma()

        counter = 0
        traj = None
        ind_assimilated = None
        ind_gathered = np.empty([0, 1], dtype=int)

        for i in range(0, n_samples, step_auv):

            """
            start plotting section
            """
            mu = self.grf.get_mu()
            std = np.sqrt(np.diag(self.grf.get_covariance_matrix()))
            print("Counter: ", counter)
            if i + step_auv <= n_samples:
                ind_start = i
                ind_end = i + step_auv
            else:
                ind_start = i
                ind_end = -1
            traj = df[:ind_end, 1:-1]

            ind_assimilated, val_assimilated = self.grf.assimilate_temporal_data(df[ind_start:ind_end])
            ind_gathered = np.append(ind_gathered, ind_assimilated.reshape(-1, 1), axis=0)

            """ save data to gis plotting. """
            v_mu = griddata(self.grid, mu.flatten(), (grid[:, 0], grid[:, 1]), method="cubic")
            v_std = griddata(self.grid, std, (grid[:, 0], grid[:, 1]), method="cubic")
            ep = norm.cdf(threshold, mu.flatten(), std)
            v_ep = griddata(self.grid, ep, (grid[:, 0], grid[:, 1]), method="cubic")
            lat, lon = WGS.xy2latlon(grid[:, 0], grid[:, 1])
            dd = np.stack((lat, lon, v_mu, v_std, v_ep), axis=1)  # xp, yp refers to xplot, yplot, which are not grid
            ddf = pd.DataFrame(dd, columns=['lat', 'lon', 'mu', 'std', 'ep'])
            ddf.to_csv(filepath + "P_{:03d}.csv".format(counter), index=False)

            lat, lon = WGS.xy2latlon(traj[:, 0], traj[:, 1])
            path = np.stack((lat, lon), axis=1)
            ddf = pd.DataFrame(path, columns=['lat', 'lon'])
            ddf.to_csv(filepath + "../traj/P_{:03d}.csv".format(counter), index=False)

            lat, lon = WGS.xy2latlon(self.grid[ind_gathered, 0].flatten(), self.grid[ind_gathered, 1].flatten())
            path = np.stack((lat, lon), axis=1)
            ddf = pd.DataFrame(path, columns=['lat', 'lon'])
            ddf.to_csv(filepath + "../indices/P_{:03d}.csv".format(counter), index=False)

            counter += 1

    def plot_tiff(self) -> None:
        """ Test if tiff image can be plotted. """
        print("hellow")
        file = "/Users/yaolin/Downloads/test_gis.tiff"
        import rasterio
        img = rasterio.open(file)
        # img = georaster.MultiBandRaster(file)
        b1 = img.read(1)
        b2 = img.read(2)
        b3 = img.read(3)
        height = b1.shape[0]
        width = b1.shape[1]
        cols, rows = np.meshgrid(np.arange(width), np.arange(height))
        xs, ys = rasterio.transform.xy(img.transform, rows, cols)
        lons = np.array(xs).flatten()
        lats = np.array(ys).flatten()

        # TODO: check if it is needed or remove it?
        lats
        pass

    def run_mission_recap(self) -> None:
        """
        Replicate the mission with the same parameters and save the plots for the mission review.

        Methodology:
            1. Loop through the waypoints.
            2. Assimilate the data.
            3. Plot the fields.
            4. Save the plots.

        Returns:
            None

        """
        sigma = 1.5
        nugget = .4
        rrtstar = RRTStarCV(weight_eibv=.5, weight_ivr=.5, sigma=sigma, nugget=nugget, budget_mode=True)
        cv = rrtstar.get_CostValley()
        grf = cv.get_grf_model()

        step_auv = 170  # samples between two waypoints.
        df = self.auv.get_dataset()
        n_samples = len(df)
        threshold = grf.get_threshold()

        # lat, lon = WGS.xy2latlon(self.grid[:, 0], self.grid[:, 1])
        # grid_wgs = np.stack((lat, lon), axis=1)

        def plot_each_component(value, traj: np.ndarray, ind_assimilated: np.ndarray, ind_gathered: np.ndarray,
                                title: str = "Salinity", cmap=get_cmap("BrBG", 10), cbar_title="Salinity",
                                vmin=10, vmax=33., stepsize=1.5, threshold=None) -> tuple:
            ax, xre_plot, yre_plot, value_refined = self.plotf_vector(self.grid[:, 1], self.grid[:, 0],
                                                                      value, title=title, cmap=cmap,
                                                                      cbar_title=cbar_title,
                              vmin=vmin, vmax=vmax, stepsize=stepsize, threshold=threshold,
                              polygon_border=self.polygon_border, polygon_obstacle=self.polygon_obstacle,
                              xlabel="East", ylabel="North")
            if len(ind_gathered) > 1:
                plt.plot(traj[:, 1], traj[:, 0], 'y.-')
                plt.plot(self.grid[ind_gathered, 1], self.grid[ind_gathered, 0], 'k.')
                plt.plot(self.grid[ind_assimilated, 1], self.grid[ind_assimilated, 0], 'b^')
                # plt.plot(lon[ind_gathered], lat[ind_gathered], 'k.')
                # plt.plot(lon[ind_assimilated], lat[ind_assimilated], 'b^')
            plt.gca().set_aspect("equal")
            return xre_plot, yre_plot, value_refined

        counter = 0
        traj = None
        ind_assimilated = None
        ind_gathered = np.empty([0, 1], dtype=int)

        for i in range(0, n_samples, step_auv):
            """
            start plotting section
            """
            mu = grf.get_mu()
            # std = np.sqrt(np.diag(grf.get_covariance_matrix()))
            fig = plt.figure(figsize=(30, 15))
            gs = GridSpec(nrows=1, ncols=2)
            ax = fig.add_subplot(gs[0])
            xp, yp, v_mu = plot_each_component(mu, traj=traj, ind_assimilated=ind_assimilated,
                                               ind_gathered=ind_gathered, title="Updated salinity field",
                                               threshold=threshold)

            ax = fig.add_subplot(gs[1])
            cost_field = cv.get_cost_field()
            xp, yp, v_cv = plot_each_component(cost_field, traj=traj, ind_assimilated=ind_assimilated,
                                               ind_gathered=ind_gathered, title="Updated cost valley",
                                               cbar_title="Cost", cmap=get_cmap("RdBu", 10), vmin=0, vmax=2.,
                                               stepsize=.1)
            # xp, yp, v_std = plot_each_component(std, traj=traj, ind_assimilated=ind_assimilated,
            #                                     ind_gathered=ind_gathered,
            #                     title="Conditional std", cbar_title="STD", cmap=get_cmap("RdBu", 10),
            #                     vmin=0, vmax=sigma + .1, stepsize=.1)
            # ax = fig.add_subplot(gs[2])
            # ep = norm.cdf(threshold, mu.flatten(), std)
            # xp, yp, v_ep = plot_each_component(ep, traj=traj, ind_assimilated=ind_assimilated,
            #                                    ind_gathered=ind_gathered,
            #                     title="Conditional EP", cbar_title="Probability", cmap=get_cmap("YlGnBu", 10),
            #                     vmin=0, vmax=1.01, stepsize=.1, threshold=.5)

            figpath = self.figpath + "ReCapCV/"
            checkfolder(figpath)
            plt.savefig(figpath + "/P_{:03d}.png".format(counter))
            plt.close("all")

            """
            end of plotting section. 
            """

            print("Counter: ", counter)
            if i + step_auv <= n_samples:
                ind_start = i
                ind_end = i + step_auv
            else:
                ind_start = i
                ind_end = -1

            ind_assimilated, val_assimilated = grf.assimilate_temporal_data(df[ind_start:ind_end])
            # lat_t, lon_t = WGS.xy2latlon(df[:ind_end, 1], df[:ind_end, 2])
            # traj = np.stack((lat_t, lon_t), axis=1)
            traj = df[:ind_end, 1:-1]
            ind_gathered = np.append(ind_gathered, ind_assimilated.reshape(-1, 1), axis=0)

            loc_now = df[ind_end, 1:-1]
            cv.update_cost_valley(loc_now=loc_now)

            counter += 1

    def is_masked(self, loc: np.ndarray) -> bool:
        """
        Check if the location is masked or not.

        Methodology:
            1. Check if the location is inside the obstacle.
            2. Check if the location is not inside the border.
            3. If either 1 or 2 is true, then the location is masked.

        Examples:
            >>> is_masked(np.array([0, 0]))
            False
            >>> is_masked(np.array([0, 100]))
            True

        Args:
            loc: np.array([x, y])

        Returns:
            masked: bool

        """
        masked = False
        if self.field.obstacle_contains(loc) or not self.field.border_contains(loc):
            masked = True
        return masked

    # def is_masked_wgs(self, loc: np.ndarray) -> bool:
    #     # TODO: add lat, lon converting thing only for plotting.
    #     """ loc: np.array([x, y])"""
    #     masked = False
    #     if self.field.obstacle_contains(loc) or not self.field.border_contains(loc):
    #         masked = True
    #     return masked

    def plotf_vector(self, xplot, yplot, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
                     cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
                     stepsize=None, threshold=None, polygon_border=None,
                     polygon_obstacle=None, xlabel=None, ylabel=None):
        """
        Plot the vector field.

        Methodology:
            1. Triangulate the data.
            2. Plot the triangulated data with mask values.
            3. Plot the polygon border and polygon obstacle.
            4. Plot the colorbar.
            5. Plot the threshold.

        Args:
            xplot: x coordinates of the data.
            yplot: y coordinates of the data.
            values: values of the data.
            title: title of the plot.
            alpha: alpha value of the plot.
            cmap: colormap of the plot.
            cbar_title: title of the colorbar.
            colorbar: whether to plot the colorbar or not.
            vmin: minimum value of the colorbar.
            vmax: maximum value of the colorbar.
            ticks: ticks of the colorbar.
            stepsize: stepsize of the colorbar.
            threshold: threshold of the plot.
            polygon_border: polygon border of the plot.
            polygon_obstacle: polygon obstacle of the plot.
            xlabel: xlabel of the plot.
            ylabel: ylabel of the plot.

        Examples:
            >>> xplot = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            >>> yplot = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            >>> values = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
            >>> plotf_vector(xplot, yplot, values, title="test", alpha=0.5, cmap=get_cmap("BrBG", 10),
            >>>              cbar_title='test', colorbar=True, vmin=0, vmax=10, ticks=None,
            >>>              stepsize=1, threshold=None, polygon_border=None,
            >>>              polygon_obstacle=None, xlabel=None, ylabel=None)

        Returns:
            fig: figure of the plot.
            ax: axis of the plot.
            cbar: colorbar of the plot.
            xp: x coordinates of the triangulated data.
            yp: y coordinates of the triangulated data.
            v: values of the triangulated data.

        Notes:
            * Maybe sometimes it cannot triangulate based on one axis, but changing to another axis might work.
            * So then the final output needs to be carefully treated so that it has the correct visualisation.
            * Also note, the floating point number can cause issues as well.
            * To show threshold as a red line, then vmin, vmax, stepsize, threshold needs to have values.

        Caution:
            * The triangulation is not perfect, so the final output might not be perfect as well.

        """
        triangulated = tri.Triangulation(xplot, yplot)
        x_triangulated = xplot[triangulated.triangles].mean(axis=1)
        y_triangulated = yplot[triangulated.triangles].mean(axis=1)

        ind_mask = []
        for i in range(len(x_triangulated)):
            ind_mask.append(self.is_masked(np.array([y_triangulated[i], x_triangulated[i]])))

        triangulated.set_mask(ind_mask)
        refiner = tri.UniformTriRefiner(triangulated)
        triangulated_refined, value_refined = refiner.refine_field(values.flatten(), subdiv=3)

        """ extract new x and y, refined ones. """
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
                linewidths[ind] = 10
                colors[ind[0]] = 'red'
            contourplot = ax.tricontourf(triangulated_refined, value_refined, levels=levels, cmap=cmap, alpha=alpha)
            ax.tricontour(triangulated_refined, value_refined, levels=levels, linewidths=linewidths, colors=colors,
                          alpha=alpha)
            # contourplot = ax.tricontourf(yre_plot, xre_plot, value_refined, levels=levels, cmap=cmap, alpha=alpha,
            #                              mask=ind_filtered)
            # ax.tricontour(yre_plot, xre_plot, value_refined, levels=levels, linewidths=linewidths, colors=colors,
            #               alpha=alpha)
        else:
            contourplot = ax.tricontourf(triangulated_refined, value_refined, cmap=cmap, alpha=alpha)
            ax.tricontour(triangulated_refined, value_refined, vmin=vmin, vmax=vmax, alpha=alpha, mask=ind_mask)
            # contourplot = ax.tricontourf(yre_plot, xre_plot, value_refined, cmap=cmap, alpha=alpha, mask=ind_filtered)
            # ax.tricontour(yre_plot, xre_plot, value_refined, vmin=vmin, vmax=vmax, alpha=alpha)

        """ How to get countour line vertices """
        # x = contourplot.collections[ind].get_paths()[0].vertices[:, 0]
        # y = contourplot.collections[ind].get_paths()[0].vertices[:, 1]

        if colorbar:
            cbar = plt.colorbar(contourplot, ax=ax, ticks=ticks)
            cbar.ax.set_title(cbar_title)

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if np.any(polygon_border):
            plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'k-.', lw=2)

        if np.any(polygon_obstacle):
            plt.plot(polygon_obstacle[:, 1], polygon_obstacle[:, 0], 'k-.', lw=2)
        return ax, xre_plot, yre_plot, value_refined


if __name__ == "__main__":
    e = EDA()

