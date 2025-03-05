"""
AgentPlot visualises the agent during the adaptive sampling for myopic 2d path planning.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-09-03
"""
from Config import Config
from Field import Field
from WGS import WGS
from usr_func.is_list_empty import is_list_empty
import os
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from matplotlib import tri
from matplotlib.pyplot import get_cmap
from shapely.geometry import Polygon, Point
from matplotlib.gridspec import GridSpec
import numpy as np
from datetime import datetime
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20

field = Field()


class AgentPlotMyopic:

    agent = None
    def __init__(self, agent, figpath) -> None:
        self.agent = agent
        self.figpath = figpath
        self.myopic = self.agent.myopic
        self.cv = self.myopic.getCostValley()
        self.grf = self.cv.get_grf_model()
        self.field = self.grf.field
        self.grid = self.field.get_grid()
        self.xgrid = self.grid[:, 0]
        self.ygrid = self.grid[:, 1]
        self.lat_grid, self.lon_grid = WGS.xy2latlon(self.grid[:, 0], self.grid[:, 1])
        self.config = Config()
        self.plg_border = self.config.get_polygon_border()
        self.plg_obs = self.config.get_polygon_obstacle()
        self.plg_border_wgs = self.config.get_wgs_polygon_border()
        self.plg_obs_wgs = self.config.get_wgs_polygon_obstacle()
        self.plg_border_wgs_shapely = Polygon(self.plg_border_wgs)
        self.plg_obs_wgs_shapely = Polygon(self.plg_obs_wgs)
        self.ylim, self.xlim = self.field.get_border_limits()
        lat, lon = WGS.xy2latlon(self.ylim, self.xlim)
        self.xlim_wgs = np.array([lon[0], lon[1]])
        self.ylim_wgs = np.array([lat[0], lat[1]])

        self.loc_start = self.config.get_loc_start()

    def plot_agent(self):
        # s0: get updated field
        mu = self.grf.get_mu()

        Sigma = self.grf.get_covariance_matrix()
        threshold = self.grf.get_threshold()
        self.cnt = self.agent.counter
        traj_past = np.array(self.agent.trajectory)

        # s1: get updated waypoints
        wp_curr = self.myopic.get_current_waypoint()
        wp_next = self.myopic.get_next_waypoint()
        wp_prev = self.myopic.get_previous_waypoint()

        # s2: get cost valley and trees.
        cost_valley = self.cv.get_cost_field()
        cost_eibv = self.cv.get_eibv_field()
        cost_ivr = self.cv.get_ivr_field()

        fig = plt.figure(figsize=(50, 20))
        gs = GridSpec(nrows=2, ncols=3)

        """ truth, mu, sigma, cost, eibv, ivr. """

        def plot_waypoints():
            ax = plt.gca()
            ax.plot(self.plg_border[:, 1], self.plg_border[:, 0], 'k-.')
            ax.plot(self.plg_obs[:, 1], self.plg_obs[:, 0], 'k-.')
            ax.plot(wp_curr[1], wp_curr[0], 'r.', markersize=20, label="Current waypoint")
            ax.plot(wp_next[1], wp_next[0], 'g.', markersize=20, label="Next waypoint")
            ax.plot(wp_prev[1], wp_prev[0], 'b.', markersize=20, label="Pioneer waypoint")
            ax.plot(self.loc_start[1], self.loc_start[0], 'rs', markersize=40, label="Deploy")
            ax.plot(traj_past[:, 1], traj_past[:, 0], 'y.-', label="Trajectory", linewidth=3, markersize=20)
            ax.set_xlabel("East [m]")
            ax.set_ylabel("North [m]")
            plt.legend()

        """ plot truth"""
        ax = fig.add_subplot(gs[0])
        mu_truth = self.agent.auv.ctd.get_salinity_at_dt_loc(dt=0, loc=self.grid)
        str_timestamp = datetime.fromtimestamp(self.agent.auv.ctd.timestamp).strftime("%Y-%m-%d %H:%M:%S")
        self.plotf_vector(self.ygrid, self.xgrid, mu_truth, title="Ground truth field at " + str_timestamp,
                          cmap=get_cmap("BrBG", 10), vmin=15, vmax=36, cbar_title="Salinity", stepsize=1.5,
                          threshold=threshold)
        plot_waypoints()

        """ plot mean """
        ax = fig.add_subplot(gs[1])
        self.plotf_vector(self.ygrid, self.xgrid, mu, title="Conditional salinity field", cmap=get_cmap("BrBG", 10),
                          vmin=15, vmax=36, cbar_title="Salinity", stepsize=1.5, threshold=threshold)
        plot_waypoints()

        """ plot var """
        ax = fig.add_subplot(gs[2])
        # im = ax.scatter(self.ygrid, self.xgrid, c=np.sqrt(np.diag(Sigma)), s=200,
        #                 cmap=get_cmap("RdBu", 10), vmin=0, vmax=2)
        # plt.title("Conditional uncertainty field")
        # plt.colorbar(im)
        self.plotf_vector(self.ygrid, self.xgrid, np.sqrt(np.diag(Sigma)), title="Conditional uncertainty field",
                          cmap=get_cmap("RdBu", 10), cbar_title="Standard deviation")
        plot_waypoints()

        """ plot cost valley and trees. """
        ax = fig.add_subplot(gs[3])
        self.plotf_vector(self.ygrid, self.xgrid, cost_valley, title="Cost Valley",
                          cmap=get_cmap("GnBu", 10), vmin=0, vmax=1.1, stepsize=.1, cbar_title="Cost")
        plot_waypoints()

        ax.set_xlim([self.xlim[0], self.xlim[1]])
        ax.set_ylim([self.ylim[0], self.ylim[1]])

        """ plot eibv field. """
        ax = fig.add_subplot(gs[4])
        self.plotf_vector(self.ygrid, self.xgrid, cost_eibv, title="EIBV cost field",
                          cmap=get_cmap("GnBu", 10), vmin=-.1, vmax=1.1, stepsize=.1, cbar_title="Cost")
        ax.set_title("EIBV field")
        plot_waypoints()

        """ plot ivr field. """
        ax = fig.add_subplot(gs[5])
        self.plotf_vector(self.ygrid, self.xgrid, cost_ivr, title="IVR cost field",
                          cmap=get_cmap("GnBu", 10), vmin=-.1, vmax=1.1, stepsize=.1, cbar_title="Cost")
        ax.set_title("IVR field")
        plot_waypoints()

        plt.savefig(self.figpath + "P_{:03d}.png".format(self.cnt))
        # plt.show()
        plt.close("all")

    def plot_agent4paper(self):
        # s0: get updated field
        mu = self.grf.get_mu()

        Sigma = self.grf.get_covariance_matrix()
        threshold = self.grf.get_threshold()
        self.cnt = self.agent.counter

        traj_past = np.array(self.agent.trajectory)
        if len(traj_past) == 0:
            traj_past_wgs = np.array([[], []]).T
        else:
            lat, lon = WGS.xy2latlon(traj_past[:, 0], traj_past[:, 1])
            traj_past_wgs = np.array([lat, lon]).T

        # s1: get updated waypoints
        wp_curr = self.myopic.get_previous_waypoint()
        lat, lon = WGS.xy2latlon(wp_curr[0], wp_curr[1])
        wp_curr_wgs = np.array([lat, lon])
        wp_next = self.myopic.get_next_waypoint()
        lat, lon = WGS.xy2latlon(wp_next[0], wp_next[1])
        wp_next_wgs = np.array([lat, lon])

        # s2: get cost valley and trees.
        cost_valley = self.cv.get_cost_field()

        fig = plt.figure(figsize=(36, 10))
        gs = GridSpec(nrows=1, ncols=3)

        """ mu, sigma, cost """

        def plot_waypoints():
            ax = plt.gca()
            ax.plot(self.plg_border_wgs[:, 1], self.plg_border_wgs[:, 0], 'k-.')
            ax.plot(self.plg_obs_wgs[:, 1], self.plg_obs_wgs[:, 0], 'k-.')
            ax.plot(traj_past_wgs[:, 1], traj_past_wgs[:, 0], 'k.-', label="Trajectory", linewidth=3, markersize=20)
            ax.plot(wp_curr_wgs[1], wp_curr_wgs[0], 'r.', markersize=20, label="Current waypoint")
            ax.plot(wp_next_wgs[1], wp_next_wgs[0], 'y.', markersize=20, label="Next waypoint")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            plt.legend(loc='lower right')

        """ plot mean"""
        ax = fig.add_subplot(gs[0])
        str_timestamp = datetime.fromtimestamp(self.agent.auv.ctd.timestamp).strftime("%H:%M")
        self.plotf_vector_wgs(self.lat_grid, self.lon_grid, mu, title="Updated salinity field at " + str_timestamp,
                          cmap=get_cmap("BrBG", 10), vmin=10, vmax=33,
                          cbar_title="Salinity", stepsize=1.5, threshold=threshold)
        plot_waypoints()

        """ plot var """
        ax = fig.add_subplot(gs[1])
        # im = ax.scatter(self.ygrid, self.xgrid, c=np.sqrt(np.diag(Sigma)), s=200,
        #                 cmap=get_cmap("RdBu", 10), vmin=0, vmax=2)
        # plt.title("Conditional uncertainty field")
        # plt.colorbar(im)
        self.plotf_vector_wgs(self.lat_grid, self.lon_grid, np.sqrt(np.diag(Sigma)),
                          title="Updated uncertainty field at "+str_timestamp,
                          cmap=get_cmap("RdBu", 10), vmin=0, vmax=.8, stepsize=.05, cbar_title="STD")
        plot_waypoints()

        """ plot cost valley. """
        ax = fig.add_subplot(gs[2])
        self.plotf_vector_wgs(self.lat_grid, self.lon_grid, cost_valley, title="Updated cost valley at " + str_timestamp,
                          cmap=get_cmap("GnBu", 10), vmin=0, vmax=1.1, stepsize=.1, cbar_title="Cost")
        plot_waypoints()

        ax.set_xlim([self.xlim_wgs[0], self.xlim_wgs[1]])
        ax.set_ylim([self.ylim_wgs[0], self.ylim_wgs[1]])

        plt.savefig(self.figpath + "MYP/MYP_{:03d}.png".format(self.cnt))
        # plt.show()
        plt.close("all")

    def plotf_vector_wgs(self, lat, lon, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
                     cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
                     stepsize=None, threshold=None, polygon_border=None,
                     polygon_obstacle=None, xlabel=None, ylabel=None):
        """ Note for triangulation:
        - Maybe sometimes it cannot triangulate based on one axis, but changing to another axis might work.
        - So then the final output needs to be carefully treated so that it has the correct visualisation.
        - Also note, the floating point number can cause issues as well.
        - Triangulation uses a different axis than lat lon after its done.
        """
        """ To show threshold as a red line, then vmin, vmax, stepsize, threshold needs to have values. """
        triangulated = tri.Triangulation(lon, lat)
        lat_triangulated = lat[triangulated.triangles].mean(axis=1)
        lon_triangulated = lon[triangulated.triangles].mean(axis=1)

        ind_mask = []
        for i in range(len(lat_triangulated)):
            ind_mask.append(self.is_masked_wgs(lat_triangulated[i], lon_triangulated[i]))
        triangulated.set_mask(ind_mask)
        refiner = tri.UniformTriRefiner(triangulated)
        triangulated_refined, value_refined = refiner.refine_field(values.flatten(), subdiv=3)

        ax = plt.gca()
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
                linewidths[ind[0]] = 4
                colors[ind[0]] = 'red'
            contourplot = ax.tricontourf(triangulated_refined, value_refined, levels=levels, cmap=cmap, alpha=alpha)
            ax.tricontour(triangulated_refined, value_refined, levels=levels, linewidths=linewidths, colors=colors,
                          alpha=alpha)
        else:
            contourplot = ax.tricontourf(triangulated_refined, value_refined, cmap=cmap, alpha=alpha)
            ax.tricontour(triangulated_refined, value_refined, vmin=vmin, vmax=vmax, alpha=alpha)

        if colorbar:
            # fig = plt.gcf()
            # cax = fig.add_axes([0.85, .1, 0.03, 0.25])  # left, bottom, width, height, in percentage for left and bottom
            # cbar = fig.colorbar(contourplot, cax=cax, ticks=ticks)

            cbar = plt.colorbar(contourplot, ax=ax, ticks=ticks, orientation='vertical')
            cbar.ax.set_title(cbar_title)
            # cbar.ax.set_ylabel(cbar_title, rotation=270, labelpad=40)
        ax.set_title(title)

        if polygon_border is not None:
            ax.plot(polygon_border[:, 1], polygon_border[:, 0], 'r-.')
        if polygon_obstacle is not None:
            plg = plt.Polygon(np.fliplr(polygon_obstacle), facecolor='w', edgecolor='r', fill=True,
                              linestyle='-.')
            plt.gca().add_patch(plg)

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        return ax

    def plotf_vector(self, xplot, yplot, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
                     cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
                     stepsize=None, threshold=None, polygon_border=None,
                     polygon_obstacle=None, xlabel=None, ylabel=None):
        """ Note for triangulation:
        - Maybe sometimes it cannot triangulate based on one axis, but changing to another axis might work.
        - So then the final output needs to be carefully treated so that it has the correct visualisation.
        - Also note, the floating point number can cause issues as well.
        """
        """ To show threshold as a red line, then vmin, vmax, stepsize, threshold needs to have values. """
        triangulated = tri.Triangulation(xplot, yplot)
        x_triangulated = xplot[triangulated.triangles].mean(axis=1)
        y_triangulated = yplot[triangulated.triangles].mean(axis=1)

        ind_mask = []
        for i in range(len(x_triangulated)):
            ind_mask.append(self.is_masked(y_triangulated[i], x_triangulated[i]))

        triangulated.set_mask(ind_mask)
        refiner = tri.UniformTriRefiner(triangulated)
        triangulated_refined, value_refined = refiner.refine_field(values.flatten(), subdiv=3)

        """ extract new x and y, refined ones. """
        # xre_plot = triangulated_refined.x
        # yre_plot = triangulated_refined.y
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
            # contourplot = ax.tricontourf(yre_plot, xre_plot, value_refined, levels=levels, cmap=cmap, alpha=alpha)
            # ax.tricontour(yre_plot, xre_plot, value_refined, levels=levels, linewidths=linewidths, colors=colors,
            #               alpha=alpha)
        else:
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
        if np.any(polygon_border):
            plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'k-.', lw=2)

        if np.any(polygon_obstacle):
            plt.plot(polygon_obstacle[:, 1], polygon_obstacle[:, 0], 'k-.', lw=2)

        return ax, value_refined

    # def plotf_vector(self, xplot, yplot, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
    #                  cbar_title='test', colorbar=True, vmin=None, vmax=None, ticks=None,
    #                  stepsize=None, threshold=None, polygon_border=None,
    #                  polygon_obstacle=None, xlabel=None, ylabel=None):
    #     """ Note for triangulation:
    #     - Maybe sometimes it cannot triangulate based on one axis, but changing to another axis might work.
    #     - So then the final output needs to be carefully treated so that it has the correct visualisation.
    #     """
    #     """ To show threshold as a red line, then vmin, vmax, stepsize, threshold needs to have values. """
    #     triangulated = tri.Triangulation(yplot, xplot)
    #     x_triangulated = xplot[triangulated.triangles].mean(axis=1)
    #     y_triangulated = yplot[triangulated.triangles].mean(axis=1)
    #
    #     ind_mask = []
    #     for i in range(len(x_triangulated)):
    #         ind_mask.append(self.is_masked(y_triangulated[i], x_triangulated[i]))
    #     triangulated.set_mask(ind_mask)
    #     refiner = tri.UniformTriRefiner(triangulated)
    #     triangulated_refined, value_refined = refiner.refine_field(values.flatten(), subdiv=3)
    #
    #     """ extract new x and y, refined ones. """
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
    #     return ax

    @staticmethod
    def is_masked(xgrid, ygrid) -> bool:
        """
        :param xgrid:
        :param ygrid:
        :return:
        """
        loc = np.array([xgrid, ygrid])
        masked = False
        if field.obstacle_contains(loc) or not field.border_contains(loc):
            masked = True
        return masked

    def is_masked_wgs(self, lat, lon) -> bool:
        p = Point(lat, lon)
        masked = False
        if not self.plg_border_wgs_shapely.contains(p) or self.plg_obs_wgs_shapely.contains(p):
            masked = True
        return masked

    # @staticmethod
    # def is_masked(x, y) -> bool:
    #     """
    #     :param x:
    #     :param y:
    #     :return:
    #     """
    #     loc = np.array([x, y])
    #     masked = False
    #     if not field.border_contains(loc):
    #         masked = True
    #     return masked




