"""
EDA in Experiment mainly handles the data visualisation for the in-situ measurement.
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
from matplotlib import tri, patches
from shapely.geometry import Polygon, LineString, Point
import plotly
import plotly.graph_objs as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import os
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20


class EDA:

    def __init__(self):
        """
        EDA preparation.
        """
        self.config = Config()
        self.auv = AUV()
        self.grf = GRF(sigma=1.5, nugget=.4, approximate_eibv=False, fast_eibv=True)

        # s0, get grid
        self.grid = self.grf.grid

        # s1, get field
        self.field = self.grf.field

        # s2, get polygons for plotting
        self.polygon_border = self.config.get_polygon_border()
        self.polygon_obstacle = self.config.get_polygon_obstacle()

        # s3, set up figpath
        self.figpath = os.getcwd() + "/../../../../OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Docs/fig/Paper/EDA/analytical/"
        checkfolder(self.figpath)
        # print(os.listdir(self.figpath))

    def refine_grid(self, resolution: float = 32.) -> None:
        """ Refind the grid with given resolution. """
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
        """ Save prior on a refined grid. """
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
        """
        sigma = 1.
        nugget = .4
        # sigma = .3
        # nugget = .01
        # c1: eibv dominant
        weight_eibv = 1.
        weight_ivr = 0.

        # # c2, ivr dominant
        # weight_eibv = 0.
        # weight_ivr = 1.
        #
        # # c3, equal weight
        # weight_eibv = .5
        # weight_ivr = .5

        self.rrtstar = RRTStarCV(weight_eibv=weight_eibv, weight_ivr=weight_ivr, sigma=sigma, nugget=nugget,
                                 budget_mode=True, approximate_eibv=True)
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
                                   foldername=self.figpath + "RRTCV/", title="RRTStar and Cost valley illustration")

    def get_3d_cost_valley(self) -> None:
        """
        To plot cost valley in 3d, it requires a different discretization. Thus a different way of producing grid needs
        to be used.
        """
        # s0, create a small demonstration GRF field.
        # grf = GRF(sigma=.3, nugget=.1)
        grf = GRF(sigma=1., nugget=.4, approximate_eibv=False)
        vp = ValleyPlotter(self.grid)

        # s1, get the ei field (eibv, ivr)
        eibv, ivr = grf.get_ei_field()
        vp.plot_3d_valley(eibv, filename=self.figpath + "eibv.html", title="Cost valley illustration, EIBV component",
                          vmin=.0)
        vp.plot_3d_valley(ivr, filename=self.figpath + "ivr.html", title="Cost valley illustration, IVR component")

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
        It uses the dataset synchronized from AUV to assimilate the kernel for the GRF model.
        - It generates plots for the mission review.
        - It saves data to csv files later can be used for GIS visualisation.
        - It updates the cost valley based on the posterior field.
        """
        sigma = 1.5
        nugget = .4
        rrtstar = RRTStarCV(weight_eibv=.5, weight_ivr=.5, sigma=sigma, nugget=nugget, budget_mode=False,
                            approximate_eibv=True)
        cv = rrtstar.get_CostValley()
        grf = cv.get_grf_model()

        step_auv = 170  # samples between two waypoints.
        df = self.auv.get_dataset()
        n_samples = len(df)
        threshold = grf.get_threshold()

        pb_lat, pb_lon = WGS.xy2latlon(self.polygon_border[:, 0], self.polygon_border[:, 1])
        po_lat, po_lon = WGS.xy2latlon(self.polygon_obstacle[:, 0], self.polygon_obstacle[:, 1])
        pb = np.stack((pb_lat, pb_lon), axis=1)
        po = np.stack((po_lat, po_lon), axis=1)
        pbs = Polygon(pb)
        pos = Polygon(po)

        def is_masked(lat, lon):
            p = Point(lat, lon)
            masked = False
            if pos.contains(p) or not pbs.contains(p):
                masked = True
            return masked

        lat, lon = WGS.xy2latlon(self.grid[:, 0], self.grid[:, 1])
        triang = tri.Triangulation(lon, lat)
        lon_triangulated = lon[triang.triangles].mean(axis=1)
        lat_triangulated = lat[triang.triangles].mean(axis=1)

        ind_mask = []
        for i in range(len(lon_triangulated)):
            ind_mask.append(is_masked(lat_triangulated[i], lon_triangulated[i]))
        triang.set_mask(ind_mask)

        def plot_each_component(value, traj: np.ndarray, ind_assimilated: np.ndarray, ind_gathered: np.ndarray,
                                title: str = "Salinity", cmap=get_cmap("BrBG", 10), cbar_title="Salinity",
                                vmin=10, vmax=33., stepsize=1.5, threshold=None, alpha=1) -> tuple:
            """
            Plot each component of the posterior field.
            """
            ax = plt.gca()

            value = value.flatten()
            refiner = tri.UniformTriRefiner(triang)
            tri_refi, value_refined = refiner.refine_field(value, subdiv=3)

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
                    linewidths[ind] = 2
                    colors[ind[0]] = 'red'
                ax.tricontourf(tri_refi, value_refined, levels=levels, cmap=cmap, alpha=alpha, vmin=vmin, vmax=vmax)
                ax.tricontour(tri_refi, value_refined, levels=levels, colors=colors, linewidths=linewidths, alpha=alpha)

            if len(ind_gathered) > 1:
                plt.plot(traj[:, 1], traj[:, 0], 'k.-')
                # plt.plot(self.grid[ind_gathered, 1], self.grid[ind_gathered, 0], 'k.')
                # plt.plot(self.grid[ind_assimilated, 1], self.grid[ind_assimilated, 0], 'b^')
                plt.plot(lon[ind_gathered], lat[ind_gathered], 'b.')
                plt.plot(lon[ind_assimilated], lat[ind_assimilated], 'g^')

            plt.colorbar(label=cbar_title)

            plt.plot(po[:, 1], po[:, 0], 'r-.')
            plt.plot(pb[:, 1], pb[:, 0], 'r-.')

            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.set_xlim([np.min(pb[:, 1]), np.max(pb[:, 1])])
            ax.set_ylim([np.min(pb[:, 0]), np.max(pb[:, 0])])
            ax.set_title(title)
            ax.set_aspect('equal')

            return ax

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
            fig = plt.figure(figsize=(20, 15))
            gs = GridSpec(nrows=1, ncols=2)
            ax = fig.add_subplot(gs[0])
            plot_each_component(mu, traj=traj, ind_assimilated=ind_assimilated,
                                               ind_gathered=ind_gathered, title="Updated salinity field",
                                               threshold=threshold)

            ax = fig.add_subplot(gs[1])
            cost_field = cv.get_cost_field()
            plot_each_component(cost_field, traj=traj, ind_assimilated=ind_assimilated,
                                               ind_gathered=ind_gathered, title="Updated cost valley",
                                               cbar_title="Cost", cmap=get_cmap("RdBu", 10), vmin=0, vmax=2.,
                                               stepsize=.1)

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
            lat_t, lon_t = WGS.xy2latlon(df[:ind_end, 1], df[:ind_end, 2])
            traj = np.stack((lat_t, lon_t), axis=1)
            # traj = df[:ind_end, 1:-1]
            ind_gathered = np.append(ind_gathered, ind_assimilated.reshape(-1, 1), axis=0)

            loc_now = df[ind_end, 1:-1]
            cv.update_cost_valley(loc_now=loc_now)

            counter += 1


if __name__ == "__main__":
    e = EDA()
    # e.get_3d_cost_valley()
    # e.get_trees_on_cost_valley()
    e.run_mission_recap()