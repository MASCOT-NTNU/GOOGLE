"""
EDA for the simulation study.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-09-05
"""
import datetime

from WGS import WGS
from Config import Config
from GRF.GRF import GRF
from usr_func.checkfolder import checkfolder
import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
from matplotlib.gridspec import GridSpec
from joblib import Parallel, delayed, dump, load
from concurrent.futures import ThreadPoolExecutor
import seaborn as sns
from matplotlib.patches import PathPatch
from matplotlib.path import Path
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib.cm import get_cmap
from sklearn.neighbors import KernelDensity
from statsmodels.nonparametric.smoothers_lowess import lowess
from time import time
from matplotlib import tri
from shapely.geometry import Polygon, Point


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 30

folderpath = "./npy/temporal/Synced/"
figpath = os.getcwd() + ("/../../../../OneDrive - NTNU/MASCOT_PhD/"
                         "Projects/GOOGLE/Docs/fig/Sim_2DNidelva/Simulator/temporal/")


class EDA:

    def __init__(self) -> None:
        self.config = Config()
        self.grf = GRF()
        self.grid = self.grf.grid
        self.threshold = self.grf.get_threshold()
        self.Ngrid = self.grid.shape[0]
        lat, lon = WGS.xy2latlon(self.grid[:, 0], self.grid[:, 1])
        self.grid_wgs = np.array([lat, lon]).T

        self.polygon_border_wgs = self.config.get_wgs_polygon_border()
        self.polygon_border_wgs_shapely = Polygon(self.polygon_border_wgs)
        self.polygon_obstacle_wgs = self.config.get_wgs_polygon_obstacle()
        self.polygon_obstacle_wgs_shapely = Polygon(self.polygon_obstacle_wgs)

        self.num_steps = self.config.get_num_steps()
        self.num_replicates = self.config.get_num_replicates()

        self.string_myopic = "SimulatorMyopic2D"
        self.string_rrt = "SimulatorRRTStar"

        self.cv = ['eibv', 'ivr', 'equal']
        self.planners = ['myopic', 'rrt']
        self.metrics = ['traj', 'ibv', 'rmse', 'vr', 'mu', 'cov', 'sigma', 'truth']

        # replicates = os.listdir(folderpath)
        # self.num_replicates = 0
        # for rep in replicates:
        #     if rep.startswith("R_"):
        #         self.num_replicates += 1
        # print("Number of replicates: ", self.num_replicates)

        # string_para = "/sigma_10/nugget_025/"
        # string_myopic = string_para + self.string_myopic + "/"
        # string_rrt = string_para + self.string_rrt + "/"

        # self.traj_myopic, self.mu_myopic, self.sigma_myopic, self.truth_myopic = self.load_sim_data(string_myopic)
        # self.traj_rrt, self.mu_rrt, self.sigma_rrt, self.truth_rrt = self.load_sim_data(string_rrt)

        self.lon_min = np.min(self.polygon_border_wgs[:, 1])
        self.lon_max = np.max(self.polygon_border_wgs[:, 1])
        self.lat_min = np.min(self.polygon_border_wgs[:, 0])
        self.lat_max = np.max(self.polygon_border_wgs[:, 0])
        self.lon_ticks = np.round(np.arange(self.lon_min, self.lon_max, 0.02), 2)
        self.lat_ticks = np.round(np.arange(self.lat_min, self.lat_max, 0.005), 2)

        """ Only DO ONCE!!! Load raw data and then reorganize it. """
        # self.load_raw_data_from_replicate_files()
        # self.organize_data_to_dict()
        """ End of Data loading!!! """

        self.load_data()
        # self.plot_metrics_total()
        # self.plot_temporal_traffic_density_map()
        # self.plot_temporal_traffic_flow_density_map4paper()
        # self.plot_metrics4paper()
        self.plot_es4paper()
        # self.plot_ground_truth()
        # self.plot_es()
        self.trajectory

    def load_data(self) -> None:
        t0 = time()
        fpath = os.getcwd() + "/../simulation_result/Synced/"
        self.trajectory = load(fpath + "trajectory.joblib")
        self.ibv = load(fpath + "ibv.joblib")
        self.rmse = load(fpath + "rmse.joblib")
        self.vr = load(fpath + "vr.joblib")
        self.mu = load(fpath + "mu.joblib")
        # self.cov = load(fpath + "cov.joblib")
        self.sigma = load(fpath + "sigma.joblib")
        self.truth = load(fpath + "truth.joblib")

        """ Load the excursion set area difference. """
        self.es_diff = load(fpath + "es_diff.joblib")

        print("Loading data takes {:.2f} seconds.".format(time() - t0))

        self.ibv_max = np.max(self.ibv['myopic']['equal'])
        self.ibv_min = np.min(self.ibv['myopic']['equal'])
        self.rmse_max = np.max(self.rmse['myopic']['equal'])
        self.rmse_min = np.min(self.rmse['myopic']['equal'])
        self.vr_max = np.max(self.vr['myopic']['equal'])
        self.vr_min = np.min(self.vr['myopic']['equal'])
        self.ibv_max += .05 * self.ibv_max
        self.ibv_min -= .05 * self.ibv_min
        self.rmse_max += .05 * self.rmse_max
        self.rmse_min -= .05 * self.rmse_min
        self.vr_max += .05 * self.vr_max
        self.vr_min -= .05 * self.vr_min

        self.xticks = np.arange(0, self.num_steps, 20)
        self.xticklabels = ['{:d}'.format(i) for i in self.xticks]

    def plot_metrics_total(self) -> None:
        """
        Plot the metric with the temporal traffic density.
        """
        for i in tqdm(range(1, self.num_steps)):
            fig = plt.figure(figsize=(50, 27))
            gs = GridSpec(nrows=3, ncols=6, figure=fig)

            """ Section I, make the other plot. """
            self.plot_es(i, 0, 0, fig, gs)

            """ Section II, make traffic density plot. """
            self.plot_temporal_traffic_density_map(i, 0, 2, fig, gs)

            """ Section III, make the metric plot. """
            self.plot_metrics(i, 0, 4, fig, gs)

            """ Section Final, save the figure. """
            plt.savefig(figpath + "P_{:03d}.png".format(i))
            plt.close("all")

    def plot_es4paper(self) -> None:
        """
        Plot the excusion set associated metrics.

        1. Area difference ratio
        2. Over set
        3. Under set

        Since it is used to plot the over set only, so only over set is selected!
        """
        def make_subplot_area(ax):
            for planner in self.planners:
                for item in self.cv:
                # for item in ['equal']:
                    es_temp = np.abs(self.es_diff[planner][item])
                    ax.errorbar(np.arange(8), y=np.mean(np.mean(np.mean(es_temp, axis=1), axis=1), axis=1),
                                yerr=np.std(np.mean(np.mean(es_temp, axis=1), axis=2)) / np.sqrt(self.num_replicates) * 1.645,
                                fmt="-o", capsize=5, label=planner.upper() + " " + item.upper())
            ax.set_xticks(np.arange(8))
            ax.set_xticklabels(['{:d}'.format(i) for i in np.arange(8)])
            ax.set_xlim([0, 7])
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Area Difference Ratio")
            ax.set_ylim([.1, .3])
            plt.legend(loc="upper right")

        def make_time_adjusted_subplot_area(ax):
            for planner in self.planners:
                for item in self.cv:
                    # esd = -np.diff(np.abs(self.es_diff[planner][item]), axis=0)
                    # for i in range(1, 6):
                    #     esd[i+1, :, :, :] += esd[i, :, :, :]
                    #     esd[i, :, :, :] = esd[i, :, :, :] / (i / 6)
                    es_temp = np.abs(self.es_diff[planner][item])
                    ax.errorbar(np.arange(8), y=np.mean(np.mean(np.mean(es_temp, axis=1), axis=1), axis=1),
                                yerr=np.std(np.mean(np.mean(es_temp, axis=1), axis=2)) / np.sqrt(self.num_replicates) * 1.645,
                                fmt="-o", capsize=5, label=planner.upper() + " " + item.upper())
            ax.set_xticks(np.arange(8))
            ax.set_xticklabels(['{:d}'.format(i) for i in np.arange(8)])
            ax.set_xlim([0, 7])
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Area Difference Ratio Over Time")
            plt.legend(loc="upper right")

        def make_subplot_overset_underset(is_over):
            ax = plt.gca()
            for planner in self.planners:
                for item in self.cv:
                    es_temp = self.es_diff[planner][item]
                    if is_over:
                        oset = np.sum(es_temp == 1, axis=-1)
                    else:
                        oset = np.sum(es_temp == -1, axis=-1)
                    ax.errorbar(np.arange(8), y=np.mean(np.mean(oset, axis=2), axis=1),
                                yerr=np.std(np.mean(oset, axis=2), axis=1) / np.sqrt(self.num_replicates) * 1.645,
                                fmt="-o", capsize=5, label=planner.upper() + " " + item.upper())
            ax.set_xticks(np.arange(8))
            ax.set_xticklabels(['{:d}'.format(i) for i in np.arange(8)])
            ax.set_xlim([0, 7])
            ax.set_xlabel("Time Step")
            if is_over:
                ax.set_ylabel("Over set")
                ax.set_ylim([35, 100])
            else:
                ax.set_ylabel("Under set")
                ax.set_ylim([80, 160])
            plt.legend(loc="upper right")

        fig = plt.figure(figsize=(24, 20))
        gs = GridSpec(nrows=1, ncols=2, figure=fig)
        ax = fig.add_subplot(gs[0])
        make_subplot_area(ax)

        ax = fig.add_subplot(gs[1])
        make_time_adjusted_subplot_area(ax)
        # make_time_adjusted_subplot_area(ax)

        plt.show()

        self.es_diff

        # ax = fig.add_subplot(gs[0, 0])
        # make_subplot_overset_underset(True)
        #
        # ax = fig.add_subplot(gs[0, 1])
        # make_subplot_overset_underset(False)

        plt.show()
        ax

    def plot_es(self, step, row_ind, col_ind, fig, gs) -> None:
        """
        Plot the excusion set associated metrics.

        1. Area difference ratio
        2. Over set
        3. Under set
        """
        # m1, calculate the area metric
        def make_subplot_area(planner, step, ax):
            for item in self.cv:
                es_temp = np.abs(self.es_diff[planner][item][:step, :, :, :])
                ax.errorbar(np.arange(step), y=np.mean(np.mean(np.mean(es_temp, axis=1), axis=1), axis=1),
                            yerr=np.std(np.mean(np.mean(es_temp, axis=1), axis=2)) / np.sqrt(self.num_replicates) * 1.645,
                            fmt="-o", capsize=5, label=planner.upper() + " " + item.upper())
            ax.set_xticks(np.arange(8))
            ax.set_xticklabels(['{:d}'.format(i) for i in np.arange(8)])
            ax.set_xlim([0, 7])
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Area Difference Ratio")
            ax.set_ylim([.1, .3])
            plt.legend(loc="upper right")

        # m2, calculate the over set and under set
        def make_subplot_overset_underset(planner, step, is_over, ax):
            for item in self.cv:
                es_temp = self.es_diff[planner][item][:step, :, :, :]
                if is_over:
                    oset = np.sum(es_temp == 1, axis=-1)
                else:
                    oset = np.sum(es_temp == -1, axis=-1)
                ax.errorbar(np.arange(step), y=np.mean(np.mean(oset, axis=1), axis=1),
                            yerr=np.std(np.mean(oset, axis=2), axis=1) / np.sqrt(self.num_replicates) * 1.645,
                            fmt="-o", capsize=5, label=planner.upper() + " " + item.upper())
            ax.set_xticks(np.arange(8))
            ax.set_xticklabels(['{:d}'.format(i) for i in np.arange(8)])
            ax.set_xlim([0, 7])
            ax.set_xlabel("Time Step")
            if is_over:
                ax.set_ylabel("Over set")
                ax.set_ylim([40, 100])
            else:
                ax.set_ylabel("Under set")
                ax.set_ylim([80, 160])
            plt.legend(loc="upper right")

        step_ind = step // 15
        ax = fig.add_subplot(gs[row_ind, col_ind])
        make_subplot_area('myopic', step_ind, ax)
        ax = fig.add_subplot(gs[row_ind, col_ind+1])
        make_subplot_area('rrt', step_ind, ax)
        ax = fig.add_subplot(gs[row_ind+1, col_ind])
        make_subplot_overset_underset('myopic', step_ind, True, ax)
        ax = fig.add_subplot(gs[row_ind+2, col_ind])
        make_subplot_overset_underset('myopic', step_ind, False, ax)
        ax = fig.add_subplot(gs[row_ind+1, col_ind+1])
        make_subplot_overset_underset('rrt', step_ind, True, ax)
        ax = fig.add_subplot(gs[row_ind+2, col_ind+1])
        make_subplot_overset_underset('rrt', step_ind, False, ax)

        # fpath = figpath + "/../ES/"
        # checkfolder(fpath)
        # def make_subplot_es(planner, item, ind_row, ind_col, step, replicate_id):
        #     # s0, plot mean
        #     es_mean = np.mean(self.es_diff[planner][item][step, replicate_id, :, :], axis=0)
        #     es_mean[es_mean >= .5] = 1
        #     es_mean[es_mean <= -.5] = -1
        #     es_mean[np.abs(es_mean) < .5] = 0
        #     ax = fig.add_subplot(gs[ind_row, ind_col])
        #     plt.scatter(self.grid_wgs[:, 1], self.grid_wgs[:, 0], c=es_mean,
        #                 cmap=get_cmap("RdBu", 3), vmin=-1, vmax=1)
        #     plt.colorbar()
        #     plt.plot(self.polygon_border_wgs[:, 1], self.polygon_border_wgs[:, 0], 'r-.')
        #     plt.plot(self.polygon_obstacle_wgs[:, 1], self.polygon_obstacle_wgs[:, 0], 'r-.')
        #     plt.xlabel("Longitude")
        #     plt.xticks(self.lon_ticks)
        #     plt.xlim([self.lon_min, self.lon_max])
        #     plt.ylim([self.lat_min, self.lat_max])
        #     plt.title(planner.upper() + " " + item.upper())
        #
        #     # s1, plot std
        #     es_std = np.std(self.es_diff[planner][item][step, replicate_id, :, :], axis=0)
        #     ax = fig.add_subplot(gs[ind_row + 1, ind_col])
        #     plt.scatter(self.grid_wgs[:, 1], self.grid_wgs[:, 0], c=es_std,
        #                 cmap=get_cmap("Blues", 3), vmin=0, vmax=1)
        #     plt.colorbar()
        #     plt.plot(self.polygon_border_wgs[:, 1], self.polygon_border_wgs[:, 0], 'r-.')
        #     plt.plot(self.polygon_obstacle_wgs[:, 1], self.polygon_obstacle_wgs[:, 0], 'r-.')
        #     plt.xlabel("Longitude")
        #     plt.xticks(self.lon_ticks)
        #     plt.xlim([self.lon_min, self.lon_max])
        #     plt.ylim([self.lat_min, self.lat_max])
        #     plt.title(planner.upper() + " " + item.upper())
        #
        # for zz in tqdm(range(self.num_replicates)):
        #     fpath = figpath + "/../ES/R_{:03d}/".format(zz)
        #     checkfolder(fpath)
        #     for k in range(8):
        #         fig = plt.figure(figsize=(25, 32))
        #         gs = GridSpec(nrows=4, ncols=3, figure=fig)
        #
        #         for i in range(len(self.planners)):
        #             for j in range(len(self.cv)):
        #                 make_subplot_es(self.planners[i], self.cv[j], i * 2, j, k, zz)
        #         plt.savefig(fpath + "P_{:03d}.png".format(k))
        #         plt.close("all")

    def plot_temporal_traffic_flow_density_map4paper(self) -> None:
        """
        Plot the traffic flow density map for each specific case. For the paper. No need to save all the figures.
        """
        num_steps = [29, 59, 89, 119]
        time_start = datetime.datetime(2022, 5, 11, 9, 52)
        fpath = figpath + "TrafficFlow/"
        checkfolder(fpath)

        def make_subplot(planner, item):
            fig = plt.figure(figsize=(48, 10))
            gs = GridSpec(nrows=1, ncols=4, figure=fig)
            for i in range(len(num_steps)):
                num_step = num_steps[i]
                traj = self.trajectory[planner][item][:, :num_step, :]
                lat, lon = WGS.xy2latlon(traj[:, :, 0], traj[:, :, 1])
                df = pd.DataFrame(np.stack((lat.flatten(), lon.flatten()), axis=1), columns=['lat', 'lon'])
                ax = fig.add_subplot(gs[i])
                sns.kdeplot(df, x='lon', y='lat', fill=True, cmap="Reds", levels=25, thresh=.1)
                plt.plot(self.polygon_border_wgs[:, 1], self.polygon_border_wgs[:, 0], 'k-.')
                plg = plt.Polygon(np.fliplr(self.polygon_obstacle_wgs), facecolor='w', edgecolor='k', fill=True,
                                  linestyle='-.')
                plt.gca().add_patch(plg)

                # Plot border and create a mask
                border_path = Path(self.polygon_border_wgs[:, ::-1])
                border_patch = PathPatch(border_path, facecolor='none', edgecolor='none')
                ax.add_patch(border_patch)

                # Mask KDE using border path
                for collection in ax.collections:
                    collection.set_clip_path(border_patch)

                if i == 0:
                    plt.ylabel("Latitude")
                else:
                    plt.ylabel("")
                plt.xlabel("Longitude")
                date_string = time_start + datetime.timedelta(hours=(num_step + 1)/30 * 2)
                plt.title(f"Density map at " + date_string.strftime("%H:%M"))
                plt.xticks(self.lon_ticks)
                plt.xlim([self.lon_min, self.lon_max])
                plt.ylim([self.lat_min, self.lat_max])

            plt.savefig(fpath + "TF_{:s}_{:s}.png".format(planner, item))
            plt.close("all")

        for planner in self.planners:
            for item in self.cv:
                make_subplot(planner, item)

    def plot_temporal_traffic_density_map(self, step: 'int', row_ind, col_ind, fig, gs) -> None:
        """
        Plot the traffic density map for each specific case.

        !!! Note: the trajectory starts from index 1 instead of 0.
        """
        def make_subplot(planner, item, step: int = 0):
            traj = self.trajectory[planner][item][:, :step, :]
            lat, lon = WGS.xy2latlon(traj[:, :, 0], traj[:, :, 1])
            for k in range(lat.shape[0]):
                plt.plot(lon[k, :], lat[k, :], 'k.-', alpha=.005)
            df = pd.DataFrame(np.stack((lat.flatten(), lon.flatten()), axis=1), columns=['lat', 'lon'])
            sns.kdeplot(df, x='lon', y='lat', fill=True, cmap="Blues", levels=25, thresh=.1)
            plt.plot(self.polygon_border_wgs[:, 1], self.polygon_border_wgs[:, 0], 'r-.')
            plg = plt.Polygon(np.fliplr(self.polygon_obstacle_wgs), facecolor='w', edgecolor='r', fill=True,
                              linestyle='-.')
            plt.gca().add_patch(plg)
            plt.xlabel("Longitude")
            plt.xticks(self.lon_ticks)
            plt.xlim([self.lon_min, self.lon_max])
            plt.ylim([self.lat_min, self.lat_max])
            plt.title(planner.upper() + " " + item.upper())

        ax = fig.add_subplot(gs[row_ind, col_ind])
        make_subplot('myopic', 'eibv', step=step)

        ax = fig.add_subplot(gs[row_ind + 1, col_ind])
        make_subplot('myopic', 'ivr', step=step)

        ax = fig.add_subplot(gs[row_ind + 2, col_ind])
        make_subplot('myopic', 'equal', step=step)

        ax = fig.add_subplot(gs[row_ind, col_ind + 1])
        make_subplot('rrt', 'eibv', step=step)

        ax = fig.add_subplot(gs[row_ind + 1, col_ind + 1])
        make_subplot('rrt', 'ivr', step=step)

        ax = fig.add_subplot(gs[row_ind + 2, col_ind + 1])
        make_subplot('rrt', 'equal', step=step)

    def plot_metrics4paper(self) -> None:
        """
        Plot the metrics for each specific case. For the paper. No need to save all the figures.
        """
        fpath = figpath + "Metrics/"
        checkfolder(fpath)
        def make_subplot_metric(data, metric, title, ax):
            # Define a mapping for the legend labels
            legend_map = {
                'eibv': 'EIBV dominant',
                'ivr': 'IVR dominant',
                'equal': 'Equal weighted'
            }

            # Preparing data for Seaborn
            df_list = []
            for key in ['eibv', 'ivr', 'equal']:
                temp_df = pd.DataFrame({
                    'Time Step': np.tile(np.arange(self.num_steps), self.num_replicates),
                    metric: data[key].flatten(),
                    'Type': [legend_map[key]] * self.num_steps * self.num_replicates  # Use the mapping here
                })
                df_list.append(temp_df)

            df = pd.concat(df_list, axis=0)

            # Using Seaborn's lineplot with uncertainty envelopes
            sns.lineplot(data=df, x='Time Step', y=metric, hue='Type', ax=ax, ci="sd", err_style="band")

            # ax.errorbar(np.arange(self.num_steps), y=np.mean(data['eibv'], axis=0),
            #             yerr=np.std(data['eibv'], axis=0) / np.sqrt(self.num_replicates) * 1.645, fmt="-o",
            #             capsize=5, label="EIBV dominant")
            # ax.errorbar(np.arange(self.num_steps), y=np.mean(data['ivr'], axis=0),
            #             yerr=np.std(data['ivr'], axis=0) / np.sqrt(self.num_replicates) * 1.645, fmt="-o",
            #             capsize=5, label="IVR dominant")
            # ax.errorbar(np.arange(self.num_steps), y=np.mean(data['equal'], axis=0),
            #             yerr=np.std(data['equal'], axis=0) / np.sqrt(self.num_replicates) * 1.645, fmt="-o",
            #             capsize=5, label="Equal weighted")
            ax.set_xticks(self.xticks, self.xticklabels)
            ax.set_xlim([0, self.num_steps])
            ax.set_xlabel("Time Step")
            ax.set_ylabel(metric)
            ax.set_title(title)
            plt.legend(loc="upper left")
            if metric == "ibv":
                ax.set_ylim([self.ibv_min, self.ibv_max])
            elif metric == "rmse":
                ax.set_ylim([self.rmse_min, self.rmse_max])
            elif metric == "vr":
                ax.set_ylim([self.vr_min, self.vr_max])
            else:
                pass
        def make_subplot(metric):
            if metric == "ibv":
                data = self.ibv
            elif metric == "rmse":
                data = self.rmse
            elif metric == "vr":
                data = self.vr
            else:
                pass
            fig = plt.figure(figsize=(24, 10))
            gs = GridSpec(nrows=1, ncols=2, figure=fig)

            ax1 = fig.add_subplot(gs[0])
            make_subplot_metric(data['myopic'], metric.upper(), "Myopic", ax1)

            ax2 = fig.add_subplot(gs[1])
            make_subplot_metric(data['rrt'], metric.upper(), "RRT*", ax2)

            # Determine the global y-limits
            ylim_min = min(ax1.get_ylim()[0], ax2.get_ylim()[0])
            ylim_max = max(ax1.get_ylim()[1], ax2.get_ylim()[1])

            # Set y-limits for both subplots
            ax1.set_ylim([ylim_min, ylim_max])
            ax2.set_ylim([ylim_min, ylim_max])

            plt.savefig(fpath + f"{metric}.png")
            plt.close("all")



        # def make_subplot_metric(data, metric, ax):
        #     # Define line styles for different methods
        #     linestyle_map = {
        #         'myopic': '-',
        #         'rrt': '--'
        #     }
        #
        #     # Define colors for different keys
        #     color_map = {
        #         'EIBV dominant': 'blue',
        #         'IVR dominant': 'red',
        #         'Equal weighted': 'green'
        #     }
        #
        #     # Define a mapping for the legend labels
        #     legend_map = {
        #         'eibv': 'EIBV dominant',
        #         'ivr': 'IVR dominant',
        #         'equal': 'Equal weighted'
        #     }
        #
        #     # Preparing data for Seaborn
        #     # Preparing data for Seaborn
        #     df_list = []
        #     for method in ['myopic', 'rrt']:
        #         for key in ['eibv', 'ivr', 'equal']:
        #             temp_df = pd.DataFrame({
        #                 'Time Step': np.tile(np.arange(self.num_steps), self.num_replicates),
        #                 metric: data[method][key].flatten(),
        #                 'Type': [legend_map[key]] * self.num_steps * self.num_replicates,
        #                 'LineStyle': [linestyle_map[method]] * self.num_steps * self.num_replicates
        #                 # Additional column for linestyle
        #             })
        #             df_list.append(temp_df)
        #
        #     df = pd.concat(df_list, axis=0)
        #
        #     # Using Seaborn's lineplot with uncertainty envelopes
        #     sns.lineplot(data=df, x='Time Step', y=metric, hue='Type', style="LineStyle", palette=color_map, ax=ax,
        #                  ci="sd", err_style="band")
        #
        #     ax.set_xticks(self.xticks, self.xticklabels)
        #     ax.set_xlim([0, self.num_steps])
        #     ax.set_xlabel("Time Step")
        #     ax.set_ylabel(metric)
        #     ax.set_title(metric.upper())
        #     plt.legend(loc="upper left")
        #     if metric == "ibv":
        #         ax.set_ylim([self.ibv_min, self.ibv_max])
        #     elif metric == "rmse":
        #         ax.set_ylim([self.rmse_min, self.rmse_max])
        #     elif metric == "vr":
        #         ax.set_ylim([self.vr_min, self.vr_max])
        #     else:
        #         pass
        #
        # def make_subplot(metric):
        #     if metric == "ibv":
        #         data = self.ibv
        #     elif metric == "rmse":
        #         data = self.rmse
        #     elif metric == "vr":
        #         data = self.vr
        #     else:
        #         pass
        #     fig, ax = plt.subplots(figsize=(12, 8))
        #     make_subplot_metric(data, metric, ax)
        #     plt.savefig(fpath + f"{metric}.png")
        #     plt.close("all")
        #
        make_subplot("ibv")
        make_subplot("rmse")
        make_subplot("vr")

        self.ibv_min


    def plot_metrics(self, step, row_ind, col_ind, fig, gs) -> None:
        """
        Plot the metrics for each specific case.
        """
        def make_subplot_metric(data, metric, planner):
            ax.errorbar(np.arange(step), y=np.mean(data['eibv'][:, :step], axis=0),
                        yerr=np.std(data['eibv'][:, :step], axis=0)/np.sqrt(self.num_replicates) * 1.645, fmt="-o", capsize=5,
                        label="EIBV dominant" + planner)
            ax.errorbar(np.arange(step), y=np.mean(data['ivr'][:, :step], axis=0),
                        yerr=np.std(data['ivr'][:, :step], axis=0)/np.sqrt(self.num_replicates) * 1.645, fmt="-o", capsize=5,
                        label="IVR dominant" + planner)
            ax.errorbar(np.arange(step), y=np.mean(data['equal'][:, :step], axis=0),
                        yerr=np.std(data['equal'][:, :step], axis=0)/np.sqrt(self.num_replicates) * 1.645, fmt="-o", capsize=5,
                        label="Equal weighted" + planner)
            ax.set_xticks(self.xticks, self.xticklabels)
            ax.set_xlim([0, self.num_steps])
            ax.set_xlabel("Time Step")
            ax.set_ylabel(metric)
            plt.legend(loc="upper right")
            if metric == "ibv":
                ax.set_ylim([self.ibv_min, self.ibv_max])
            elif metric == "rmse":
                ax.set_ylim([self.rmse_min, self.rmse_max])
            elif metric == "vr":
                ax.set_ylim([self.vr_min, self.vr_max])
            else:
                pass

        ax = fig.add_subplot(gs[row_ind, col_ind])
        make_subplot_metric(self.ibv['myopic'], "IBV", "Myopic")

        ax = fig.add_subplot(gs[row_ind + 1, col_ind])
        make_subplot_metric(self.rmse['myopic'], "RMSE", "Myopic")

        ax = fig.add_subplot(gs[row_ind + 2, col_ind])
        make_subplot_metric(self.vr['myopic'], "VR", "Myopic")

        ax = fig.add_subplot(gs[row_ind, col_ind + 1])
        make_subplot_metric(self.ibv['rrt'], "IBV", "RRT")

        ax = fig.add_subplot(gs[row_ind + 1, col_ind + 1])
        make_subplot_metric(self.rmse['rrt'], "RMSE", "RRT")

        ax = fig.add_subplot(gs[row_ind + 2, col_ind + 1])
        make_subplot_metric(self.vr['rrt'], "VR", "RRT")

    def plot_ground_truth(self) -> None:
        """
        Plot the ground truth for one replicate to check.
        """
        num_replicate = 0
        fpath = figpath + "Truth/"
        checkfolder(fpath)
        tid_start = 10

        ind = np.random.randint(0, self.num_replicates, 1)
        fig = plt.figure(figsize=(48, 10))
        gs = GridSpec(nrows=1, ncols=4, figure=fig)
        for i in range(29, self.num_steps, 30):
            truth_temp = np.mean(self.truth['myopic']['eibv'][ind, i, :], axis=0)
            ax = fig.add_subplot(gs[i // 30])
            print(i)
            if i == 119:
                cbar = False
            else:
                cbar = False
            self.plotf_vector(self.grid_wgs[:, 0], self.grid_wgs[:, 1], truth_temp, alpha=1., cmap=get_cmap("BrBG", 10),
                              title="Truth", vmin=10, vmax=33, stepsize=1.5, threshold=self.threshold,
                              colorbar=cbar, cbar_title="Salinity (PSU)", polygon_border=self.polygon_border_wgs,
                              polygon_obstacle=self.polygon_obstacle_wgs)
            ax.set_xlabel("Longitude")
            if i == 29:
                ax.set_ylabel("Latitude")
            ax.set_title(f"Ground truth field at {int(tid_start + i / 30 * 2)}:00 ")
            # cbar = plt.colorbar(, ax=ax, label='Colorbar Title', pad=-0.15, orientation='vertical')

        # plt.savefig(fpath + "groundtruth2.png")
        # plt.close("all")
        plt.show()
        truth_temp

    def load_raw_data_from_replicate_files(self) -> None:
        """
        Load raw data from the replicate files. Needed after running the simulation study.
        """
        def load_single_file_data(file) -> 'dict':
            """
            Load data from a single file.
            """
            single_file_dataset = {}
            single_file_dataset[file] = {}
            for item in self.cv:
                filepath = folderpath + file + "/" + item + "/"
                single_file_dataset[file][item] = {}
                for planner in self.planners:
                    if planner == "myopic":
                        data = np.load(filepath + "myopic.npz")
                    else:
                        data = np.load(filepath + "rrtstar.npz")
                    single_file_dataset[file][item][planner] = {}
                    for metric in self.metrics:
                        single_file_dataset[file][item][planner][metric] = data[metric]
            return single_file_dataset

        t0 = time()
        files = os.listdir(folderpath)
        dataset = {}

        # s0, initialize the dataset dictionary.
        for i in range(self.num_replicates):
            num_replicate = "R_{:03d}".format(i)
            dataset[num_replicate] = {}
            for item in self.cv:
                dataset[num_replicate][item] = {}
                for planner in self.planners:
                    dataset[num_replicate][item][planner] = {}
                    for metric in self.metrics:
                        # print("num_replicate: ", num_replicate, " | item: ", item, " | planner: ", planner, " | metric: ", metric)
                        dataset[num_replicate][item][planner][metric] = None

        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = [executor.submit(load_single_file_data, file) for file in files if file.startswith("R_")]
            for future in tqdm(futures):
                data = future.result()
                file = list(data.keys())[0]
                dataset[file] = data[file]

        print("Loading data takes {:.2f} seconds.".format(time() - t0))
        self.dataset = dataset

    def organize_data_to_dict(self) -> None:
        """
        Organize the data to a dictionary. Only once, no need for future use.
        """
        self.trajectory = {}
        self.ibv = {}
        self.rmse = {}
        self.vr = {}
        self.mu = {}
        self.cov = {}
        self.sigma = {}
        self.truth = {}

        t0 = time()
        for planner in self.planners:
            self.trajectory[planner] = {}
            self.ibv[planner] = {}
            self.rmse[planner] = {}
            self.vr[planner] = {}
            self.mu[planner] = {}
            self.cov[planner] = {}
            self.sigma[planner] = {}
            self.truth[planner] = {}
            for item in self.cv:
                self.trajectory[planner][item] = np.empty([self.num_replicates, self.num_steps, 2])
                self.ibv[planner][item] = np.empty([self.num_replicates, self.num_steps])
                self.rmse[planner][item] = np.empty([self.num_replicates, self.num_steps])
                self.vr[planner][item] = np.empty([self.num_replicates, self.num_steps])
                self.mu[planner][item] = np.empty([self.num_replicates, self.num_steps, self.Ngrid])
                self.cov[planner][item] = np.empty([self.num_replicates, self.num_steps // 15, self.Ngrid, self.Ngrid])
                self.sigma[planner][item] = np.empty([self.num_replicates, self.num_steps, self.Ngrid])
                self.truth[planner][item] = np.empty([self.num_replicates, self.num_steps, self.Ngrid])
        for i in range(self.num_replicates):
            num_replicate = "R_{:03d}".format(i)
            for planner in self.planners:
                for item in self.cv:
                    self.trajectory[planner][item][i, :, :] = self.dataset[num_replicate][item][planner]['traj']
                    self.ibv[planner][item][i, :] = self.dataset[num_replicate][item][planner]['ibv']
                    self.rmse[planner][item][i, :] = self.dataset[num_replicate][item][planner]['rmse']
                    self.vr[planner][item][i, :] = self.dataset[num_replicate][item][planner]['vr']
                    self.mu[planner][item][i, :, :] = self.dataset[num_replicate][item][planner]['mu']
                    self.cov[planner][item][i, :, :, :] = self.dataset[num_replicate][item][planner]['cov']
                    self.sigma[planner][item][i, :, :] = self.dataset[num_replicate][item][planner]['sigma']
                    self.truth[planner][item][i, :, :] = self.dataset[num_replicate][item][planner]['truth']

        """ Calculate the excursion set. """
        B = 100
        N_cov = self.cov['myopic']['eibv'].shape[1]
        self.es_diff = {}
        def get_excursion_set(mu: np.ndarray) -> np.ndarray:
            """
            Return the excursion set.
            """
            es = np.zeros_like(mu)
            es[mu <= self.threshold] = 1
            return es

        t0 = time()
        for planner in self.planners:
            self.es_diff[planner] = {}
            for item in self.cv:
                self.es_diff[planner][item] = []
                for i in range(N_cov):
                    self.es_diff[planner][item].append([])
                    for j in range(self.num_replicates):
                        self.es_diff[planner][item][i].append([])
                        mu_cond = self.mu[planner][item][j, i * 15, :]
                        cov_cond = self.cov[planner][item][j, i, :, :]
                        L_cond = np.linalg.cholesky(cov_cond)
                        es_truth = get_excursion_set(self.truth[planner][item][j, i * 15, :])
                        for k in range(B):
                            mu_sample = mu_cond + L_cond @ np.random.randn(self.Ngrid)
                            es_sample = get_excursion_set(mu_sample)
                            es_diff_temp = es_sample - es_truth
                            self.es_diff[planner][item][i][j].append(es_diff_temp)
                self.es_diff[planner][item] = np.array(self.es_diff[planner][item])
        print("Generating excursion set takes {:.2f} seconds.".format(time() - t0))

        fpath = os.getcwd() + "/../simulation_result/Synced/"
        dump(self.trajectory, fpath + "trajectory.joblib")
        dump(self.ibv, fpath + "ibv.joblib")
        dump(self.rmse, fpath + "rmse.joblib")
        dump(self.vr, fpath + "vr.joblib")
        dump(self.mu, fpath + "mu.joblib")
        dump(self.cov, fpath + "cov.joblib")
        dump(self.sigma, fpath + "sigma.joblib")
        dump(self.truth, fpath + "truth.joblib")

        """ Save the excursion set area difference. """
        dump(self.es_diff, fpath + "es_diff.joblib")

        t1 = time()
        print("Reorganizing data takes {:.2f} seconds.".format(t1 - t0))

    def plot_trajectory_temporal(self) -> None:

        cv = ['eibv', 'ivr', 'equal']
        planners = ['myopic', 'rrt']
        fields = ['mu', 'sigma', 'truth']

        def make_subplot(traj, value, num_step, cmap=get_cmap("BrBG", 10),
                         vmin: float = 10, vmax: float = 30, name: str = "Test"):
            plt.scatter(self.grid_wgs[:, 1], self.grid_wgs[:, 0], c=value[num_step, :],
                        cmap=cmap, vmin=vmin, vmax=vmax, alpha=.5)
            plt.colorbar()
            lat, lon = WGS.xy2latlon(traj[:num_step, 0], traj[:num_step, 1])
            plt.plot(lon, lat, 'k.-')
            plt.plot(self.polygon_border_wgs[:, 1], self.polygon_border_wgs[:, 0], 'r-.')
            plg = plt.Polygon(np.fliplr(self.polygon_obstacle_wgs), facecolor='w', edgecolor='r', fill=True,
                              linestyle='-.')
            plt.gca().add_patch(plg)
            plt.xlabel("Longitude")
            plt.xticks(self.lon_ticks)
            plt.xlim([self.lon_min, self.lon_max])
            plt.ylim([self.lat_min, self.lat_max])
            plt.title(name)

        for num_step in range(1, self.num_steps):
            print("NUM: ", num_step)
            for num_replicate in range(self.num_replicates):
                for field in fields:
                    fpath = figpath + "/R_{:03d}".format(num_replicate) + "/" + field + "/"
                    checkfolder(fpath)
                    fig = plt.figure(figsize=(25, 15))
                    gs = GridSpec(nrows=2, ncols=3)

                    for i in range(len(planners)):
                        for j in range(len(cv)):
                            print("planner: ", planners[i], " cv: ", cv[j], " field: ", field)
                            traj = self.dataset["R_{:03d}".format(num_replicate)][cv[j]][planners[i]]["traj"]
                            data_field = self.dataset["R_{:03d}".format(num_replicate)][cv[j]][planners[i]][field]

                            ax = fig.add_subplot(gs[i, j])
                            if field == "sigma":
                                vmin=0
                                vmax=.5
                                cmap=get_cmap("RdBu", 10)
                            else:
                                vmin=10
                                vmax=30
                                cmap=get_cmap("BrBG", 10)
                            make_subplot(traj, data_field, num_step, name=planners[i] + " " + cv[j],
                                         cmap=cmap, vmin=vmin, vmax=vmax)

                    plt.savefig(fpath + "/P_{:03d}.png".format(num_step))

        self.dataset
        print("he")

    def load_sim_data(self, string: str = "/sigma_10/nugget_025/SimulatorRRTStar") -> tuple:
        """
        Reorganize the data from the simulation study.
        """
        traj = np.empty([self.num_replicates, 3, self.num_steps, 2])
        mu = np.empty([self.num_replicates, 3, self.num_steps, self.Ngrid])
        sigma = np.empty([self.num_replicates, 3, self.num_steps, self.Ngrid])
        truth = np.empty([self.num_replicates, 3, self.num_steps, self.Ngrid])

        for i in range(self.num_replicates):
            rep = "R_{:03d}".format(i)
            datapath = folderpath + rep + string
            data_eibv_dominant = np.load(datapath + "eibv.npz")
            data_ivr_dominant = np.load(datapath + "ivr.npz")
            data_equal = np.load(datapath + "eq.npz")

            # s0, extract trajectory
            traj[i, 0, :, :] = data_eibv_dominant["traj"]
            traj[i, 1, :, :] = data_ivr_dominant["traj"]
            traj[i, 2, :, :] = data_equal["traj"]

            # s1, extract mu
            mu[i, 0, :, :] = data_eibv_dominant["mu_data"]
            mu[i, 1, :, :] = data_ivr_dominant["mu_data"]
            mu[i, 2, :, :] = data_equal["mu_data"]

            # s2, extract sigma
            sigma[i, 0, :, :] = data_eibv_dominant["sigma_data"]
            sigma[i, 1, :, :] = data_ivr_dominant["sigma_data"]
            sigma[i, 2, :, :] = data_equal["sigma_data"]

            # s3, extract truth
            truth[i, 0, :, :] = data_eibv_dominant["mu_truth_data"]
            truth[i, 1, :, :] = data_ivr_dominant["mu_truth_data"]
            truth[i, 2, :, :] = data_equal["mu_truth_data"]
        return traj, mu, sigma, truth

    def plot_trajectory_static_truth(self) -> None:
        def make_subplot(traj, num_step, j: int = 0):
            for k in range(traj.shape[0]):
                lat, lon = WGS.xy2latlon(traj[k, j, :num_step, 0], traj[k, j, :num_step, 1])
                # df = pd.DataFrame(np.stack((lat.flatten(), lon.flatten()), axis=1), columns=['lat', 'lon'])
                # sns.kdeplot(df, x='lon', y='lat', fill=True, cmap="Blues", levels=25, thresh=.1)
                plt.plot(lon, lat, 'k.-', alpha=.5)
            plt.plot(self.polygon_border_wgs[:, 1], self.polygon_border_wgs[:, 0], 'r-.')
            plg = plt.Polygon(np.fliplr(self.polygon_obstacle_wgs), facecolor='w', edgecolor='r', fill=True,
                              linestyle='-.')
            plt.gca().add_patch(plg)
            plt.xlabel("Longitude")

            plt.xticks(self.lon_ticks)

            plt.xlim([self.lon_min, self.lon_max])
            plt.ylim([self.lat_min, self.lat_max])
            plt.title(["EIBV dominant" if j == 0 else "IVR dominant" if j == 1 else "Equal"][0])

        for num_step in range(1, self.num_steps):
            print("NUM: ", num_step)
            fig = plt.figure(figsize=(36, 20))
            gs = GridSpec(nrows=2, ncols=3)
            ax = fig.add_subplot(gs[0, 0])
            make_subplot(self.traj_rrt, num_step, 0)

            ax = fig.add_subplot(gs[0, 1])
            make_subplot(self.traj_rrt, num_step, 1)

            ax = fig.add_subplot(gs[0, 2])
            make_subplot(self.traj_rrt, num_step, 2)

            ax = fig.add_subplot(gs[1, 0])
            make_subplot(self.traj_myopic, num_step, 0)

            ax = fig.add_subplot(gs[1, 1])
            make_subplot(self.traj_myopic, num_step, 1)

            ax = fig.add_subplot(gs[1, 2])
            make_subplot(self.traj_myopic, num_step, 2)

            plt.savefig(figpath + "/P_{:03d}.png".format(num_step))
            plt.close("all")

        self.rmse
        pass

    def load_metricdata4simulator(self, string) -> tuple:
        self.traj = None
        self.ibv = None
        self.rmse = None
        self.vr = None

        self.traj = np.empty([0, 3, self.num_steps, 2])
        self.ibv = np.empty([0, 3, self.num_steps])
        self.rmse = np.empty([0, 3, self.num_steps])
        self.vr = np.empty([0, 3, self.num_steps])

        for i in range(self.num_replicates):
            rep = "R_{:03d}".format(i)

            datapath = folderpath + rep + string
            r_traj = np.load(datapath + "traj.npy").reshape(1, 3, self.num_steps, 2)
            r_ibv = np.load(datapath + "ibv.npy").reshape(1, 3, self.num_steps)
            r_vr = np.load(datapath + "vr.npy").reshape(1, 3, self.num_steps)
            r_rmse = np.load(datapath + "rmse.npy").reshape(1, 3, self.num_steps)

            self.traj = np.append(self.traj, r_traj, axis=0)
            self.ibv = np.append(self.ibv, r_ibv, axis=0)
            self.vr = np.append(self.vr, r_vr, axis=0)
            self.rmse = np.append(self.rmse, r_rmse, axis=0)

        return self.traj, self.ibv, self.vr, self.rmse

    def plot_figure_density(self, traj: np.ndarray, num_step: int, title: str) -> None:
        print("num: ", num_step)

        def make_subplot(j: int = 0):
            lat, lon = WGS.xy2latlon(traj[:, j, :num_step, 0], traj[:, j, :num_step, 1])
            df = pd.DataFrame(np.stack((lat.flatten(), lon.flatten()), axis=1), columns=['lat', 'lon'])
            sns.kdeplot(df, x='lon', y='lat', fill=True, cmap="Blues", levels=25, thresh=.1)
            plt.plot(self.polygon_border_wgs[:, 1], self.polygon_border_wgs[:, 0], 'r-.')
            plg = plt.Polygon(np.fliplr(self.polygon_obstacle_wgs), facecolor='w', edgecolor='r', fill=True,
                              linestyle='-.')
            plt.gca().add_patch(plg)
            plt.xlabel("Longitude")
            plt.xticks(self.lon_ticks)

            plt.xlim([self.lon_min, self.lon_max])
            plt.ylim([self.lat_min, self.lat_max])
            plt.title(["EIBV dominant" if j == 0 else "IVR dominant" if j == 1 else "Equal"][0])

        fig = plt.figure(figsize=(36, 10))
        plt.ylabel("Latitude")
        plt.yticks(self.lat_ticks)

        gs = GridSpec(nrows=1, ncols=3)

        ax = fig.add_subplot(gs[0])
        make_subplot(0)

        ax = fig.add_subplot(gs[1])
        make_subplot(1)

        ax = fig.add_subplot(gs[2])
        make_subplot(2)

        plt.suptitle(title + " density map at time step {:03d}".format(num_step))
        plt.savefig(figpath + "/Simulation/P_{:03d}.png".format(num_step))
        # plt.show()
        plt.close("all")

    def plot_metric_analysis(self) -> None:
        # def organize_dataset(data_myopic, data_rrt, num_steps, metric) -> 'pd.DataFrame':
        #     dataset = []
        #     for i in range(len(num_steps)):
        #         for j in range(data_myopic.shape[0]):
        #             for k in range(data_myopic.shape[1]):
        #                 if k == 0:
        #                     cv = "EIBV dominant"
        #                 elif k == 1:
        #                     cv = "IVR dominant"
        #                 else:
        #                     cv = "Equal weighted"
        #                 dataset.append([data_myopic[j, k, num_steps[i]-1], 'Myopic', '{:2d}'.format(num_steps[i]), cv])
        #                 dataset.append([data_rrt[j, k, num_steps[i]-1], 'RRT*', '{:2d}'.format(num_steps[i]), cv])
        #     df = pd.DataFrame(dataset, columns=[metric, 'Path planner', 'Step', 'cost valley'])
        #     return df

        def organize_dataset(data, max_num: int, metric) -> 'pd.DataFrame':
            dataset = []
            for i in range(0, max_num):
                for j in range(data.shape[0]):
                    for k in range(data.shape[1]):
                        if k == 0:
                            cv = "EIBV dominant"
                        elif k == 1:
                            cv = "IVR dominant"
                        else:
                            cv = "Equal weighted"
                        dataset.append([data[j, k, i], "{:03d}".format(i), cv])
            df = pd.DataFrame(dataset, columns=[metric, 'Step', 'Cost valley'])
            return df

        # num_steps = [20, 45, 70, 100]
        # sns.set_palette(["#EED2EE", "#ADD8E6", "#BFEFFF", "#FFDAB9"])
        # sns.set_palette(["#FFB347", "#FF6961", "#FDB813"])
        # sns.set_palette(["#98FB98", "#EE82EE", "#F08080"])
        # sns.set_palette(["#ADD8E6", "#FFC0CB", "#9370DB"])
        # sns.set_palette(["#FFFF00", "#FFA07A", "#FF4500"])

        # max_num = 100
        max_num = 45
        ticks = np.arange(0, max_num, 5)
        ticklabels = ['{:d}'.format(i) for i in ticks]
        df = organize_dataset(self.ibv_rrt, max_num, "IBV")
        fig = plt.figure(figsize=(30, 8))
        gs = GridSpec(nrows=1, ncols=3)
        ax = fig.add_subplot(gs[0])
        g = sns.lineplot(x="Step", y="IBV", hue="Cost valley", data=df)
        g.set_xticks(ticks)  # <--- set the ticks first
        g.set_xticklabels(ticklabels)  # <--- set the labels second

        ax = fig.add_subplot(gs[1])
        df = organize_dataset(self.vr_rrt, max_num, "VR")
        g = sns.lineplot(x="Step", y="VR", hue="Cost valley", data=df)
        g.set_xticks(ticks)  # <--- set the ticks first
        g.set_xticklabels(ticklabels)  # <--- set the labels second

        ax = fig.add_subplot(gs[2])
        df = organize_dataset(self.rmse_rrt, max_num, "RMSE")
        g = sns.lineplot(x="Step", y="RMSE", hue="Cost valley", data=df)
        g.set_xticks(ticks)  # <--- set the ticks first
        g.set_xticklabels(ticklabels)  # <--- set the labels second

        # plt.savefig(figpath + "Simulation/result_rrt.png")

        # g = sns.catplot(x="Step", y="IBV", hue="cost valley", data=df, kind="box")
        # g.set_titles("{col_name} {col_var}")
        # fig = plt.gcf()
        plt.show()
        # fig.savefig(figpath + "Simulation/IBV.png")
        # fig.show()

        # df = organize_dataset(self.vr_myopic, self.vr_rrt, num_steps, "VR")
        # plt.figure(figsize=(15, 7))
        # g = sns.catplot(x="Step", y="VR", hue="Path planner", col="cost valley", data=df, kind="box")
        # g.set_titles("{col_name} {col_var}")
        # fig = plt.gcf()
        # fig.savefig(figpath + "Simulation/VR.png")
        # fig.show(fig)
        #
        # df = organize_dataset(self.rmse_myopic, self.rmse_rrt, num_steps, "RMSE")
        # plt.figure(figsize=(15, 7))
        # g = sns.catplot(x="Step", y="RMSE", hue="Path planner", col="cost valley", data=df, kind="box")
        # g.set_titles("{col_name} {col_var}")
        # fig = plt.gcf()
        # fig.savefig(figpath + "Simulation/RMSE.png")
        # fig.show()
        # # plt.close("all")
        # plt.show()

        plt.show()
        pass

    def plot_density_map_for_cost_valley(self, traj: np.ndarray, num_step: list, title: str, ind_cv: int=0) -> None:

        def make_subplot(num_step: int = 0):
            lat, lon = WGS.xy2latlon(traj[:, ind_cv, :num_step, 0], traj[:, ind_cv, :num_step, 1])
            df = pd.DataFrame(np.stack((lat.flatten(), lon.flatten()), axis=1), columns=['lat', 'lon'])
            if title.upper().find("RRT") != -1:
                thresh = .025
            else:
                thresh = .25
            sns.kdeplot(df, x='lon', y='lat', fill=True, cmap="Blues", levels=25, thresh=thresh)
            plt.plot(self.polygon_border_wgs[:, 1], self.polygon_border_wgs[:, 0], 'r-.')
            plg = plt.Polygon(np.fliplr(self.polygon_obstacle_wgs), facecolor='w', edgecolor='r', fill=True,
                              linestyle='-.')
            plt.gca().add_patch(plg)
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.yticks(self.lat_ticks)
            plt.xticks(self.lon_ticks)
            plt.xlim([self.lon_min, self.lon_max])
            plt.ylim([self.lat_min, self.lat_max])
            plt.title("Step {:d}".format(num_step))
            # plt.title(["EIBV dominant" if ind_cv == 0 else "IVR dominant" if ind_cv == 1 else "Equal"][0])

        fig = plt.figure(figsize=(35, 7))
        gs = GridSpec(nrows=1, ncols=len(num_step))

        for i in range(len(num_step)):
            ax = fig.add_subplot(gs[i])
            make_subplot(num_step[i])

        # plt.suptitle(title + " traffic density map")
        stitle = ["eibv_dominant" if ind_cv == 0 else "ivr_dominant" if ind_cv == 1 else "equal"][0]
        plt.savefig(figpath + "Simulation/" + stitle + "_{}.png".format(title))
        plt.close("all")

    def plot_traffic_density_map(self) -> None:
        num_steps = [20, 45, 70, 100]

        traj_rrt = self.traj_rrt
        traj_myopic = self.traj_myopic

        # self.plot_density_map_for_cost_valley(traj_rrt, num_steps, "rrt")
        [self.plot_density_map_for_cost_valley(traj_rrt, num_steps, "rrt", ind_cv=k) for k in range(3)]
        [self.plot_density_map_for_cost_valley(traj_myopic, num_steps, "myopic", ind_cv=k) for k in range(3)]

        # [self.plot_figure_density(traj_myopic, i, "Myopic") for i in range(1, 100)]
        # [self.plot_figure_density(traj_rrt, i, "RRT*") for i in range(1, 100)]
        # self.plot_figure_density(traj_rrt, "RRT*")
        pass

    def plot_cost_components(self) -> None:
        eibv, ivr = self.grf.get_ei_field()

        # self.plotf_vector(self.grid_wgs[:, 0], self.grid_wgs[:, 1], eibv, alpha=1., cmap=get_cmap("RdBu", 25),
        #                   title="EIBV", vmin=0, vmax=1.1, stepsize=.1, colorbar=True, cbar_title="Cost",
        #                   polygon_border=self.polygon_border_wgs, polygon_obstacle=self.polygon_obstacle_wgs)

        fig = plt.figure(figsize=(20, 8))
        gs = GridSpec(nrows=1, ncols=2)
        ax = fig.add_subplot(gs[0])
        self.plotf_vector(self.grid_wgs[:, 0], self.grid_wgs[:, 1], eibv, alpha=1., cmap=get_cmap("RdBu", 25),
                          title="EIBV", vmin=0, vmax=1.1, stepsize=.1, colorbar=True, cbar_title="Cost",
                          polygon_border=self.polygon_border_wgs, polygon_obstacle=self.polygon_obstacle_wgs)
        plg = plt.Polygon(np.fliplr(self.polygon_obstacle_wgs), facecolor='w', edgecolor='r', fill=True,
                          linestyle='-.')
        plt.gca().add_patch(plg)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.yticks(self.lat_ticks)
        plt.xticks(self.lon_ticks)
        plt.xlim([self.lon_min, self.lon_max])
        plt.ylim([self.lat_min, self.lat_max])

        ax = fig.add_subplot(gs[1])
        self.plotf_vector(self.grid_wgs[:, 0], self.grid_wgs[:, 1], ivr, alpha=1., cmap=get_cmap("RdBu", 25),
                          title="EIBV", vmin=0, vmax=1.1, stepsize=.1, colorbar=True, cbar_title="Cost",
                          polygon_border=self.polygon_border_wgs, polygon_obstacle=self.polygon_obstacle_wgs)

        plg = plt.Polygon(np.fliplr(self.polygon_obstacle_wgs), facecolor='w', edgecolor='r', fill=True,
                          linestyle='-.')
        plt.gca().add_patch(plg)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.yticks(self.lat_ticks)
        plt.xticks(self.lon_ticks)
        plt.xlim([self.lon_min, self.lon_max])
        plt.ylim([self.lat_min, self.lat_max])
        plt.savefig(figpath + "Simulation/info_fields.png")
        plt.close("all")

        fig = plt.figure(figsize=(20, 8))
        gs = GridSpec(nrows=1, ncols=2)
        ax = fig.add_subplot(gs[0])
        self.plotf_vector(self.grid_wgs[:, 0], self.grid_wgs[:, 1], np.zeros_like(ivr), alpha=1.,
                          cmap=get_cmap("RdBu", 25), title="Obstacle", vmin=0, vmax=1.1, stepsize=.5, colorbar=True,
                          cbar_title="Cost", polygon_border=self.polygon_border_wgs,
                          polygon_obstacle=self.polygon_obstacle_wgs)
        plg = plt.Polygon(np.fliplr(self.polygon_obstacle_wgs), facecolor='w', edgecolor='r', fill=True,
                          linestyle='-.')
        plt.gca().add_patch(plg)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.yticks(self.lat_ticks)
        plt.xticks(self.lon_ticks)
        plt.xlim([self.lon_min, self.lon_max])
        plt.ylim([self.lat_min, self.lat_max])

        ax = fig.add_subplot(gs[1])
        self.plotf_vector(self.grid_wgs[:, 0], self.grid_wgs[:, 1], np.zeros_like(ivr), alpha=1.,
                          cmap=get_cmap("RdBu", 25),
                          title="Budget", vmin=0, vmax=1.1, stepsize=.5, colorbar=True, cbar_title="Cost",
                          polygon_border=self.polygon_border_wgs, polygon_obstacle=self.polygon_obstacle_wgs)
        plg = plt.Polygon(np.fliplr(self.polygon_obstacle_wgs), facecolor='w', edgecolor='r', fill=True,
                          linestyle='-.')
        plt.gca().add_patch(plg)
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.yticks(self.lat_ticks)
        plt.xticks(self.lon_ticks)
        plt.xlim([self.lon_min, self.lon_max])
        plt.ylim([self.lat_min, self.lat_max])
        # plt.scatter(self.grid_wgs[:, 1], self.grid_wgs[:, 0], c=eibv, cmap=get_cmap("BrBG", 10), vmin=0, vmax=1)
        # plt.colorbar()
        # plt.show()
        plt.savefig(figpath + "Simulation/op_fields.png")
        plt.close("all")
        # plt.show()

        mu_truth
        pass

    def is_masked(self, lat, lon) -> bool:
        p = Point(lat, lon)
        masked = False
        if not self.polygon_border_wgs_shapely.contains(p) or self.polygon_obstacle_wgs_shapely.contains(p):
            masked = True
        return masked

    def plotf_vector(self, lat, lon, values, title=None, alpha=None, cmap=get_cmap("BrBG", 10),
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
            ind_mask.append(self.is_masked(lat_triangulated[i], lon_triangulated[i]))
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
            # cbar.ax.set_title(cbar_title, rotation=270)
            cbar.ax.set_ylabel(cbar_title, rotation=270, labelpad=40)
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


if __name__ == "__main__":
    e = EDA()
    # e.plot_metric_analysis()
    # e.plot_cost_components()
    # e.plot_traffic_density_map()



