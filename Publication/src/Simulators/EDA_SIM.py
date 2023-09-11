"""
EDA for the simulation study.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-09-05
"""

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
import pandas as pd
from scipy.stats import gaussian_kde
from matplotlib.cm import get_cmap
from sklearn.neighbors import KernelDensity
from time import time
from matplotlib import tri
from shapely.geometry import Polygon, Point


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 20

folderpath = "./npy/temporal/"
figpath = os.getcwd() + ("/../../../../OneDrive - NTNU/MASCOT_PhD/"
                         "Projects/GOOGLE/Docs/fig/Sim_2DNidelva/Simulator/temporal")


class EDA:

    def __init__(self) -> None:
        self.config = Config()
        self.grf = GRF()
        self.grid = self.grf.grid
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

        replicates = os.listdir(folderpath)
        self.num_replicates = 0
        for rep in replicates:
            if rep.startswith("R_"):
                self.num_replicates += 1
        print("Number of replicates: ", self.num_replicates)

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

        self.load_data()
        self.plot_ground_truth()

        self.trajectory

    def load_data(self) -> None:
        t0 = time()
        fpath = os.getcwd() + "/../simulation_result/temporal/"
        self.trajectory = load(fpath + "trajectory.joblib")
        self.ibv = load(fpath + "ibv.joblib")
        self.rmse = load(fpath + "rmse.joblib")
        self.vr = load(fpath + "vr.joblib")
        self.mu = load(fpath + "mu.joblib")
        self.cov = load(fpath + "cov.joblib")
        self.sigma = load(fpath + "sigma.joblib")
        self.truth = load(fpath + "truth.joblib")
        print("Loading data takes {:.2f} seconds.".format(time() - t0))

    def plot_ground_truth(self) -> None:
        """
        Plot the ground truth for one replicate to check.
        """
        num_replicate = 0
        fpath = figpath + "/R_{:03d}/".format(num_replicate)
        checkfolder(fpath)
        for i in range(1, self.num_steps):
            plt.figure(figsize=(20, 5))
            gs = GridSpec(nrows=1, ncols=3)
            ax = plt.subplot(gs[0, 0])
            plt.scatter(self.grid_wgs[:, 1], self.grid_wgs[:, 0], c=self.truth["myopic"]["eibv"][num_replicate, i, :],
                        cmap=get_cmap("BrBG", 10), vmin=10, vmax=30, alpha=.5)
            plt.colorbar()
            traj = self.trajectory["myopic"]["eibv"][num_replicate, :i, :]
            lat, lon = WGS.xy2latlon(traj[:, 0], traj[:, 1])
            plt.plot(lon, lat, 'k.-')
            plt.plot(self.polygon_border_wgs[:, 1], self.polygon_border_wgs[:, 0], 'r-.')
            plg = plt.Polygon(np.fliplr(self.polygon_obstacle_wgs), facecolor='w', edgecolor='r', fill=True,
                                linestyle='-.')
            plt.gca().add_patch(plg)
            plt.xlabel("Longitude")
            plt.xticks(self.lon_ticks)
            plt.xlim([self.lon_min, self.lon_max])
            plt.ylim([self.lat_min, self.lat_max])
            plt.title("Myopic EIBV")

            ax = plt.subplot(gs[0, 1])
            plt.scatter(self.grid_wgs[:, 1], self.grid_wgs[:, 0], c=self.truth["rrt"]["eibv"][num_replicate, i, :],
                        cmap=get_cmap("BrBG", 10), vmin=10, vmax=30, alpha=.5)
            plt.colorbar()
            traj = self.trajectory["rrt"]["eibv"][num_replicate, :i, :]
            lat, lon = WGS.xy2latlon(traj[:, 0], traj[:, 1])
            plt.plot(lon, lat, 'k.-')
            plt.plot(self.polygon_border_wgs[:, 1], self.polygon_border_wgs[:, 0], 'r-.')
            plg = plt.Polygon(np.fliplr(self.polygon_obstacle_wgs), facecolor='w', edgecolor='r', fill=True,
                                linestyle='-.')
            plt.gca().add_patch(plg)
            plt.xlabel("Longitude")
            plt.xticks(self.lon_ticks)
            plt.xlim([self.lon_min, self.lon_max])
            plt.ylim([self.lat_min, self.lat_max])
            plt.title("RRT EIBV")

            ax = plt.subplot(gs[0, 2])
            plt.scatter(self.grid_wgs[:, 1], self.grid_wgs[:, 0], c=self.truth["myopic"]["eibv"][num_replicate, i, :] -
                        self.truth["rrt"]["eibv"][num_replicate, i, :], cmap=get_cmap("RdBu", 10), vmin=-1, vmax=1)
            plt.colorbar()
            plt.plot(self.polygon_border_wgs[:, 1], self.polygon_border_wgs[:, 0], 'r-.')
            plt.xlabel("Longitude")
            plt.xticks(self.lon_ticks)
            plt.xlim([self.lon_min, self.lon_max])
            plt.ylim([self.lat_min, self.lat_max])
            plt.title("Difference")

            plt.tight_layout()
            plt.savefig(fpath + "P_{:03d}.png".format(i))
            plt.close("all")

    def load_raw_data_from_replicate_files(self) -> 'dict':
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

        self.cv = ['eibv', 'ivr', 'equal']
        self.planners = ['myopic', 'rrt']
        self.metrics = ['traj', 'ibv', 'rmse', 'vr', 'mu', 'cov', 'sigma', 'truth']

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
        return dataset

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

        fpath = os.getcwd() + "/../simulation_result/temporal/"
        dump(self.trajectory, fpath + "trajectory.joblib")
        dump(self.ibv, fpath + "ibv.joblib")
        dump(self.rmse, fpath + "rmse.joblib")
        dump(self.vr, fpath + "vr.joblib")
        dump(self.mu, fpath + "mu.joblib")
        dump(self.cov, fpath + "cov.joblib")
        dump(self.sigma, fpath + "sigma.joblib")
        dump(self.truth, fpath + "truth.joblib")
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
        #

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

            cbar = plt.colorbar(contourplot, ax=ax, ticks=ticks)
            cbar.ax.set_title(cbar_title)
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



