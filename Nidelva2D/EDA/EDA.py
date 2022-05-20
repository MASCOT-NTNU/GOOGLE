"""
This script does simple EDA analysis
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-05-13
"""
import matplotlib.pyplot as plt

from GOOGLE.Nidelva2D.EDA.grfar_model import GRFAR
from GOOGLE.Nidelva2D.Config.Config import FILEPATH, LATITUDE_ORIGIN, LONGITUDE_ORIGIN, DEPTH_LAYER
from usr_func import *
from DataHandler.SINMOD import SINMOD
from GOOGLE.Nidelva2D.EDA.plotting_func import plotf_vector
from GOOGLE.Nidelva2D.EDA.RRTStarCV import RRTStarCV


DATAPATH = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/20220511/GOOGLE/"
SINMODPATH = "/Users/yaolin/HomeOffice/GOOGLE/Experiments/20220511/"
figpath = "/Users/yaolin/HomeOffice/GOOGLE/Experiments/20220511/fig/"


class EDA:

    def __init__(self):
        self.load_auv_data()
        self.load_grfar_model()
        self.load_rrtstar()

    def load_auv_data(self):
        self.data_auv = pd.read_csv(DATAPATH + "data_sync.csv").to_numpy()
        self.timestamp_auv = self.data_auv[:, 0]
        self.lat_auv = self.data_auv[:, 1]
        self.lon_auv = self.data_auv[:, 2]
        self.depth_auv = self.data_auv[:, 3]
        self.salinity_auv = self.data_auv[:, 4]
        self.temperature_auv = self.data_auv[:, 5]
        print("AUV data is loaded successfully!")

    def load_sinmod_data(self, data_exists=True):
        if not data_exists:
            self.sinmod = SINMOD()
            self.sinmod.load_sinmod_data(raw_data=True)
            coordinates_auv = np.vstack((self.lat_auv, self.lon_auv, self.depth_auv)).T
            self.sinmod.get_data_at_coordinates(coordinates_auv)
        else:
            self.data_sinmod = pd.read_csv(SINMODPATH+"data_sinmod.csv")
            print("SINMOD data is loaded successfully!")
            print(self.data_sinmod.head())
        pass

    def load_grfar_model(self):
        self.grfar_model = GRFAR()
        self.grf_grid = self.grfar_model.grf_grid
        self.N_grf_grid = self.grf_grid.shape[0]
        print("S2: GRFAR model is loaded successfully!")

    def load_rrtstar(self):
        self.rrtstar = RRTStarCV()
        print("S3: RRTStar is loaded successfully!")

    def assimilate_data(self, dataset):
        print("dataset before filtering: ", dataset[:10, :])
        depth_dataset = np.abs(dataset[:, 2])
        ind_selected_depth_layer = np.where((depth_dataset >= .25) * (depth_dataset <= DEPTH_LAYER + .5))[0]
        dataset = dataset[ind_selected_depth_layer, :]
        print("dataset after filtering: ", dataset[:10, :])
        t1 = time.time()
        dx = (vectorise(dataset[:, 0]) @ np.ones([1, self.N_grf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.grf_grid[:, 0]).T) ** 2
        dy = (vectorise(dataset[:, 1]) @ np.ones([1, self.N_grf_grid]) -
              np.ones([dataset.shape[0], 1]) @ vectorise(self.grf_grid[:, 1]).T) ** 2
        dist = dx + dy
        ind_min_distance = np.argmin(dist, axis=1)
        t2 = time.time()
        ind_assimilated = np.unique(ind_min_distance)
        salinity_assimilated = np.zeros(len(ind_assimilated))
        for i in range(len(ind_assimilated)):
            ind_selected = np.where(ind_min_distance == ind_assimilated[i])[0]
            salinity_assimilated[i] = np.mean(dataset[ind_selected, 3])
        print("Data assimilation takes: ", t2 - t1)
        self.auv_data = []
        print("Reset auv_data: ", self.auv_data)
        return vectorise(ind_assimilated), vectorise(salinity_assimilated)

    def plot_scatter_data(self):
        fig = go.Figure(data=go.Scatter3d(
            x=self.lon_auv,
            y=self.lat_auv,
            z=-self.depth_auv,
            mode='markers',
            marker=dict(color=self.data_auv[:, 3], size=10)
        ))
        plotly.offline.plot(fig, filename=FILEPATH+"fig/EDA/samples.html", auto_open=True)
        pass

    def plot_2d(self):
        plt.scatter(self.lon_auv, self.lat_auv, c=self.salinity_auv, cmap=get_cmap("BrBG", 10), vmin=22, vmax=26.8)
        plt.colorbar()
        plt.show()

    def plot_sinmod_on_grf_grid(self):
        lat_grid, lon_grid = xy2latlon(self.grf_grid[:, 0], self.grf_grid[:, 1], LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        lat_p, lon_p = xy2latlon(self.rrtstar.polygon_border[:, 0], self.rrtstar.polygon_border[:, 1], LATITUDE_ORIGIN,
                                 LONGITUDE_ORIGIN)
        polygon_border = np.vstack((lat_p, lon_p)).T

        lat_o, lon_o = xy2latlon(self.rrtstar.polygon_obstacle[:, 0], self.rrtstar.polygon_obstacle[:, 1], LATITUDE_ORIGIN,
                                 LONGITUDE_ORIGIN)
        polygon_obstacle = np.vstack((lat_o, lon_o)).T

        plt.figure(figsize=(12, 10))
        plotf_vector(lon=lon_grid, lat=lat_grid, values=self.grfar_model.mu_sinmod,
                     title="SINMOD on grid on 20220511",
                     cmap=get_cmap("BrBG", 10), cbar_title="Salinity", vmin=2, vmax=32, stepsize=1, threshold=26.8,
                     polygon_border=polygon_border, polygon_obstacle=polygon_obstacle, xlabel="Lon [deg]", ylabel="Lat [deg]")
        plt.savefig(figpath + "sinmod_grid.jpg")
        plt.show()

    def plot_prior(self):
        lat_grid, lon_grid = xy2latlon(self.grf_grid[:, 0], self.grf_grid[:, 1], LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        lat_p, lon_p = xy2latlon(self.rrtstar.polygon_border[:, 0], self.rrtstar.polygon_border[:, 1], LATITUDE_ORIGIN,
                                 LONGITUDE_ORIGIN)
        polygon_border = np.vstack((lat_p, lon_p)).T

        lat_o, lon_o = xy2latlon(self.rrtstar.polygon_obstacle[:, 0], self.rrtstar.polygon_obstacle[:, 1],
                                 LATITUDE_ORIGIN,
                                 LONGITUDE_ORIGIN)
        polygon_obstacle = np.vstack((lat_o, lon_o)).T

        plt.figure(figsize=(12, 10))
        plotf_vector(lon=lon_grid, lat=lat_grid, values=self.grfar_model.mu_prior,
                     title="SINMOD on grid on 20220511",
                     cmap=get_cmap("BrBG", 10), cbar_title="Salinity", vmin=2, vmax=32, stepsize=1, threshold=26.8,
                     polygon_border=polygon_border, polygon_obstacle=polygon_obstacle, xlabel="Lon [deg]",
                     ylabel="Lat [deg]")
        plt.savefig(figpath + "prior.jpg")
        plt.show()

if __name__ == "__main__":
    e = EDA()
    e.load_sinmod_data(data_exists=True)
    e.plot_prior()
    # e.plot_sinmod_on_grf_grid()
    # e.plot_2d()
    # e.get_residual_with_sinmod()
    # e.plot_scatter_data()

#%%
plt.scatter(e.grf_grid[:, 1], e.grf_grid[:, 0], c=e.grfar_model.mu_prior, cmap=get_cmap("BrBG", 10), vmin=10, vmax=27)
plt.colorbar()
plt.show()

#%%

import os
import numpy as np
import matplotlib.pyplot as plt
import torch
import plotly
import netCDF4
from datetime import datetime
from matplotlib.cm import get_cmap
import re

LATITUDE_ORIGIN = 63.4269097
LONGITUDE_ORIGIN = 10.3969375
CIRCUMFERENCE = 40075000 # [m], circumference
circumference = CIRCUMFERENCE
def latlon2xy(lat, lon, lat_origin, lon_origin):
    x = np.deg2rad((lat - lat_origin)) / 2 / np.pi * CIRCUMFERENCE
    y = np.deg2rad((lon - lon_origin)) / 2 / np.pi * CIRCUMFERENCE * np.cos(np.deg2rad(lat))
    return x, y

file = "/Users/yaolin/Library/CloudStorage/OneDrive-NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/samples_2022.05.11.nc"
figpath = "/Users/yaolin/HomeOffice/GOOGLE/Experiments/20220511/fig/"
# file = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/SINMOD_DATA/samples_2022.05.10.nc"
# figpath = "/Users/yaolin/HomeOffice/GOOGLE/Experiments/20220509/fig/"

sinmod = netCDF4.Dataset(file)
ind_before = re.search("samples_", file)
ind_after = re.search(".nc", file)
date_string = file[ind_before.end():ind_after.start()]
ref_timestamp = datetime.strptime(date_string, "%Y.%m.%d").timestamp()
timestamp = np.array(sinmod["time"]) * 24 * 3600 + ref_timestamp #change ref timestamp
lat_sinmod = np.array(sinmod['gridLats'])
lon_sinmod = np.array(sinmod['gridLons'])
depth_sinmod = np.array(sinmod['zc'])
salinity_sinmod = np.array(sinmod['salinity'])

# for i in range(salinity_sinmod.shape[0]):
#     print(i)
#     plt.figure(figsize=(12, 10))
#     plt.scatter(lon_sinmod, lat_sinmod, c=salinity_sinmod[i, 0, :, :], cmap=get_cmap("BrBG", 8), vmin=10, vmax=26.8)
#     plt.xlabel("Lon [deg]")
#     plt.ylabel("Lat [deg]")
#     plt.title("SINMOD Surface Salinity Estimation on " + datetime.fromtimestamp(timestamp[i]).strftime("%H:%M, %Y-%m-%d"))
#     plt.colorbar()
#     plt.savefig(figpath+"sinmod/P_{:03d}.jpg".format(i))
#     plt.close("all")

#%%
lat_grid, lon_grid = xy2latlon(e.grf_grid[:, 0], e.grf_grid[:, 1], LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
sal_mean = np.mean(salinity_sinmod[:, 0, :, :], axis=0)
plt.figure(figsize=(12, 10))
# plt.scatter(lon_sinmod, lat_sinmod, c=sal_mean, cmap=get_cmap("BrBG", 8), vmin=10, vmax=26.8)
plt.xlabel("Lon [deg]")
plt.ylabel("Lat [deg]")

plt.scatter(lon_grid, lat_grid, c=e.grfar_model.mu_sinmod, cmap=get_cmap("BrBG", 8), vmin=10, vmax=26.8)
# plt.plot(lon_grid, lat_grid, 'y.', alpha=.3)
plt.colorbar()
plt.title("SINMOD" + datetime.fromtimestamp(timestamp[0]).strftime("%Y-%m-%d"))
plt.savefig(figpath+"sinmod_grid.jpg")
plt.show()


#



