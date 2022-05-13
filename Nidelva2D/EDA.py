"""
This script does simple EDA analysis
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-05-13
"""
import matplotlib.pyplot as plt

from GOOGLE.Nidelva2D.Config.Config import FILEPATH, LATITUDE_ORIGIN, LONGITUDE_ORIGIN
from usr_func import *


DATAPATH = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/20220511/GOOGLE/"


class EDA:

    def __init__(self):
        self.load_auv_data()
        pass

    def load_auv_data(self):
        self.data = pd.read_csv(DATAPATH+"data_sync.csv").to_numpy()
        self.lat = self.data[:, 0]
        self.lon = self.data[:, 1]
        self.depth = self.data[:, 2]
        self.salinity = self.data[:, 3]
        print("AUV data is loaded successfully!")

    def plot_scatter_data(self):
        fig = go.Figure(data=go.Scatter3d(
            x=self.data[:, 1],
            y=self.data[:, 0],
            z=-self.data[:, 2],
            mode='markers',
            marker=dict(color=self.data[:, 3], size=10)
        ))
        plotly.offline.plot(fig, filename=FILEPATH+"fig/EDA/samples.html", auto_open=True)
        pass

    def plot_2d(self):
        plt.scatter(self.lon, self.lat, c=self.salinity, cmap=get_cmap("BrBG", 10), vmin=4, vmax=30)
        plt.colorbar()
        plt.show()


if __name__ == "__main__":
    e = EDA()
    e.plot_2d()
    # e.plot_scatter_data()




