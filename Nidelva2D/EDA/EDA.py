"""
This script does simple EDA analysis
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-05-13
"""
from GOOGLE.Nidelva2D.grfar_model import GRFAR
from GOOGLE.Nidelva2D.Config.Config import FILEPATH, LATITUDE_ORIGIN, LONGITUDE_ORIGIN, DEPTH_LAYER
from usr_func import *
from DataHandler.SINMOD import SINMOD


DATAPATH = "/Users/yaolin/OneDrive - NTNU/MASCOT_PhD/Data/Nidelva/20220511/GOOGLE/"


class EDA:

    def __init__(self):
        self.load_auv_data()
        self.load_grfar_model()

    def load_auv_data(self):
        self.data_auv = pd.read_csv(DATAPATH + "data_sync.csv").to_numpy()
        self.timestamp_auv = self.data_auv[:, 0]
        self.lat_auv = self.data_auv[:, 1]
        self.lon_auv = self.data_auv[:, 2]
        self.depth_auv = self.data_auv[:, 3]
        self.salinity_auv = self.data_auv[:, 4]
        self.temperature_auv = self.data_auv[:, 5]
        print("AUV data is loaded successfully!")

    def load_sinmod_data(self):

        self.sinmod = SINMOD()
        self.sinmod.load_sinmod_data(raw_data=True)
        coordinates_auv = np.vstack((self.lat_auv, self.lon_auv, self.depth_auv)).T
        self.sinmod.get_data_at_coordinates(coordinates_auv)
        pass

    def load_grfar_model(self):
        self.grfar_model = GRFAR()
        self.grf_grid = self.grfar_model.grf_grid
        self.N_grf_grid = self.grf_grid.shape[0]
        print("S2: GRFAR model is loaded successfully!")

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

    def get_residual_with_sinmod(self):
        self.sinmod = SINMOD()
        self.sinmod.load_sinmod_data(raw_data=True)
        coordinates_auv = np.vstack((self.lat_auv, self.lon_auv, self.depth_auv)).T
        self.sinmod.get_data_at_coordinates(coordinates_auv)
        pass

if __name__ == "__main__":
    e = EDA()
    e.plot_2d()
    # e.get_residual_with_sinmod()
    # e.plot_scatter_data()









