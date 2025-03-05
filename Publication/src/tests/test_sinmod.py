""" Unit test for SINMOD data handler

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-08-24
"""
import os
from unittest import TestCase
from SINMOD import SINMOD
from WGS import WGS
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from matplotlib.gridspec import GridSpec
import matplotlib.tri as tri
from shapely.geometry import Polygon, Point


class TestSINMOD(TestCase):
    """

    """
    def setUp(self) -> None:
        sinmod_path = os.getcwd() + "/../sinmod/samples_2022.05.11.nc"
        self.sinmod = SINMOD(sinmod_path)

    def test_get_data_from_sinmod(self) -> None:
        # c1: one depth layer
        N = 100
        lat = np.linspace(63.438381, 63.453735, N)
        lon = np.linspace(10.359198, 10.425457, N)
        # depth = np.random.uniform(0.5, 5.5, 5)
        depth = [0.5]
        coord = []
        for i in range(len(lat)):
            for j in range(len(lon)):
                for k in range(len(depth)):
                    coord.append([lat[i], lon[j], depth[k]])
        coord = np.array(coord)
        x, y = WGS.latlon2xy(coord[:, 0], coord[:, 1])
        locations = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), coord[:, 2].reshape(-1, 1)))
        df = self.sinmod.get_data_at_locations(locations)
        plt.scatter(df[:, 1], df[:, 0], c=df[:, -1],
                    cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
        plt.show()

        # c2: many depth layers
        N = 100
        lat = np.linspace(63.438381, 63.453735, N)
        lon = np.linspace(10.359198, 10.425457, N)
        depth = np.linspace(0.5, 5.5, 3)
        coord = []
        for i in range(len(lat)):
            for j in range(len(lon)):
                for k in range(len(depth)):
                    coord.append([lat[i], lon[j], depth[k]])
        coord = np.array(coord)
        x, y = WGS.latlon2xy(coord[:, 0], coord[:, 1])
        locations = np.hstack((x.reshape(-1, 1), y.reshape(-1, 1), coord[:, 2].reshape(-1, 1)))
        df = self.sinmod.get_data_at_locations(locations)
        fig = plt.figure(figsize=(len(depth) * 8, 8))
        gs = GridSpec(ncols=len(depth), nrows=1, figure=fig)
        for i in range(len(depth)):
            ind = np.where(coord[:, 2] == depth[i])[0]
            fig.add_subplot(gs[i])
            plt.scatter(df[ind, 1], df[ind, 0], c=df[ind, -1],
                        cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
            plt.colorbar()
        plt.show()

    def test_get_data_for_gmrf_grid(self) -> None:
        from GRF.GRF import GRF
        grf = GRF()
        grid = grf.grid
        grid = np.hstack((grid, 1.5 * np.ones((grid.shape[0], 1))))
        df_sinmod = self.sinmod.get_data_at_locations(grid)

        import plotly
        import plotly.graph_objs as go
        from plotly.subplots import make_subplots

        # make 3D scatter plot
        fig = make_subplots(rows=1, cols=1,
                            specs=[[{'type': 'scatter3d'}]])

        fig.add_trace(go.Scatter3d(
            x=df_sinmod[:, 1],
            y=df_sinmod[:, 0],
            z=-df_sinmod[:, 2],
            mode='markers',
            marker=dict(
                size=10,
                color=df_sinmod[:, -1],
                cmin=0,
                cmax=35,
                colorscale='Viridis',
                # opacity=0.8
                showscale=True
            )
        ), row=1, col=1)

        fig.update_layout(
            scene=dict(
                xaxis=dict(title='Longitude'),
                yaxis=dict(title='Latitude'),
                zaxis=dict(title='Depth'),
            ),
            title='SINMOD salinity data',
            autosize=True,
            width=1000,
            height=1000,
            margin=dict(l=65, r=50, b=65, t=90)
        )

        plotly.offline.plot(fig, filename='/Users/yaolin/Downloads/sinmod.html', auto_open=True)
