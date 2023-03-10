"""
This module mainly handles cost valley visualisation.
"""
from Field import Field
from WGS import WGS
from Config import Config
from usr_func.checkfolder import checkfolder
from scipy.interpolate import griddata
import numpy as np
import plotly
import plotly.graph_objects as go
from plotly.subplots import make_subplots


class ValleyPlotter:
    """
    ValleyPlotter mainly handles the interpolation and discretization for visualization purposes.
    It utilizes the surface plot from plotly module.
    """
    def __init__(self, grid: np.ndarray):
        self.grid = grid
        self.config = Config()
        self.polygon_border = self.config.get_polygon_border()
        self.polygon_obstacle = self.config.get_polygon_obstacle()

    def plot_trees_on_3d_valley(self, value, nodes, cv, traj, wp_now, wp_next, title: str,
                                foldername: str, resolution: float = 32., vmin: float = .0, vmax: float = 1.) -> None:
        checkfolder(foldername)
        """ s0, get more find grid discretization from the original algorithm. """
        field = Field(neighbour_distance=resolution)
        grid = field.get_grid()

        """ s1, interpolate to a bigger rectangular box. """
        v_int = griddata(self.grid, value, (grid[:, 0], grid[:, 1]), method="cubic")

        """
        s2, to plot them onto 3d surface plot in plotly, they must be in rectangular grid. Therefore, interpolate
        again on a more fine grid  
        """
        nx = 200
        ny = 200
        x = grid[:, 1]
        y = grid[:, 0]
        xmin, ymin = map(np.amin, [x, y])
        xmax, ymax = map(np.amax, [x, y])
        xv = np.linspace(xmin, xmax, nx)
        yv = np.linspace(ymin, ymax, ny)
        grid_x, grid_y = np.meshgrid(xv, yv)
        grid_value = griddata((x, y), v_int, (grid_x, grid_y), method="linear")
        """ Note, griddata method must be linear, or else all the values might be nan. """

        lat, lon = WGS.xy2latlon(yv, xv)

        """ s3, make plots. """
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])

        fig.update_coloraxes(colorscale="RdBu", cmin=vmin, cmax=vmax,
                             colorbar=dict(lenmode='fraction', len=.5, thickness=20,
                                           tickfont=dict(size=18, family="Times New Roman"),
                                           title="Cost",
                                           titlefont=dict(size=18, family="Times New Roman")),
                             colorbar_x=.75)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=.0, y=-1., z=1.3)
        )
        fig.update_layout(
            showlegend=True,
            title={
                'text': title,
                'y': 0.85,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=30, family="Times New Roman"),
            },
            scene=dict(
                # xaxis=dict(range=[np.amin(points_int[:, 0]), np.amax(points_int[:, 0])]),
                # yaxis=dict(range=[np.amin(points_int[:, 1]), np.amax(points_int[:, 1])]),
                zaxis=dict(nticks=4, range=[0, 1.], showticklabels=False),
                xaxis_tickfont=dict(size=14, family="Times New Roman"),
                yaxis_tickfont=dict(size=14, family="Times New Roman"),
                zaxis_tickfont=dict(size=14, family="Times New Roman"),
                xaxis_title=dict(text="Longitude", font=dict(size=18, family="Times New Roman")),
                yaxis_title=dict(text="Latitude", font=dict(size=18, family="Times New Roman")),
                zaxis_title=dict(text="", font=dict(size=18, family="Times New Roman")),
            ),
            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.25),
            scene_camera=camera,
        )

        fig.add_trace(go.Surface(
            x=lon,
            y=lat,
            z=grid_value,
            coloraxis="coloraxis",
            opacity=.5,
            contours={
                # "z": {"show": True, "start": .0, "end": 1., "size": 0.01}
            },
        ),
            row=1, col=1
        )

        lat, lon = WGS.xy2latlon(wp_now[0], wp_now[1])
        cp = cv.get_cost_at_location(wp_now)
        fig.add_trace(go.Scatter3d(
            x=[lon],
            y=[lat],
            z=[cp[0]],
            name="Current waypoint",
            mode="markers",
            marker=dict(
                size=20,
                color="black",
            ),
            showlegend=False,
        ),
            row=1, col=1
        )

        lat, lon = WGS.xy2latlon(wp_next[0], wp_next[1])
        cp = cv.get_cost_at_location(wp_next)
        fig.add_trace(go.Scatter3d(
            x=[lon],
            y=[lat],
            z=[cp[0]],
            name="Global minimum cost waypoint",
            mode="markers",
            marker=dict(
                size=20,
                color="yellow",
            ),
            showlegend=False,
        ),
            row=1, col=1
        )

        # x = self.polygon_border[:, 0]
        # y = self.polygon_border[:, 1]
        # lat, lon = WGS.xy2latlon(x, y)
        # fig.add_trace(go.Scatter3d(
        #     x=lon,
        #     y=lat,
        #     z=np.ones_like(lat) * .9,
        #     mode="lines",
        #     line=dict(
        #         width=100,
        #         color="black",
        #     )),
        #     row=1, col=1
        # )
        # TODO: add vertical walls

        counter = 0
        for node in nodes:
            print("counter: ", counter)
            # print(node)
            p_node = node.get_parent()
            l_node = node.get_location()
            if not p_node:
                continue
            else:
                lp_node = p_node.get_location()
                c_node = cv.get_cost_at_location(l_node)
                cp_node = cv.get_cost_at_location(lp_node)
                lat, lon = WGS.xy2latlon(l_node[0], l_node[1])
                lat_p, lon_p = WGS.xy2latlon(lp_node[0], lp_node[1])
                x = np.stack((lon, lon_p))
                y = np.stack((lat, lat_p))
                z = [c_node[0], cp_node[0]]
                fig.add_trace(go.Scatter3d(
                    # x=[l_node[1], lp_node[1]],
                    # y=[l_node[0], lp_node[0]],
                    # z=[c_node[0], cp_node[0]],
                    x=x,
                    y=y,
                    z=z,
                    mode="lines",
                    line=dict(
                        # width=10,
                        color="green",
                    ),
                    showlegend=False,
                ),
                    row=1, col=1
                )
                # plotly.offline.plot(fig, filename=filename, auto_open=True)
                fig.write_image(foldername + "P_{:03d}.png".format(counter), width=1980, height=1080)
                counter += 1

        lat, lon = WGS.xy2latlon(traj[:, 0], traj[:, 1])
        c_traj = []
        for k in range(len(traj)):
            c_traj.append(cv.get_cost_at_location(traj[k, :])[0])
        fig.add_trace(go.Scatter3d(
            x=lon,
            y=lat,
            z=c_traj,
            name="Trajectory",
            mode="lines",
            line=dict(
                width=10,
                color="blue",
            ),
            showlegend=False,
        ),
            row=1, col=1
        )

        plotly.offline.plot(fig, filename=foldername + "P_{:03d}.html".format(counter), auto_open=True)
        # fig.write_image(filename, width=1980, height=1080)
        pass


    def plot_3d_valley(self, value, title: str,
                       filename: str, resolution: float = 32., vmin: float = .0, vmax: float = 1.):
        """ s0, get more find grid discretization from the original algorithm. """
        field = Field(neighbour_distance=resolution)
        grid = field.get_grid()

        """ s1, interpolate to a bigger rectangular box. """
        v_int = griddata(self.grid, value, (grid[:, 0], grid[:, 1]), method="cubic")

        """
        s2, to plot them onto 3d surface plot in plotly, they must be in rectangular grid. Therefore, interpolate
        again on a more fine grid  
        """
        nx = 200
        ny = 200
        x = grid[:, 1]
        y = grid[:, 0]
        xmin, ymin = map(np.amin, [x, y])
        xmax, ymax = map(np.amax, [x, y])
        xv = np.linspace(xmin, xmax, nx)
        yv = np.linspace(ymin, ymax, ny)
        grid_x, grid_y = np.meshgrid(xv, yv)
        grid_value = griddata((x, y), v_int, (grid_x, grid_y), method="linear")
        """ Note, griddata method must be linear, or else all the values might be nan. """

        lat, lon = WGS.xy2latlon(yv, xv)

        """ s3, make plots. """
        fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
        fig.add_trace(go.Surface(
            x=lon,
            y=lat,
            z=grid_value,
            coloraxis="coloraxis",
            contours={
                "z": {"show": True, "start": .0, "end": 1., "size": 0.01}
            },
        ),
            row=1, col=1
        )

        # x = self.polygon_border[:, 0]
        # y = self.polygon_border[:, 1]
        # lat, lon = WGS.xy2latlon(x, y)
        # fig.add_trace(go.Scatter3d(
        #     x=lon,
        #     y=lat,
        #     z=np.ones_like(lat) * .9,
        #     mode="lines",
        #     line=dict(
        #         width=100,
        #         color="black",
        #     )),
        #     row=1, col=1
        # )
        # TODO: add vertical walls

        fig.update_coloraxes(colorscale="RdBu", cmin=vmin, cmax=vmax,
                             colorbar=dict(lenmode='fraction', len=.5, thickness=20,
                                                              tickfont=dict(size=20, family="Times New Roman"),
                                                              title="Cost",
                                                              titlefont=dict(size=30, family="Times New Roman")),
                             colorbar_x=.7)
        camera = dict(
            up=dict(x=0, y=0, z=1),
            center=dict(x=0, y=0, z=0),
            eye=dict(x=.0, y=-1., z=1.3)
        )
        fig.update_layout(
            title={
                'text': title,
                'y': 0.85,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': dict(size=40, family="Times New Roman"),
            },
            scene=dict(
                # xaxis=dict(range=[np.amin(points_int[:, 0]), np.amax(points_int[:, 0])]),
                # yaxis=dict(range=[np.amin(points_int[:, 1]), np.amax(points_int[:, 1])]),
                zaxis=dict(nticks=4, range=[0, 1.], showticklabels=False),
                xaxis_tickfont=dict(size=20, family="Times New Roman"),
                yaxis_tickfont=dict(size=20, family="Times New Roman"),
                zaxis_tickfont=dict(size=20, family="Times New Roman"),
                xaxis_title=dict(text="", font=dict(size=30, family="Times New Roman")),
                yaxis_title=dict(text="", font=dict(size=30, family="Times New Roman")),
                # xaxis_title=dict(text="Longitude", font=dict(size=30, family="Times New Roman")),
                # yaxis_title=dict(text="Latitude", font=dict(size=30, family="Times New Roman")),
                zaxis_title=dict(text="", font=dict(size=30, family="Times New Roman")),
            ),

            scene_aspectmode='manual',
            scene_aspectratio=dict(x=1, y=1, z=.25),
            scene_camera=camera,
        )
        fig.update_yaxes(title_standoff=100)
        fig.update_xaxes(title_standoff=100)

        plotly.offline.plot(fig, filename=filename, auto_open=True)
        # fig.write_image(filename, width=1980, height=1080)


# #%%
# import plotly
# import plotly.graph_objects as go
# from plotly.subplots import make_subplots
#
# fig = make_subplots(rows=1, cols=1, specs=[[{'type': 'scene'}]])
# fig.add_trace(go.Scatter3d(
#     x=[0, 1],
#     y=[0, 1],
#     z=[0, 1],
#     mode="lines + markers",
#     marker=dict(
#         size=20,
#         color="black"
#     ),
#     line=dict(
#         width=10,
#         color="green",
#     )),
#     row=1, col=1
# )
# plotly.offline.plot(fig, filename="/Users/yaolin/Downloads/fig/test.html", auto_open=True)



