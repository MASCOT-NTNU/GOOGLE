
import plotly.graph_objects as go
import plotly
import os


class Scatter3DPlot:

    def __init__(self, coordinates, filename):
        self.coordinates = coordinates
        self.filename = filename
        self.plot()

    def plot(self):
        fig = go.Figure(data=[go.Scatter3d(
            x=self.coordinates[:, 1],
            y=self.coordinates[:, 0],
            z=-self.coordinates[:, 2],
            mode='markers+lines',
            marker=dict(
                size=12,
                color="black",
            ),
            line = dict(
                width=3,
                color="yellow",
            )
        )])
        plotly.offline.plot(fig,
                            filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/" + self.filename + ".html",
                            auto_open=False)
        os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/"+self.filename+".html")

