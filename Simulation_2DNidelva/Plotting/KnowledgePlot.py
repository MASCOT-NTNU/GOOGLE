"""
This script plots the knowledge
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-18
"""


from GOOGLE.Simulation_2DNidelva.Plotting.plotting_func import plotf_vector


class KnowledgePlot:

    def __init__(self, knowledge=None, vmin=28, vmax=30, filename="mean"):
        if knowledge is None:
            raise ValueError("")
        self.knowledge = knowledge
        self.coordinates = self.knowledge.coordinates
        self.vmin = vmin
        self.vmax = vmax
        self.filename = filename
        self.plot()

    def plot(self):

        trajectory = np.array(self.knowledge.trajectory)


        plotly.offline.plot(fig, filename = self.filename+".html", auto_open = False)
            # os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/"+self.filename+".html")
        # fig.write_image(self.filename+".png", width=1980, height=1080, engine = "orca")


