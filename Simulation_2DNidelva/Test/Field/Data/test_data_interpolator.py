import matplotlib.pyplot as plt
import pandas as pd

# from GOOGLE.Field.Grid.gridWithinPolygonGenerator import GridGenerator
# from GOOGLE.Field.Grid.HexagonalGrid2D import HexgonalGrid2DGenerator
from GOOGLE.Simulation_2DNidelva.Field.Grid.HexagonalGrid3D import HexgonalGrid3DGenerator
from DataHandler.SINMOD import SINMOD
from usr_func import *

PATH_OPERATION_AREA = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Config/OpArea.csv"
PATH_MUNKHOLMEN = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Config/Munkholmen.csv"
NEIGHBOUR_DISTANCE = 150
DEPTH = [0, 2, 5]

polygon = pd.read_csv(PATH_OPERATION_AREA).to_numpy()
munkholmen = pd.read_csv(PATH_MUNKHOLMEN).to_numpy()
gridGenerator = HexgonalGrid3DGenerator(polygon_within=polygon, polygon_without=munkholmen,
                                        depth=DEPTH, neighbour_distance=NEIGHBOUR_DISTANCE)
# gridGenerator = GridGenerator(polygon=polygon, depth=[0], distance_neighbour=DISTANCE_LATERAL, no_children=6, points_allowed=5000)
# grid = gridGenerator.grid
coordinates = gridGenerator.coordinates

sinmod = SINMOD()
sinmod.get_data_at_coordinates(coordinates)

# plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
# plt.plot(polygon[:, 1], polygon[:, 0], 'r-.')
# plt.plot(munkholmen[:, 1], munkholmen[:, 0], 'r-.')
# plt.show()

# import plotly.graph_objects as go
#%%
fig = go.Figure(data=[go.Scatter3d(
    x=sinmod.dataset_interpolated.iloc[:, 1],
    y=sinmod.dataset_interpolated.iloc[:, 0],
    z=-sinmod.dataset_interpolated.iloc[:, 2],
    mode='markers',
    marker=dict(
        size=12,
        color=sinmod.dataset_interpolated.iloc[:, 3],
        cmin=0,
        cmax=30,
        opacity=0.8
    )
)])

# # tight layout
# fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
plotly.offline.plot(fig, filename="/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/fig/data.html", auto_open=False)
os.system("open -a \"Google Chrome\" /Users/yaoling/OneDrive\ -\ NTNU/MASCOT_PhD/Projects/GOOGLE/fig/data.html")
# fig.show()

#%%
depth_layer = np.unique(sinmod.dataset_interpolated.iloc[:, 2])
for i in range(len(depth_layer)):

    ind_depth = np.where(sinmod.dataset_interpolated.iloc[:, 2] == depth_layer[i])[0]
    plt.figure()
    plt.scatter(sinmod.dataset_interpolated.iloc[ind_depth, 1], sinmod.dataset_interpolated.iloc[ind_depth, 0],
                c=sinmod.dataset_interpolated.iloc[ind_depth, 3], cmap="Paired", vmin=25, vmax=35)
    plt.colorbar()
    plt.title("depth_{:01d}".format(i))
plt.show()


