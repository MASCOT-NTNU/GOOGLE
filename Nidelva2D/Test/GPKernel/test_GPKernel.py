import numpy as np

from usr_func import *
from GOOGLE.Simulation_2DNidelva.Config.Config import *
from GOOGLE.Simulation_2DNidelva.GPKernel.GPKernel import GPKernel
from GOOGLE.Simulation_2DNidelva.Tree.Knowledge import Knowledge

dataset = pd.read_csv(PATH_DATA).to_numpy()
coordinates = dataset[:, 0:3]
x, y = latlon2xy(coordinates[:, 0], coordinates[:, 1], LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
z = np.zeros_like(x)
x, y, z = map(vectorise, [x, y, z])
coordinates_xyz = np.hstack((x, y, z))

mu_prior = vectorise(dataset[:, -1])

knowledge = Knowledge(coordinates_xy=coordinates_xyz)
knowledge.mu_prior = mu_prior
t = GPKernel(knowledge=knowledge)


#%%
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
cmap = get_cmap("RdBu", 10)

# plt.scatter(t.coordinates[:, 1], t.coordinates[:, 0], c=t.mu_prior, cmap=cmap, vmin=10, vmax=32)
# plt.colorbar()
#
# plt.show()
t.get_obstacle_field()
# t.get_variance_reduction_field()
# t.get_eibv_field()

plt.scatter(t.coordinates_xyz[:, 1], t.coordinates_xyz[:, 0], c=t.knowledge.mu_truth, cmap=cmap, vmin=16, vmax=32)
plt.colorbar()
plt.show()
plt.scatter(t.coordinates_xyz[:, 1], t.coordinates_xyz[:, 0], c=t.knowledge.mu_prior, cmap=cmap, vmin=16, vmax=32)
plt.colorbar()
plt.show()
# t.get_cost_valley()
#%%
plt.imshow(t.knowledge.Sigma_cond, vmin=0, vmax=1)
plt.colorbar()
plt.show()
#%%
plt.scatter(coordinates_xyz[:, 1], coordinates_xyz[:, 0], c=mu_prior, cmap="RdBu", vmin=16, vmax=32)
plt.colorbar()
plt.show()
