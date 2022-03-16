

from GOOGLE.Simulation_2DNidelva.GPKernel.GPKernel import GPKernel

t = GPKernel()


#%%
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
cmap = get_cmap("RdBu", 10)

# plt.scatter(t.coordinates[:, 1], t.coordinates[:, 0], c=t.mu_prior, cmap=cmap, vmin=10, vmax=32)
# plt.colorbar()
#
# plt.show()
t.get_obstacle_field()
t.get_variance_reduction_field()
t.get_eibv_field()

plt.scatter(t.coordinates[:, 1], t.coordinates[:, 0], c=t.cost_eibv, cmap=cmap)
plt.colorbar()
plt.show()
# t.get_cost_valley()
# plt.imshow(t.Sigma_prior, vmin=0, vmax=4)
# plt.colorbar()

