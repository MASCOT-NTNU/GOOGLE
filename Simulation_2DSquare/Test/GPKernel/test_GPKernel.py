import matplotlib.pyplot as plt

from GOOGLE.Simulation_2DSquare.GPKernel.GPKernel import *
from GOOGLE.Simulation_2DSquare.Tree.Knowledge import Knowledge


filepath_grid = FILEPATH + "Field/Grid/Grid.csv"
grid = pd.read_csv(filepath_grid).to_numpy()
x_vector = vectorise(grid[:, 0])
y_vector = vectorise(grid[:, 1])
filepath_mu_prior = FILEPATH + "Field/Data/mu_prior.csv"
mu_prior = vectorise(pd.read_csv(filepath_mu_prior)['mu_prior'].to_numpy())
polygon_border = np.array(BORDER)
polygon_obstacles = np.array(OBSTACLES)

knowledge = Knowledge(grid=grid, polygon_border=polygon_border, polygon_obstacles=polygon_obstacles, threshold=THRESHOLD)
knowledge.mu_prior = mu_prior

current_loc = Location(0, 0)
previous_loc = Location(1, 0)
goal_loc = Location(1, 1)
budget = BUDGET
gp = GPKernel(knowledge)
gp.get_cost_valley(current_loc=current_loc, previous_loc=previous_loc, goal_loc=goal_loc, budget=BUDGET)
plt.scatter(x_vector, y_vector, c=gp.cost_direction, cmap=CMAP)
plt.colorbar()
plt.show()

#%%
plt.imshow(gp.knowledge.Sigma_prior)
plt.colorbar()
plt.show()



