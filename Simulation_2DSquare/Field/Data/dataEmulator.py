"""
This script generates the data for the simulation study
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-24
"""

from GOOGLE.Simulation_2DSquare.Config.Config import *
from GOOGLE.Simulation_2DSquare.Tree.Location import *
from usr_func import *

filepath_grid = FILEPATH + "Field/Grid/Grid.csv"
grid = pd.read_csv(filepath_grid).to_numpy()
x = grid[:, 0]
y = grid[:, 1]


mu_prior = (.5 * (1 - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .07)) +
            .5 * (1 - np.exp(- ((x - .0) ** 2 + (y - .5) ** 2) / .07)))
    # 1 - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .05))
    # 1 - np.exp(- ((x - .5) ** 2 + (y - .0) ** 2) / .004) +
    # 1 - np.exp(- ((x - .99) ** 2 + (y - .1) ** 2) / .1))
plt.scatter(x, y, c=mu_prior, cmap=CMAP)
plt.colorbar()
plt.show()

x, y, mu_prior = map(vectorise, [x, y, mu_prior])

df = pd.DataFrame(np.hstack((x, y, mu_prior)), columns=['x', 'y', 'mu_prior'])
df.to_csv(FILEPATH + "Field/Data/mu_prior.csv", index=False)



