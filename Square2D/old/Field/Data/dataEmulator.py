# """
# This script generates the data for the simulation study
# Author: Yaolin Ge
# Contact: yaolin.ge@ntnu.no
# Date: 2022-06-17
# """
# import os
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from TAICHI.Square2D.Config.Config import FILEPATH, CMAP
# from usr_func import vectorise
#
# filepath_grid = FILEPATH + "Config/GRFGrid.csv"
# grid = pd.read_csv(filepath_grid).to_numpy()
# x = grid[:, 0]
# y = grid[:, 1]
#
#
# mu_prior = (.5 * (1 - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .07)) +
#             .5 * (1 - np.exp(- ((x - .0) ** 2 + (y - .5) ** 2) / .07)))
#     # 1 - np.exp(- ((x - 1.) ** 2 + (y - .5) ** 2) / .05))
#     # 1 - np.exp(- ((x - .5) ** 2 + (y - .0) ** 2) / .004) +
#     # 1 - np.exp(- ((x - .99) ** 2 + (y - .1) ** 2) / .1))
# plt.scatter(x, y, c=mu_prior, cmap=CMAP)
# plt.colorbar()
# plt.show()
#
# from usr_func import interpolate_2d
# grid_x, grid_y, values = interpolate_2d(x, y, 100, 100, mu_prior, "cubic")
# plt.scatter(grid_x, grid_y, c=values, cmap=CMAP)
# plt.colorbar()
# plt.show()
#
# x, y, mu_prior = map(vectorise, [x, y, mu_prior])
#
# df = pd.DataFrame(np.hstack((x, y, mu_prior)), columns=['x', 'y', 'value'])
# df.to_csv(FILEPATH + "Config/mu_prior.csv", index=False)
# os.system("say data is saved sucessfully!")
#
#



