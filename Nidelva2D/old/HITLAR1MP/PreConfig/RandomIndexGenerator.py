"""
This script generates the potential random locations within a certain constraint
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-23
"""


from GOOGLE.Nidelva2D.Config.Config import FILEPATH
import numpy as np
N = int(1e6)
ind = np.random.rand(N)

#%% save random locations
np.save(FILEPATH+"Config/RandomIndices.npy", ind)

#%% test saved locations
t = np.load(FILEPATH+"Config/RandomIndices.npy")




