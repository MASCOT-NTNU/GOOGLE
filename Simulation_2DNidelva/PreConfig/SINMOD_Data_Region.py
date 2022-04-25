"""
This class will get the operational area from SINMOD
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-04-25
"""

import numpy as np
import pandas as pd
from usr_func import latlon2xy
from GOOGLE.Simulation_2DNidelva.Config.Config import FILEPATH, LATITUDE_ORIGIN, LONGITUDE_ORIGIN

box = np.array([[63.4441527, 10.3296626],
                [63.4761121, 10.3948786],
                [63.4528538, 10.45186239],
                [63.4209213, 10.38662725]])
x, y = latlon2xy(box[:, 0], box[:, 1], LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
box_sinmod = np.vstack((x, y)).T

df = pd.DataFrame(box_sinmod, columns=['x', 'y'])

df.to_csv(FILEPATH+"PreConfig/SINMOD_Data_Region.csv", index=False)

#%% check with plot
import matplotlib.pyplot as plt
plt.plot(box[:, 1], box[:, 0])
plt.show()

#%% check with qgis
import os
os.system('qgis')

