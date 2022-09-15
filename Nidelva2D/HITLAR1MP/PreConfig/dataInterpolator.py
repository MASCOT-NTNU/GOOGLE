"""
This script interpolates data from sinmod onto the coordinates
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""
import numpy as np
import pandas as pd
from DataHandler.SINMOD import SINMOD
from GOOGLE.Nidelva2D.Config.Config import FILEPATH, LATITUDE_ORIGIN, LONGITUDE_ORIGIN
from usr_func import xy2latlon

grf_grid = pd.read_csv(FILEPATH+"Config/GRFGrid.csv").to_numpy()

lat, lon = xy2latlon(grf_grid[:, 0], grf_grid[:, 1], LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
coordinates = np.vstack((lat, lon, np.ones_like(lat))).T

sinmod = SINMOD()
sinmod.load_sinmod_data(raw_data=True)
sinmod.get_data_at_coordinates(coordinates)



