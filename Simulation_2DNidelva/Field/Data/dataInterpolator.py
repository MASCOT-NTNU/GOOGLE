"""
This script interpolates data from sinmod onto the coordinates
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-16
"""
import pandas as pd
from DataHandler.SINMOD import SINMOD

PATH_CONFIG = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Simulation_2DNidelva/Config/"
coordinates = pd.read_csv(PATH_CONFIG+"Grid.csv").to_numpy()

sinmod = SINMOD()
sinmod.load_sinmod_data(raw_data=True)
sinmod.get_data_at_coordinates(coordinates)



