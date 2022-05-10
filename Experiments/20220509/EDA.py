
# load AUV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from usr_func import *

from DataHandler.SINMOD import SINMOD
circumference = 40075000


datapath = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Experiments/20220509/"
# sal = pd.read_csv("Salinity.csv")
# est = pd.read_csv("EstimatedState.csv")
# depth = pd.read_csv("Depth.csv")


#% Data extraction from the raw data
# rawTemp = pd.read_csv(datapath + "Temperature.csv", delimiter=', ', header=0, engine='python')
rawLoc = pd.read_csv(datapath + "EstimatedState.csv", delimiter=', ', header=0, engine='python')
rawSal = pd.read_csv(datapath + "Salinity.csv", delimiter=', ', header=0, engine='python')
rawDepth = pd.read_csv(datapath + "Depth.csv", delimiter=', ', header=0, engine='python')

# To group all the time stamp together, since only second accuracy matters
rawSal.iloc[:, 0] = np.ceil(rawSal.iloc[:, 0])
# rawTemp.iloc[:, 0] = np.ceil(rawTemp.iloc[:, 0])
# rawCTDTemp = rawTemp[rawTemp.iloc[:, 2] == 'SmartX']
rawLoc.iloc[:, 0] = np.ceil(rawLoc.iloc[:, 0])
rawDepth.iloc[:, 0] = np.ceil(rawDepth.iloc[:, 0])
rawDepth.iloc[:, 0] = np.ceil(rawDepth.iloc[:, 0])

depth_ctd = rawDepth[rawDepth.iloc[:, 2] == 'SmartX']["value (m)"].groupby(rawDepth["timestamp (seconds since 01/01/1970)"]).mean()
depth_dvl = rawDepth[rawDepth.iloc[:, 2] == 'DVL']["value (m)"].groupby(rawDepth["timestamp (seconds since 01/01/1970)"]).mean()
depth_est = rawLoc["depth (m)"].groupby(rawLoc["timestamp (seconds since 01/01/1970)"]).mean()

# indices used to extract data
lat_origin = rawLoc["lat (rad)"].groupby(rawLoc["timestamp (seconds since 01/01/1970)"]).mean()
lon_origin = rawLoc["lon (rad)"].groupby(rawLoc["timestamp (seconds since 01/01/1970)"]).mean()
x_loc = rawLoc["x (m)"].groupby(rawLoc["timestamp (seconds since 01/01/1970)"]).mean()
y_loc = rawLoc["y (m)"].groupby(rawLoc["timestamp (seconds since 01/01/1970)"]).mean()
z_loc = rawLoc["z (m)"].groupby(rawLoc["timestamp (seconds since 01/01/1970)"]).mean()
depth = rawLoc["depth (m)"].groupby(rawLoc["timestamp (seconds since 01/01/1970)"]).mean()
time_loc = rawLoc["timestamp (seconds since 01/01/1970)"].groupby(rawLoc["timestamp (seconds since 01/01/1970)"]).mean()
time_sal= rawSal["timestamp (seconds since 01/01/1970)"].groupby(rawSal["timestamp (seconds since 01/01/1970)"]).mean()
# time_temp = rawCTDTemp["timestamp (seconds since 01/01/1970)"].groupby(rawCTDTemp["timestamp (seconds since 01/01/1970)"]).mean()
dataSal = rawSal["value (psu)"].groupby(rawSal["timestamp (seconds since 01/01/1970)"]).mean()
# dataTemp = rawCTDTemp.iloc[:, -1].groupby(rawCTDTemp["timestamp"]).mean()

#% Rearrange data according to their timestamp
data = []
time_mission = []
xauv = []
yauv = []
zauv = []
dauv = []
sal_auv = []
temp_auv = []
lat_auv = []
lon_auv = []

for i in range(len(time_loc)):
#     if np.any(time_sal.isin([time_loc.iloc[i]])) and np.any(time_temp.isin([time_loc.iloc[i]])):
    if np.any(time_sal.isin([time_loc.iloc[i]])):
        time_mission.append(time_loc.iloc[i])
        xauv.append(x_loc.iloc[i])
        yauv.append(y_loc.iloc[i])
        zauv.append(z_loc.iloc[i])
        dauv.append(depth.iloc[i])
        lat_temp = np.rad2deg(lat_origin.iloc[i]) + np.rad2deg(x_loc.iloc[i] * np.pi * 2.0 / circumference)
        lat_auv.append(lat_temp)
        lon_auv.append(np.rad2deg(lon_origin.iloc[i]) + np.rad2deg(y_loc.iloc[i] * np.pi * 2.0 / (circumference * np.cos(np.deg2rad(lat_temp)))))
        sal_auv.append(dataSal[time_sal.isin([time_loc.iloc[i]])].iloc[0])
#         temp_auv.append(dataTemp[time_temp.isin([time_loc.iloc[i]])].iloc[0])
    else:
        print(datetime.fromtimestamp(time_loc.iloc[i]))
        continue

lat4, lon4 = 63.446905, 10.419426  # right bottom corner
lat_auv = np.array(lat_auv).reshape(-1, 1)
lon_auv = np.array(lon_auv).reshape(-1, 1)
Dx = np.deg2rad(lat_auv - lat4) / 2 / np.pi * circumference
Dy = np.deg2rad(lon_auv - lon4) / 2 / np.pi * circumference * np.cos(np.deg2rad(lat_auv))

xauv = np.array(xauv).reshape(-1, 1)
yauv = np.array(yauv).reshape(-1, 1)

alpha = np.deg2rad(60)
Rc = np.array([[np.cos(alpha), -np.sin(alpha)], [np.sin(alpha), np.cos(alpha)]])
TT = (Rc @ np.hstack((Dx, Dy)).T).T
xauv_new = TT[:, 0].reshape(-1, 1)
yauv_new = TT[:, 1].reshape(-1, 1)

zauv = np.array(zauv).reshape(-1, 1)
dauv = np.array(dauv).reshape(-1, 1)
sal_auv = np.array(sal_auv).reshape(-1, 1)
# temp_auv = np.array(temp_auv).reshape(-1, 1)
time_mission = np.array(time_mission).reshape(-1, 1)

# datasheet = np.hstack((time_mission, lat_auv, lon_auv, xauv, yauv, zauv, dauv, sal_auv, temp_auv))

coordinates = np.hstack((lat_auv, lon_auv, np.zeros_like(lat_auv)))

plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
plt.show()
sinmod = SINMOD()
#%%
sinmod.load_sinmod_data(raw_data=True)
sinmod.get_data_at_coordinates(coordinates)

#%%
data_sinmod = pd.read_csv(datapath + "data_interpolated.csv").to_numpy()

lat_sinmod = data_sinmod[:, 0]
lon_sinmod = data_sinmod[:, 1]
depth_sinmod = data_sinmod[:, 2]
sal_sinmod = data_sinmod[:, -1]
plt.scatter(lon_sinmod, lat_sinmod, c=sal_sinmod, cmap=get_cmap("BrBG", 10), vmin=10, vmax=20)
plt.colorbar()
plt.show()

#%%
LATITUDE_ORIGIN = 63.4269097
LONGITUDE_ORIGIN = 10.3969375
x, y = latlon2xy(lat_sinmod, lon_sinmod, LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
residual = sal_auv.flatten() - sal_sinmod
plt.scatter(lon_sinmod, lat_sinmod, c=residual, cmap=get_cmap("BrBG", 10))
plt.colorbar()
plt.show()
#%%
from skgstat import Variogram
v = Variogram(coordinates=np.vstack((x, y)).T, values=residual,
              use_nugget=True, n_lags=50, maxlag=2000)

v.plot()
plt.show()
v.cof

