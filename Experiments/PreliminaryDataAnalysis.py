
# load AUV
import os

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from datetime import datetime
from usr_func import *

from DataHandler.SINMOD import SINMOD
circumference = 40075000


datapath = os.getcwd() + "/GOOGLE/Experiments/20220510/csv/"
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

# plt.plot(coordinates[:, 1], coordinates[:, 0], 'k.')
plt.scatter(coordinates[:, 1], coordinates[:, 0], c=sal_auv, cmap=get_cmap("BrBG", 10), vmin=5, vmax=30)
plt.colorbar()
plt.show()

#%%
from usr_func import *

fig = go.Figure(data=go.Scatter3d(
    x=lon_auv.flatten(),
    y=lat_auv.flatten(),
    z=-dauv.flatten(),
    mode="markers",
    marker=dict(color=sal_auv.flatten(), size=2, cmin=25, cmax=30, showscale=True),
    # line=dict(color='black', width=1),
))

plotly.offline.plot(fig, filename="/Users/yaolin/HomeOffice/GOOGLE/Experiments/auv.html", auto_open=True)


#%%
#
# sinmod = SINMOD()
# #%% data interpolation section
# sinmod.load_sinmod_data(raw_data=True)
# # sinmod.get_data_at_coordinates(coordinates)
# #%%
#
# DATAPATH = "/Users/yaolin/HomeOffice/GOOGLE/Experiments/20220510/"
# #% Step II: extract data by section
# p1 = coordinates[0:5000,:]
# sinmod.get_data_at_coordinates(p1, filename=DATAPATH+'p1.csv')
#
# #%%
# p2 = coordinates[5000:10000,:]
# sinmod.get_data_at_coordinates(p2, filename=DATAPATH+'p2.csv')
# os.system('say complete 2')
# #%%
# p3 = coordinates[10000:-1,:]
# sinmod.get_data_at_coordinates(p3, filename=DATAPATH+'p3.csv')
# os.system('say complete 3')
#
# #%%
#
# datapath = "/Users/yaolin/HomeOffice/GOOGLE/Experiments/20220510/"
# import os
# import pandas as pd
#
# df = []
# files = os.listdir(datapath)
# for file in files:
#     if file.endswith(".csv"):
#         df.append(pd.read_csv(datapath+file))
#
# #%%
# # for file in files:
# file = files[0]
# df1 = pd.read_csv(datapath+file)
#
# file = files[1]
# df2 = pd.read_csv(datapath+file)
#
# file = files[2]
# df3 = pd.read_csv(datapath+file)
# #%%
# # file = files[3]
# # df4 = pd.read_csv(datapath+file)
# #
# # file = files[4]
# # df5 = pd.read_csv(datapath+file)
#
# # df = np.vstack((df1, df2, df3, df4, df5))
# df = pd.concat([d for d in df], ignore_index=True, sort=False)
# df.to_csv(datapath + "data_sinmod.csv", index=False)
# os.system('say complete all')
#%%
datapath = "/Users/yaolin/HomeOffice/GOOGLE/Experiments/20220510/"
# data_sinmod = pd.read_csv(datapath + "data_interpolated.csv").to_numpy()
data_sinmod = pd.read_csv(datapath + "data_sinmod.csv").to_numpy()

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
residual = sal_auv[:-1].flatten() - sal_sinmod
plt.scatter(lon_sinmod, lat_sinmod, c=residual, cmap=get_cmap("BrBG", 10))
plt.colorbar()
plt.show()
#%%
ind = np.random.randint(0, len(x), 6000)
from skgstat import Variogram
v = Variogram(coordinates=np.vstack((x[ind], y[ind])).T, values=residual[ind],
              use_nugget=True, n_lags=50, maxlag=2000)

v.plot()
plt.show()
v.cof

