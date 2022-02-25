"""
This usr_defined function contains universal functions needed
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-23
"""

'''
SYSTEM config
'''
import os, time, pathlib, re
from datetime import datetime

'''
Data importing modules
'''
import netCDF4

'''
SCIENTIFIC COMPUTING
'''
import numpy as np
from scipy.stats import mvn, norm
from skgstat import Variogram
import pandas as pd

'''
INTERPOLATION
'''
from scipy.interpolate import griddata
from scipy.interpolate import interpn
from scipy.interpolate import NearestNDInterpolator

'''
Plotting
'''
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
import matplotlib.path as mplPath  # used to determine whether a point is inside the grid or not
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})
# plt.rcParams.update({'font.style': 'oblique'})
import plotly
plotly.io.orca.config.save()
plotly.io.orca.config.executable = '/usr/local/bin/orca'
import plotly.graph_objects as go
from plotly.subplots import make_subplots

'''
CONSTANTS
CIRCUMFERENCE: earth circumference
SINMOD_MAX_DEPTH_LAYER: down to which layer should the data be used
'''
CIRCUMFERENCE = 40075000 # [m], circumference
SINMOD_MAX_DEPTH_LAYER = 8


def deg2rad(deg):
    return deg / 180 * np.pi


def rad2deg(rad):
    return rad / np.pi * 180


def latlon2xy(lat, lon, lat_origin, lon_origin):
    x = deg2rad((lat - lat_origin)) / 2 / np.pi * CIRCUMFERENCE
    y = deg2rad((lon - lon_origin)) / 2 / np.pi * CIRCUMFERENCE * np.cos(deg2rad(lat))
    return x, y


def xy2latlon(x, y, lat_origin, lon_origin):
    lat = lat_origin + rad2deg(x * np.pi * 2.0 / CIRCUMFERENCE)
    lon = lon_origin + rad2deg(y * np.pi * 2.0 / (CIRCUMFERENCE * np.cos(deg2rad(lat))))
    return lat, lon


def get_rotational_matrix(alpha):
    R = np.array([[np.cos(deg2rad(alpha)), np.sin(deg2rad(alpha))],
                  [-np.sin(deg2rad(alpha)), np.cos(deg2rad(alpha))]])
    return R


def setLoggingFilename(filename):
    import logging
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    DATE_FORMAT = "%m/%d/%Y %H:%M:%S %p"
    logging.basicConfig(filename=filename, level=logging.DEBUG, format=LOG_FORMAT, datefmt=DATE_FORMAT)


def vectorise(value):
    return np.array(value).reshape(-1, 1)


def EIBV_1D(threshold, mu, Sig, F, R):
    Sigxi = Sig @ F.T @ np.linalg.solve(F @ Sig @ F.T + R, F @ Sig)
    V = Sig - Sigxi
    sa2 = np.diag(V).reshape(-1, 1)  # the corresponding variance term for each location
    IntA = 0.0
    for i in range(len(mu)):
        sn2 = sa2[i]
        m = mu[i]
        IntA = IntA + mvn.mvnun(-np.inf, threshold, m, sn2)[0] - mvn.mvnun(-np.inf, threshold, m, sn2)[0] ** 2
    return IntA


def get_excursion_prob_1d(mu, Sigma, threshold):
    excursion_prob = np.zeros_like(mu)
    for i in range(excursion_prob.shape[0]):
        excursion_prob[i] = norm.cdf(threshold, mu[i], Sigma[i, i])
    return excursion_prob


def get_excursion_set(mu, threshold):
    excursion_set = np.zeros_like(mu)
    excursion_set[mu < threshold] = True
    return excursion_set


def GPupd(mu_cond, Sigma_cond, F, R, y_sampled):
    C = F @ Sigma_cond @ F.T + R
    mu_cond = mu_cond + Sigma_cond @ F.T @ np.linalg.solve(C,(y_sampled - F @ mu_cond))
    Sigma_cond = Sigma_cond - Sigma_cond @ F.T @ np.linalg.solve(C, F @ Sigma_cond)
    return mu_cond, Sigma_cond


def getFVector(ind, N):
    F = np.zeros([1, N])
    F[0, ind] = True
    return F


def get_grid_ind_at_nearest_loc(loc, coordinates):
    lat, lon, depth = loc
    dx, dy = latlon2xy(coordinates[:, 0], coordinates[:, 1], lat, lon)
    dz = coordinates[:, 2] - depth
    dist = dx ** 2 + dy ** 2 + dz ** 2
    ind = np.argmin(dist)
    return ind


def isEven(value):
    if value % 2 == 0:
        return True
    else:
        return False


def round2base(x, base=1.):
    return base * np.round(x/base)



def getRotationalMatrix_WGS2USR(angle):
    R = np.array([[np.cos(angle), -np.sin(angle), 0],
                  [np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])
    return R


def getRotationalMatrix_USR2WGS(angle):
    R = np.array([[np.cos(angle), np.sin(angle), 0],
                  [-np.sin(angle), np.cos(angle), 0],
                  [0, 0, 1]])
    return R


def checkfolder(folder):
    path = pathlib.Path(folder)
    path.mkdir(parents=True, exist_ok=True)
    print(folder + "is created")


def interpolate_2d(x, y, nx, ny, value, interpolation_method="linear"):
    '''
    Use griddata to interpolate to a finer results
    '''
    xmin, ymin = map(np.amin, [x, y])
    xmax, ymax = map(np.amax, [x, y])
    points = np.hstack((vectorise(x), vectorise(y)))
    xv = np.linspace(xmin, xmax, nx)
    yv = np.linspace(ymin, ymax, ny)
    grid_x, grid_y = np.meshgrid(xv, yv)
    grid_value = griddata(points, value, (grid_x, grid_y), method=interpolation_method)
    return grid_x, grid_y, grid_value


def interpolate_3d(x, y, z, value):
    '''
    Interpolates values for 3d grid by interpolate on 2d layers and combine them together
    '''
    z_layer = np.unique(z)
    grid = []
    values = []
    nx = 50
    ny = 50
    nz = len(z_layer)
    for i in range(nz):
        ind_layer = np.where(z == z_layer[i])[0]
        grid_x, grid_y, grid_value = interpolate_2d(x[ind_layer], y[ind_layer], nx=nx, ny=ny,
                                                    value=value[ind_layer], interpolation_method="cubic")
        grid_value = refill_nan_values(grid_value)
        for j in range(grid_x.shape[0]):
            for k in range(grid_x.shape[1]):
                grid.append([grid_x[j, k], grid_y[j, k], z_layer[i]])
                values.append(grid_value[j, k])

    grid = np.array(grid)
    values = np.array(values)
    return grid, values


def refill_nan_values(data):
    mask = np.where(~np.isnan(data))
    interp = NearestNDInterpolator(np.transpose(mask), data[mask])
    filled_data = interp(*np.indices(data.shape))
    return filled_data


def get_indices_equal2value(array, value):
    return np.where(array == value)[0]











