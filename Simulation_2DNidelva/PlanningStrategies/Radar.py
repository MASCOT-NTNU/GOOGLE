"""
This script generates candidate location within the range
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-01-22
"""
import numpy as np

from usr_func import *


class Radar:

    lat_radar = None
    lon_radar = None
    depth_radar = None
    distance_radar = None

    def __init__(self, knowledge=None):
        self.knowledge = knowledge
        self.get_radar_surface()
        self.find_candidates_loc()
        pass

    def find_candidates_loc(self):
        delta_x, delta_y = latlon2xy(self.knowledge.coordinates[:, 0], self.knowledge.coordinates[:, 1],
                                     self.knowledge.coordinates[self.knowledge.ind_now, 0],
                                     self.knowledge.coordinates[self.knowledge.ind_now, 1])  # using the distance

        delta_z = self.knowledge.coordinates[:, 2] - self.knowledge.coordinates[self.knowledge.ind_now, 2]  # depth distance in z-direction
        self.distance_vector = (delta_x ** 2 / (1.5 * self.knowledge.distance_lateral) ** 2) + \
                              (delta_y ** 2 / (1.5 * self.knowledge.distance_lateral) ** 2) + \
                              (delta_z ** 2 / (self.knowledge.distance_vertical + 0.3) ** 2)
        self.knowledge.ind_cand = np.where((self.distance_vector <= 1) * (self.distance_vector > self.knowledge.distance_self))[0]

    # TODO: fix the plotting isosurface for the radar
    def get_radar_surface(self):

        nx = ny = 100
        nz = 10
        margin_xy = 10
        margin_z = 0.3
        xmin, xmax = -2 * self.knowledge.distance_neighbours + margin_xy, 2 * self.knowledge.distance_neighbours - margin_xy
        ymin, ymax = -2 * self.knowledge.distance_neighbours + margin_xy, 2 * self.knowledge.distance_neighbours - margin_xy
        zmin, zmax = -self.knowledge.distance_vertical - margin_z, self.knowledge.distance_vertical + margin_z
        x = np.linspace(xmin, xmax, nx)
        y = np.linspace(ymin, ymax, ny)
        z = np.linspace(zmin, zmax, nz)
        xv, yv, zv = np.meshgrid(x, y, z)
        xv, yv, zv = map(vectorise, [xv, yv, zv])
        self.distance_radar = (xv ** 2 / (2 * self.knowledge.distance_lateral - margin_xy)** 2) + \
                              (yv ** 2 / (2 * self.knowledge.distance_lateral - margin_xy) ** 2) + \
                              (zv ** 2 / (self.knowledge.distance_vertical + margin_z) ** 2)
        ind_radar_surface = np.where(self.distance_radar <= 1)[0]
        # ind_radar_surface = np.arange(len(self.distance_radar))
        self.lat_radar, self.lon_radar = xy2latlon(xv[ind_radar_surface], yv[ind_radar_surface],
                                                   self.knowledge.coordinates[self.knowledge.ind_now, 0],
                                                   self.knowledge.coordinates[self.knowledge.ind_now, 1])
        self.depth_radar = zv[ind_radar_surface] + self.knowledge.coordinates[self.knowledge.ind_now, 2]
        self.distance_radar = self.distance_radar[ind_radar_surface]
        print("lat_radar:", self.lat_radar)
        print("lon_radar: ", self.lon_radar)
        print("depth_radar: ", self.depth_radar)
        print("ind_radar", ind_radar_surface)
        print("length: ", len(self.lat_radar))



