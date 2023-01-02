"""
This class will get the operational area
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-23
"""

import geopandas
import pandas as pd
from shapely.geometry import Polygon, GeometryCollection
from usr_func import *

'''
Path
'''
PATH_SHAPE_FILE = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/GIS/Munkholmen.shp"
SINMOD_SHAPE_FILE = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Config/SINMOD_Data_Region.csv"
PATH_OPERATION_AREA = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Config/OpArea.csv"
PATH_MUNKHOLMEN = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Config/Munkholmen.csv"


class OpArea:

    def __init__(self):
        self.get_operational_area()
        self.save_operational_areas()
        pass

    def get_operational_area(self):
        self.munkholmen_shape_file = geopandas.read_file(PATH_SHAPE_FILE)
        self.sinmod_shape_file = np.fliplr(pd.read_csv(SINMOD_SHAPE_FILE))
        self.polygon_sinmod = Polygon(self.sinmod_shape_file)

        self.Trondheimsfjord = self.munkholmen_shape_file[self.munkholmen_shape_file['name'] == "Trondheimsfjorden"]['geometry']
        self.polygon_trondheimsfjorden = self.Trondheimsfjord.to_numpy()[0][0]
        self.lon_fjord = np.array(self.polygon_trondheimsfjorden.exterior.xy[0])
        self.lat_fjord = np.array(self.polygon_trondheimsfjorden.exterior.xy[1])

        self.Munkholmen = self.munkholmen_shape_file[self.munkholmen_shape_file['name'] == "Munkholmen"]['geometry']
        self.polygon_munkholmen = self.Munkholmen.to_numpy()[0]
        self.lon_munkholmen = np.array(self.polygon_munkholmen.exterior.xy[0])
        self.lat_munkholmen = np.array(self.polygon_munkholmen.exterior.xy[1])

        self.intersection = []
        self.intersection.append(self.polygon_trondheimsfjorden.intersection(self.polygon_sinmod))
        self.intersection.append(self.polygon_trondheimsfjorden.intersection(self.polygon_munkholmen))

        self.operational_regions = GeometryCollection(self.intersection)
        self.operational_areas = self.operational_regions.geoms

        fig = plt.figure(figsize=(10, 10))

        lon_operational_area = np.array(self.operational_areas[0].exterior.xy[0])
        lat_operational_area = np.array(self.operational_areas[0].exterior.xy[1])
        plt.plot(lon_operational_area, lat_operational_area, 'k-.', markersize=500)

        for i in range(len(self.operational_areas[1].geoms)):
            lon_operational_area = np.array(self.operational_areas[1].geoms[i].coords)[:, 0]
            lat_operational_area = np.array(self.operational_areas[1].geoms[i].coords)[:, 1]
            plt.plot(lon_operational_area, lat_operational_area, 'k-.', markersize=500)

        # # plt.plot(sinmod_region['lon'], sinmod_region['lat'], 'r-', ms=12)
        # plt.scatter(sinmod.lon, sinmod.lat, c=sinmod.salinity_average[0, :, :], cmap="Paired", vmin=0, vmax=33)
        # plt.colorbar()
        # # plt.plot(lon_munk, lat_munk, 'k-')
        plt.show()
        pass

    def save_operational_areas(self):
        lon_operational_area = vectorise(self.operational_areas[0].exterior.xy[0])
        lat_operational_area = vectorise(self.operational_areas[0].exterior.xy[1])
        OpArea = np.hstack((lat_operational_area, lon_operational_area))
        df_OpArea = pd.DataFrame(OpArea, columns=['lat', 'lon'])
        df_OpArea.to_csv(PATH_OPERATION_AREA, index=False)

        OpMunkholmen = np.hstack((vectorise(self.lat_munkholmen), vectorise(self.lon_munkholmen)))
        df_munkholmen = pd.DataFrame(OpMunkholmen, columns=['lat', 'lon'])
        df_munkholmen.to_csv(PATH_MUNKHOLMEN, index=False)
        pass


if __name__ == "__main__":
    op = OpArea()

