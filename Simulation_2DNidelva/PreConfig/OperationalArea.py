"""
This class will get the operational area
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-23
"""

import geopandas
import pandas as pd

from usr_func import *
from GOOGLE.Simulation_2DNidelva.Config.Config import FILEPATH, LATITUDE_ORIGIN, LONGITUDE_ORIGIN
from rdp import rdp # used to smooth path
BUFFER_SIZE_BORDER = -200 # [m]
BUFFER_SIZE_MUNKHOLMEN = 150 # [m]

'''
Path
'''
INPUT_PATH_MUNKHOLMEN_SHAPE_FILE = FILEPATH + "../GIS/Munkholmen.shp"
INPUT_PATH_SINMOD_SHAPE_FILE = FILEPATH + "PreConfig/SINMOD_Data_Region.csv"
OUTPUT_PATH_BORDER = FILEPATH + "PreConfig/Polygon_border.csv"
OUTPUT_PATH_MUNKHOLMEN = FILEPATH + "PreConfig/Polygon_munkholmen.csv"
OUTPUT_PATH_POLYGON_BORDER = FILEPATH + "Config/Polygon_border.csv"
OUTPUT_PATH_POLYGON_OBSTACLE = FILEPATH + "Config/Polygon_obstacle.csv"


class OpArea:

    def __init__(self):
        self.get_operational_area()
        # self.save_operational_areas()
        pass

    def get_operational_area(self):
        self.munkholmen_shape_file = geopandas.read_file(INPUT_PATH_MUNKHOLMEN_SHAPE_FILE)
        self.sinmod_shape_file = np.fliplr(pd.read_csv(INPUT_PATH_SINMOD_SHAPE_FILE))
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

    def get_buffered_area(self):
        lon_operational_area = vectorise(self.operational_areas[0].exterior.xy[0])
        lat_operational_area = vectorise(self.operational_areas[0].exterior.xy[1])
        polygon_border = np.hstack((lat_operational_area, lon_operational_area))
        polygon_obstacle = np.hstack((vectorise(self.lat_munkholmen), vectorise(self.lon_munkholmen)))

        # == buffer obstacle
        polygon_wgs_obstacle_shorten = self.get_buffered_polygon(polygon_obstacle, BUFFER_SIZE_MUNKHOLMEN)
        polygon_wgs_border_shorten = self.get_buffered_polygon(polygon_border, BUFFER_SIZE_BORDER)

        print("Before, ", polygon_border.shape, polygon_obstacle.shape)
        print("After: ", polygon_wgs_border_shorten.shape, polygon_wgs_obstacle_shorten.shape)
        x, y = latlon2xy(polygon_wgs_obstacle_shorten[:, 0], polygon_wgs_obstacle_shorten[:, 1],
                         LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        x, y = map(vectorise, [x, y])
        df = pd.DataFrame(np.hstack((x, y)), columns=["x", "y"])
        df.to_csv(OUTPUT_PATH_POLYGON_OBSTACLE, index=False)
        plt.plot(y, x, 'r.-')
        plt.show()

        # polygon_selected = polygon_wgs_border_shorten[1:-1]
        # polygon_selected = np.append(polygon_selected, polygon_selected[0, :].reshape(1, -1), axis=0)
        # df = pd.DataFrame(polygon_selected, columns=["lat", "lon"])
        x, y = latlon2xy(polygon_wgs_border_shorten[:, 0], polygon_wgs_border_shorten[:, 1],
                         LATITUDE_ORIGIN, LONGITUDE_ORIGIN)
        x, y = map(vectorise, [x, y])
        df = pd.DataFrame(np.hstack((x, y)), columns=["x", "y"])
        df.to_csv(OUTPUT_PATH_POLYGON_BORDER, index=False)

        plt.plot(polygon_obstacle[:, 1], polygon_obstacle[:, 0], 'k-')
        plt.plot(polygon_border[:, 1], polygon_border[:, 0], 'k-')
        plt.plot(polygon_wgs_obstacle_shorten[:, 1], polygon_wgs_obstacle_shorten[:, 0], 'r-')
        plt.plot(polygon_wgs_border_shorten[:, 1], polygon_wgs_border_shorten[:, 0], 'r-')
        # plt.plot(polygon_selected[:, 1], polygon_selected[:, 0], 'r-')
        plt.grid()
        plt.show()
        plt.plot(y, x, 'r.-')
        plt.show()

        df = pd.DataFrame(polygon_wgs_border_shorten, columns=['lat', 'lon'])
        df.to_csv(FILEPATH+"Test/polygon_border.csv", index=False)

        df = pd.DataFrame(polygon_wgs_obstacle_shorten, columns=['lat', 'lon'])
        df.to_csv(FILEPATH + "Test/polygon_obstacle.csv", index=False)

    def get_buffered_polygon(self, polygon, buffer_size):
        x, y = latlon2xy(polygon[:, 0], polygon[:, 1], 0, 0)
        polygon_xy = np.hstack((vectorise(x), vectorise(y)))
        polygon_xy_shapely = Polygon(polygon_xy)
        polygon_xy_shapely_buffered = polygon_xy_shapely.buffer(buffer_size)
        x_buffer, y_buffer = polygon_xy_shapely_buffered.exterior.xy

        # == shorten obstacle
        # polygon_xy_buffer = np.hstack((vectorise(x_buffer), vectorise(y_buffer)))
        # polygon_xy_buffer_shorten = rdp(polygon_xy_buffer, epsilon=epsilon)
        # lat_wgs_shorten, lon_wgs_shorten = xy2latlon(polygon_xy_buffer_shorten[:, 0],
        #                                              polygon_xy_buffer_shorten[:, 1],
        #                                              0, 0)
        lat_wgs, lon_wgs = xy2latlon(vectorise(x_buffer), vectorise(y_buffer), 0, 0)
        polygon_wgs_buffer_shorten = np.hstack((vectorise(lat_wgs), vectorise(lon_wgs)))
        return polygon_wgs_buffer_shorten

    def save_operational_areas(self):
        lon_operational_area = vectorise(self.operational_areas[0].exterior.xy[0])
        lat_operational_area = vectorise(self.operational_areas[0].exterior.xy[1])
        OpArea = np.hstack((lat_operational_area, lon_operational_area))
        df_OpArea = pd.DataFrame(OpArea, columns=['lat', 'lon'])
        df_OpArea.to_csv(OUTPUT_PATH_BORDER, index=False)

        OpMunkholmen = np.hstack((vectorise(self.lat_munkholmen), vectorise(self.lon_munkholmen)))
        df_munkholmen = pd.DataFrame(OpMunkholmen, columns=['lat', 'lon'])
        df_munkholmen.to_csv(OUTPUT_PATH_MUNKHOLMEN, index=False)
        pass


if __name__ == "__main__":
    op = OpArea()
    op.get_buffered_area()

