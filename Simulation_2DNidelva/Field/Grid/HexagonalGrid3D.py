"""
This script generates regular hexgonal grid points within certain boundary
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-25
"""


from usr_func import *


class HexgonalGrid3DGenerator:

    # TODO: use HexgonalGrid2D to produce 2D grid
    def __init__(self, polygon_within=None, polygon_without=None, depth=None, neighbour_distance=0):
        self.polygon_within = polygon_within
        self.polygon_without = polygon_without
        self.depth = depth
        self.neighbour_distance = neighbour_distance
        self.setup_polygons()
        self.get_bigger_box()
        self.get_grid_within_boundaries()

    def setup_polygons(self):
        self.polygon_within_path = mplPath.Path(self.polygon_within)
        self.polygon_without_path = mplPath.Path(self.polygon_without)

    def get_bigger_box(self):
        self.box_lat_min, self.box_lon_min = map(np.amin, [self.polygon_within[:, 0], self.polygon_within[:, 1]])
        self.box_lat_max, self.box_lon_max = map(np.amax, [self.polygon_within[:, 0], self.polygon_within[:, 1]])

    def get_grid_within_boundaries(self):
        self.get_distance_coverage()
        self.get_lateral_gap()
        self.get_vertical_gap()

        self.grid_x = np.arange(0, self.box_vertical_range, self.vertical_distance)
        self.grid_y = np.arange(0, self.box_lateral_range, self.lateral_distance)
        self.grid_xyz = []
        self.grid_wgs = []
        for i in range(len(self.grid_y)):
            for j in range(len(self.grid_x)):
                if isEven(j):
                    x = self.grid_x[j]
                    y = self.grid_y[i] + self.lateral_distance / 2
                else:
                    x = self.grid_x[j]
                    y = self.grid_y[i]
                lat, lon = xy2latlon(x, y, self.box_lat_min, self.box_lon_min)
                if self.isWithin((lat, lon)) and self.isWithout((lat, lon)):
                    for k in range(len(self.depth)):
                        self.grid_xyz.append([x, y, self.depth[k]])
                        self.grid_wgs.append([lat, lon, self.depth[k]])

        self.grid_xyz = np.array(self.grid_xyz)
        self.grid_wgs = np.array(self.grid_wgs)
        self.coordinates = self.grid_wgs

    def get_distance_coverage(self):
        self.box_lateral_range, self.box_vertical_range = latlon2xy(self.box_lat_max, self.box_lon_max,
                                                                    self.box_lat_min, self.box_lon_min)

    def get_lateral_gap(self):
        self.lateral_distance = self.neighbour_distance * np.cos(deg2rad(60)) * 2

    def get_vertical_gap(self):
        self.vertical_distance = self.neighbour_distance * np.sin(deg2rad(60))

    def isWithin(self, location):
        return self.polygon_within_path.contains_point(location)

    def isWithout(self, location):
        return not self.polygon_without_path.contains_point(location)


if __name__ == "__main__":
    PATH_OPERATION_AREA = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Config/OpArea.csv"
    PATH_MUNKHOLMEN = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Config/Munkholmen.csv"
    polygon = pd.read_csv(PATH_OPERATION_AREA).to_numpy()
    munkholmen = pd.read_csv(PATH_MUNKHOLMEN).to_numpy()
    grid = HexgonalGrid2D(polygon_within=polygon, polygon_without=munkholmen, neighbour_distance=500)





