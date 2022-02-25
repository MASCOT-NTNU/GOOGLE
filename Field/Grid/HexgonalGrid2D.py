"""
This script generates regular hexgonal grid points within certain boundary
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-02-25
"""


from usr_func import *


class HexgonalGrid2D:

    def __init__(self, polygon_within=None, polygon_without=None, neighbour_distance=0):
        self.polygon_within = polygon_within
        self.polygon_without = polygon_without
        self.neighbour_distance = neighbour_distance
        print("Hello world")
        self.get_rectangular_boundary()
        self.get_grid_within_polygon()

        pass

    def get_rectangular_boundary(self):
        self.box_lat_min, self.box_lon_min = map(np.amin, [self.polygon_within[:, 0], self.polygon_within[:, 1]])
        self.box_lat_max, self.box_lon_max = map(np.amax, [self.polygon_within[:, 0], self.polygon_within[:, 1]])
        self.rectangular_boundary = np.array([[self.box_lat_min, self.box_lon_min],
                                              [self.box_lat_min, self.box_lon_max],
                                              [self.box_lat_max, self.box_lon_max],
                                              [self.box_lat_max, self.box_lon_min]])
        self.rectangular_boundary_path = mplPath.Path(self.rectangular_boundary)

    def get_grid_within_polygon(self):
        self.get_coverage()
        self.get_lateral_distance()
        self.get_vertical_distance()

        self.grid_x = np.arange(0, self.box_vertical_range, self.vertical_distance)
        self.grid_y = np.arange(0, self.box_lateral_range, self.lateral_distance)
        self.grid_xy = []
        self.grid_wgs = 
        for i in range(len(self.grid_y)):
            for j in range(len(self.grid_x)):
                if isEven(j):
                    self.grid_xy.append([self.grid_y[i] + self.lateral_distance / 2, self.grid_x[j]])
                else:
                    self.grid_xy.append([self.grid_y[i], self.grid_x[j]])
        self.grid_xy = np.array(self.grid_xy)
        # self.vertical, self.lateral = np.meshgrid(self.grid_vertical, self.grid_lateral)
        plt.plot(self.grid[:, 0], self.grid[:, 1], 'k.')
        plt.show()
        pass

    def get_coverage(self):
        self.box_lateral_range, self.box_vertical_range = latlon2xy(self.box_lat_max, self.box_lon_max,
                                                                    self.box_lat_min, self.box_lon_min)

    def get_lateral_distance(self):
        self.lateral_distance = self.neighbour_distance * np.cos(deg2rad(60)) * 2

    def get_vertical_distance(self):
        self.vertical_distance = self.neighbour_distance * np.sin(deg2rad(60))


if __name__ == "__main__":
    PATH_OPERATION_AREA = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Config/OpArea.csv"
    PATH_MUNKHOLMEN = "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Projects/GOOGLE/Config/Munkholmen.csv"
    polygon = pd.read_csv(PATH_OPERATION_AREA).to_numpy()
    munkholmen = pd.read_csv(PATH_MUNKHOLMEN).to_numpy()
    grid = HexgonalGrid2D(polygon_within=polygon, polygon_without=munkholmen, neighbour_distance=1200)





