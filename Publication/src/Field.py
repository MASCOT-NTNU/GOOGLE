"""
Field object defines the desired field of operation. It has a border polygon determines the boundary region, and it uses
list of obstacles to determine the area where operation needs to be avoided.
"""
from Config import Config
from WGS import WGS
from usr_func.is_list_empty import is_list_empty
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial.distance import cdist
from math import cos, sin, radians
from typing import Union


class Field:
    __config = Config()
    __grid = np.empty([0, 2])
    __neighbour_hash_table = dict()
    __neighbour_distance = 240  # metres between neighbouring locations.
    __plg = __config.get_polygon_operational_area()
    x, y = WGS.latlon2xy(__plg[:, 0], __plg[:, 1])
    __polygon_border = np.stack((x, y), axis=1)
    __polygon_border_shapely = Polygon(__polygon_border)

    """ Get the xy limits and gaps for the bigger box """
    xb = __polygon_border[:, 0]
    yb = __polygon_border[:, 1]
    __xmin, __ymin = map(np.amin, [xb, yb])
    __xmax, __ymax = map(np.amax, [xb, yb])
    __xlim = np.array([__xmin, __xmax])
    __ylim = np.array([__ymin, __ymax])
    __ygap = __neighbour_distance * cos(radians(60)) * 2
    __xgap = __neighbour_distance * sin(radians(60))

    def __init__(self):
        self.__construct_grid()
        self.__construct_hash_neighbours()

    def set_neighbour_distance(self, value: float) -> None:
        """ Set the neighbour distance """
        self.__neighbour_distance = value

    def set_polygon_border(self, value: np.ndarray) -> None:
        """ Set the polygon border, only one Nx2 dimension allowed """
        self.__polygon_border = value

    @staticmethod
    def border_contains(loc: np.ndarray) -> bool:
        """ Test if point is within the border polygon """
        x, y = loc
        point = Point(x, y)
        return Field.__polygon_border_shapely.contains(point)

    @staticmethod
    def is_border_in_the_way(loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """ Check if border is in the way between loc_start and loc_end. """
        xs, ys = loc_start
        xe, ye = loc_end
        line = LineString([(xs, ys), (xe, ye)])
        return Field.__polygon_border_shapely.intersects(line)

    def __construct_grid(self) -> None:
        """ Construct the field grid based on the instruction given above.
        - Construct regular meshgrid.
        .  .  .  .
        .  .  .  .
        .  .  .  .
        - Then move the even row to the right side.
        .  .  .  .
          .  .  .  .
        .  .  .  .
        - Then remove illegal locations.
        - Then add the depth layers.
        """
        gx = np.arange(Field.__xmin, Field.__xmax, Field.__xgap)  # get [0, x_gap, 2*x_gap, ..., (n-1)*x_gap]
        gy = np.arange(Field.__ymin, Field.__ymax, Field.__ygap)
        grid2d = []
        counter_grid2d = 0
        for i in range(len(gx)):
            for j in range(len(gy)):
                if i % 2 == 0:
                    y = gy[j] + Field.__ygap / 2
                    x = gx[i]
                else:
                    y = gy[j]
                    x = gx[i]
                loc = np.array([x, y])
                if self.border_contains(loc):
                    grid2d.append([x, y])
                    counter_grid2d += 1
        self.__grid = np.array(grid2d)
        
    def __construct_hash_neighbours(self) -> None:
        """ Construct the hash table for containing neighbour indices around each waypoint.
        - Directly use the neighbouring radius to determine the neighbouring indices.
        """
        no_grid = self.__grid.shape[0]
        ERROR_BUFFER = .01 * self.__neighbour_distance
        for i in range(no_grid):
            xy_c = self.__grid[i].reshape(1, -1)
            dist = cdist(self.__grid, xy_c)
            ind_n = np.where((dist <= self.__neighbour_distance + ERROR_BUFFER) *
                             (dist >= self.__neighbour_distance - ERROR_BUFFER))[0]
            self.__neighbour_hash_table[i] = ind_n

    def get_neighbour_indices(self, ind_now: Union[int, np.ndarray]) -> np.ndarray:
        """ Return neighbouring indices according to index current. """
        if type(ind_now) == np.int64:
            return self.__neighbour_hash_table[ind_now]
        else:
            neighbours = np.empty([0])
            for i in range(len(ind_now)):
                idnn = self.__neighbour_hash_table[ind_now[i]]
                neighbours = np.append(neighbours, idnn)
            return np.unique(neighbours.astype(int))

    def get_grid(self) -> np.ndarray:
        """
        Returns: waypoints
        """
        return self.__grid

    @staticmethod
    def get_neighbour_distance() -> float:
        """ Return neighbour distance. """
        return Field.__neighbour_distance

    @staticmethod
    def get_polygon_border() -> np.ndarray:
        """
        Returns: border polygon
        """
        return Field.__polygon_border

    def get_location_from_ind(self, ind: Union[int, list, np.ndarray]) -> np.ndarray:
        """
        Return waypoint locations using ind.
        """
        return self.__grid[ind, :]

    def get_ind_from_location(self, location: np.ndarray) -> Union[int, np.ndarray, None]:
        """
        Args:
            location: np.array([xp, yp])
        Returns: index of the closest waypoint.
        """

        if len(location) > 0:
            dm = location.ndim
            if dm == 1:
                d = cdist(self.__grid, location.reshape(1, -1))
                return np.argmin(d, axis=0)[0]
            elif dm == 2:
                d = cdist(self.__grid, location)
                return np.argmin(d, axis=0)
            else:
                return None
        else:
            return None

    @staticmethod
    def get_border_limits():
        return Field.__xlim, Field.__ylim


if __name__ == "__main__":
    f = Field()

