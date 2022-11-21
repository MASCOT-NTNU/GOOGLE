"""
Field object defines the desired field of operation. It has a border polygon determines the boundary region, and it uses
list of obstacles to determine the area where operation needs to be avoided.
"""

from usr_func.is_list_empty import is_list_empty
import numpy as np
from shapely.geometry import Polygon, Point, LineString
from scipy.spatial.distance import cdist
from math import cos, sin, radians
from typing import Union


class Field:

    __grid = np.empty([0, 2])
    __neighbour_distance = .05
    __polygon_border = np.array([[.0, .0],
                                 [1., .0],
                                 [1., 1.],
                                 [.0, 1.]])
    __polygon_obstacles = [np.array([[.4, .4],
                                     [.6, .5],
                                     [.5, .6],
                                     [.3, .4]])]
    __obs_free = True
    __polygon_border_shapely = Polygon(__polygon_border)
    if not is_list_empty(__polygon_obstacles):
        __polygon_obstacles_shapely = []
        for po in __polygon_obstacles:
            __polygon_obstacles_shapely.append(Polygon(po))
        __obs_free = False

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
        self.__construct_field()

    def set_neighbour_distance(self, value: float) -> None:
        """ Set the neighbour distance """
        self.__neighbour_distance = value

    def set_polygon_border(self, value: np.ndarray) -> None:
        """ Set the polygon border, only one Nx2 dimension allowed """
        self.__polygon_border = value

    def set_polygon_obstacles(self, value: list) -> None:
        """ Set the polygons for obstacles, can have multiple obstacles """
        self.__polygon_obstacles = value

    @staticmethod
    def border_contains(loc: np.ndarray) -> bool:
        """ Test if point is within the border polygon """
        x, y = loc
        point = Point(x, y)
        return Field.__polygon_border_shapely.contains(point)

    @staticmethod
    def obstacles_contain(loc: np.ndarray) -> bool:
        """ Test if point is colliding with any obstacle polygons """
        x, y = loc
        point = Point(x, y)
        obs = False
        for posi in Field.__polygon_obstacles_shapely:
            if posi.contains(point):
                obs = True
                break
        return obs

    @staticmethod
    def is_border_in_the_way(loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """ Check if border is in the way between loc_start and loc_end. """
        xs, ys = loc_start
        xe, ye = loc_end
        line = LineString([(xs, ys), (xe, ye)])
        return Field.__polygon_border_shapely.intersects(line)

    @staticmethod
    def is_obstacle_in_the_way(loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """ Check if obstacles are in the way between loc_start and loc_end. """
        xs, ys = loc_start
        xe, ye = loc_end
        line = LineString([(xs, ys), (xe, ye)])
        collision = False
        for posi in Field.__polygon_obstacles_shapely:
            if posi.intersects(line):
                collision = True
                break
        return collision

    def __construct_field(self) -> None:
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
                if j % 2 == 0:
                    x = gx[i] + Field.__xgap / 2
                    y = gy[j]
                else:
                    x = gx[i]
                    y = gy[j]
                loc = np.array([x, y])
                if self.__obs_free:
                    if self.border_contains(loc):
                        grid2d.append([x, y])
                        counter_grid2d += 1
                else:
                    if self.border_contains(loc) and not self.obstacles_contain(loc):
                        grid2d.append([x, y])
                        counter_grid2d += 1
        self.__grid = np.array(grid2d)

    def get_grid(self):
        """
        Returns: waypoints
        """
        return self.__grid

    @staticmethod
    def get_polygon_border():
        """
        Returns: border polygon
        """
        return Field.__polygon_border

    @staticmethod
    def get_polygon_obstacles():
        """
        Returns: obstacles' polygons.
        """
        return Field.__polygon_obstacles

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

