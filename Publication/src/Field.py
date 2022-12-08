"""
Field handles field discretization and validation.
- generate grid discretization.
- check legal conditions of given locations.
- check collision with obstacles.
"""
from Config import Config
import numpy as np
from shapely.geometry import Point, LineString
from scipy.spatial.distance import cdist
from math import cos, sin, radians
from typing import Union


class Field:
    """
    Field handles everything with regarding to the field element.
    """
    def __init__(self):
        # config element
        self.__config = Config()
        self.__neighbour_distance = 120  # metres between neighbouring locations.

        # border element
        self.__polygon_border = self.__config.get_polygon_border()
        self.__polygon_border_shapely = self.__config.get_polygon_border_shapely()
        self.__line_border_shapely = LineString(self.__polygon_border)

        # obstacle element
        self.__polygon_obstacle = self.__config.get_polygon_obstacle()
        self.__polygon_obstacle_shapely = self.__config.get_polygon_obstacle_shapely()
        self.__line_obstacle_shapely = LineString(self.__polygon_obstacle)
        
        """ Get the xy limits and gaps for the bigger box """
        xb = self.__polygon_border[:, 0]
        yb = self.__polygon_border[:, 1]
        self.__xmin, self.__ymin = map(np.amin, [xb, yb])
        self.__xmax, self.__ymax = map(np.amax, [xb, yb])
        self.__xlim = np.array([self.__xmin, self.__xmax])
        self.__ylim = np.array([self.__ymin, self.__ymax])
        self.__ygap = self.__neighbour_distance * cos(radians(60)) * 2
        self.__xgap = self.__neighbour_distance * sin(radians(60))

        # grid element
        self.__grid = np.empty([0, 2])
        self.__construct_grid()

        # neighbour element
        self.__neighbour_hash_table = dict()
        self.__construct_hash_neighbours()

    def set_neighbour_distance(self, value: float) -> None:
        """ Set the neighbour distance """
        self.__neighbour_distance = value

    def border_contains(self, loc: np.ndarray) -> bool:
        """ Test if point is within the border polygon """
        x, y = loc
        point = Point(x, y)
        return self.__polygon_border_shapely.contains(point)

    def obstacle_contains(self, loc: np.ndarray) -> bool:
        """ Test if obstacle contains the point. """
        x, y = loc
        point = Point(x, y)
        return self.__polygon_obstacle_shapely.contains(point)

    def is_border_in_the_way(self, loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """ Check if border is in the way between loc_start and loc_end. """
        xs, ys = loc_start
        xe, ye = loc_end
        line = LineString([(xs, ys), (xe, ye)])
        return self.__line_border_shapely.intersects(line)

    def is_obstacle_in_the_way(self, loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """ Check if border is in the way between loc_start and loc_end. """
        xs, ys = loc_start
        xe, ye = loc_end
        line = LineString([(xs, ys), (xe, ye)])
        return self.__line_obstacle_shapely.intersects(line)

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
        gx = np.arange(self.__xmin, self.__xmax, self.__xgap)  # get [0, x_gap, 2*x_gap, ..., (n-1)*x_gap]
        gy = np.arange(self.__ymin, self.__ymax, self.__ygap)
        grid2d = []
        counter_grid2d = 0
        for i in range(len(gx)):
            for j in range(len(gy)):
                if i % 2 == 0:
                    y = gy[j] + self.__ygap / 2
                    x = gx[i]
                else:
                    y = gy[j]
                    x = gx[i]
                loc = np.array([x, y])
                if self.border_contains(loc) and not self.obstacle_contains(loc):
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
        """ Return neighbouring indices according to given current index. """
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
        Returns: grf grid.
        """
        return self.__grid

    def get_neighbour_distance(self) -> float:
        """ Return neighbour distance. """
        return self.__neighbour_distance

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

    def get_border_limits(self):
        return self.__xlim, self.__ylim


if __name__ == "__main__":
    f = Field()

