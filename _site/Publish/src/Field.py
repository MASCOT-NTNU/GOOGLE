"""
Field module handles field discretization and validation.

Author:
    Yaolin Ge
    yaolin.ge@ntnu.no

Objective:
    1. Generate grid discretization.
    2. Check legal conditions of given locations.
    3. Check collision with obstacles.

Examples:
    >>> from Field import Field
    >>> f = Field()
    >>> f.get_grid()
    np.array([[ 0.  0.]
        [ 0.  1.]
        ...
        [ 9.  8.]])
    >>> f.get_neighbour_indices(0)
    1 2 3 4

    >>> f.border_contains(np.array([10, 10]))
    True
    >>> f.obstacle_contains(np.array([10, 10]))
    False

    >>> f.is_border_in_the_way(np.array([10, 10]), np.array([20, 20]))
    False
    >>> f.is_obstacle_in_the_way(np.array([10, 10]), np.array([20, 20]))
    True

    >>> f.set_neighbour_distance(100)
    >>> f.get_neighbour_distance()
    100.0

    >>> f.get_border_limits()
    (0.0, 0.0, 9.0, 8.0)

    >>> f.get_border_polygon()
    np.array([[ 0.,  0.],
        [ 0.,  8.],
        [ 9.,  8.],
        [ 9.,  0.],
        [ 0.,  0.]])

    >>> f.get_obstacle_polygon()
    np.array([[ 2.,  2.],
        [ 2.,  6.],
        [ 7.,  6.],
        [ 7.,  2.],
        [ 2.,  2.]])

    >>> f.get_border_line()
    LineString([(0.0, 0.0), (0.0, 8.0), (9.0, 8.0), (9.0, 0.0), (0.0, 0.0)])

    >>> f.get_obstacle_line()
    LineString([(2.0, 2.0), (2.0, 6.0), (7.0, 6.0), (7.0, 2.0), (2.0, 2.0)])

    >>> f.get_ind_from_location(np.array([10, 10]))
    0

    >>> f.get_location_from_ind(10)
    np.array([ 1.,  0.])

Notes:
    1. The coordinate system is x-y with x pointing up, y pointing to the right which cooresponds to NED system
    (North-East-Down).
    2. The origin is at the bottom left corner of the field, i.e. the bottom left corner has coordinates (0, 0).

References:
    TODO: Add references.
    TODO: Add figures to demonstrate the capaibility.

"""

from Config import Config
import numpy as np
from shapely.geometry import Point, LineString
from scipy.spatial.distance import cdist
from math import cos, sin, radians
from typing import Union


class Field:
    """
    Class to handle everything related to the field element, including border, obstacles, grid, and neighbors.

    Attributes:
        __config (Config): Configuration object.
        __neighbour_distance (float): Distance between neighboring locations.
        __polygon_border (np.ndarray): Array of polygon border vertices.
        __polygon_border_shapely (shapely.geometry.Polygon): Shapely polygon object for border.
        __line_border_shapely (shapely.geometry.LineString): Shapely line string object for border.
        __polygon_obstacle (np.ndarray): Array of polygon obstacle vertices.
        __polygon_obstacle_shapely (shapely.geometry.Polygon): Shapely polygon object for obstacle.
        __line_obstacle_shapely (shapely.geometry.LineString): Shapely line string object for obstacle.
        __xmin (float): Minimum x value of the border.
        __ymin (float): Minimum y value of the border.
        __xmax (float): Maximum x value of the border.
        __ymax (float): Maximum y value of the border.
        __xlim (np.ndarray): Array of [xmin, xmax].
        __ylim (np.ndarray): Array of [ymin, ymax].
        __ygap (float): Gap between rows.
        __xgap (float): Gap between columns.
        __grid (np.ndarray): Array of field grid points.
        __neighbour_hash_table (dict): Hash table for containing neighboring indices around each waypoint.

    """
    def __init__(self, neighbour_distance: float = 120) -> None:
        """
        Initialize the Field object.

        Args:
            neighbour_distance: float, distance between neighbouring locations.

        Returns:
            None

        """
        self.__config = Config()
        self.__neighbour_distance = neighbour_distance  # metres between neighbouring locations.

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
        """
        Set the neighbour distance.

        Args:
            value: New neighbour distance in meters.

        Returns:
            None
        """
        self.__neighbour_distance = value

    def border_contains(self, loc: np.ndarray) -> bool:
        """
        Test if point is within the border polygon.

        Args:
            loc: Location (x, y) to test.

        Returns:
            True if point is within the border polygon, False otherwise.

        """
        x, y = loc
        point = Point(x, y)
        return self.__polygon_border_shapely.contains(point)

    def obstacle_contains(self, loc: np.ndarray) -> bool:
        """
        Test if obstacle contains the point.

        Args:
            loc: Location (x, y) to test.

        Returns:
            True if obstacle contains the point, False otherwise.

        """
        x, y = loc
        point = Point(x, y)
        return self.__polygon_obstacle_shapely.contains(point)

    def is_border_in_the_way(self, loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """
        Check if border is in the way between loc_start and loc_end.

        Args:
            loc_start: Start location.
            loc_end: End location.

        Returns:
            True if border is in the way, False otherwise.

        """
        xs, ys = loc_start
        xe, ye = loc_end
        line = LineString([(xs, ys), (xe, ye)])
        return self.__line_border_shapely.intersects(line)

    def is_obstacle_in_the_way(self, loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """
        Check if obstacle is in the way between loc_start and loc_end.

        Args:
            loc_start: Start location.
            loc_end: End location.

        Returns:
            True if obstacle is in the way, False otherwise.

        """
        xs, ys = loc_start
        xe, ye = loc_end
        line = LineString([(xs, ys), (xe, ye)])
        return self.__line_obstacle_shapely.intersects(line)

    def __construct_grid(self) -> None:
        """
        Construct the field grid using a regular meshgrid.

        Steps:
            1. Create a meshgrid with x and y coordinates.
            2. Shift the y coordinates by half the y gap.
            3. Filter out the points that are not within the border polygon or within obstacle polygon.
            4. Append the points to the grid.

        Returns:
            None

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
        """
        Construct the hash table for containing neighbour indices around each waypoint.

        Methods: Directly use the neighbouring radius to determine the neighbouring indices.

        Returns:
            None

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
        """
        Return neighbouring indices according to given current index.

        Args:
            ind_now: Current index.

        Returns:
            An array of neighbouring indices.

        """
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
        Return the grid.

        Args:
            None

        Returns:
            The grid in np.ndarray format.

            Examples:
                grid = np.array([[x1, y1], [x2, y2], ..., [xn, yn]])

        """
        return self.__grid

    def get_neighbour_distance(self) -> float:
        """
        Return neighbour distance.

        Args:
            None

        Returns:
            Neighbour distance as a float number.

        """
        return self.__neighbour_distance

    def get_location_from_ind(self, ind: Union[int, list, np.ndarray]) -> np.ndarray:
        """
        Return waypoint locations using ind.

        Args:
            ind: Index of the waypoint.

        Returns:
            Waypoint locations in np.ndarray format.

        Examples:
            ind = 1
            location = np.array([x1, y1])

        """
        return self.__grid[ind, :]

    def get_ind_from_location(self, location: np.ndarray) -> Union[np.ndarray, None]:
        """
        Return waypoint index using location.

        Args:
            location: Waypoint location.

        Returns:
            Waypoint index in np.ndarray format.

        Examples:
            ind = 1
            location = np.array([x1, y1])

        """

        if len(location) > 0:
            dm = location.ndim
            if dm == 1:
                d = cdist(self.__grid, location.reshape(1, -1))
                return np.argmin(d, axis=0)
            elif dm == 2:
                d = cdist(self.__grid, location)
                return np.argmin(d, axis=0)
            else:
                return None
        else:
            return None

    def get_border_limits(self):
        """
        Return the border limits.

        Returns:
            The border limits in tuple format.

        Examples:
            xlim = (xmin, xmax)
            ylim = (ymin, ymax)

        """
        return self.__xlim, self.__ylim


if __name__ == "__main__":
    f = Field()

