"""
Discretizes the rectangular field formed by (xrange, yrange) with distance_neighbour.
Sets the boundary and neighbour distance for the discretization under NED coordinate system.
- N: North
- E: East
- D: Down

Args:
    polygon_border: border vertices defined by [[x1, y1], [x2, y2], ..., [xn, yn]].
    polygon_obstacles: multiple obstalce vertices defined by [[[x11, y11], [x21, y21], ... [xn1, yn1]], [[...]]].
    depths: multiple depth layers [d0, d1, d2, ..., dn].
    distance_neighbour: distance between neighbouring waypoints.

The resulting grid will be like:
    _________
   /  .   .  \
  /  .  /\   .\
  \   ./__\   .\
   \.   .   .  /
    \_________/

Get:
    Waypoints: [[x0, y0, z0],
               [x1, y1, z1],
               ...
               [xn, yn, zn]]
    Neighbour hash tables: {0: [1, 2, 3], 1: [0, 2, 3], ..., }
"""
from typing import Any, Union
import numpy as np
from scipy.spatial.distance import cdist
from math import cos, sin, radians
from shapely.geometry import Polygon, Point
from usr_func.is_list_empty import is_list_empty


class WaypointGraph:

    def __init__(self):
        self.__waypoints = np.empty([0, 3])  # put it inside the initialisation to avoid mutation.
        self.__neighbour = dict()
        self.__neighbour_distance = 0
        self.__depths = []
        self.__polygon_border = np.array([[0, 0],
                                          [0, 0],
                                          [0, 0]])
        self.__polygon_obstacles = [np.array([[]])]

    def set_neighbour_distance(self, value: float) -> None:
        """ Set the neighbour distance """
        self.__neighbour_distance = value

    def set_depth_layers(self, value: list) -> None:
        """ Set the depth layers """
        self.__depths = np.array(value)

    def set_polygon_border(self, value: np.ndarray) -> None:
        """ Set the polygon border, only one Nx2 dimension allowed """
        self.__polygon_border = value
        self.__polygon_border_shapely = Polygon(self.__polygon_border)

    def set_polygon_obstacles(self, value: list) -> None:
        """ Set the polygons for obstacles, can have multiple obstacles """
        self.__polygon_obstacles = value
        self.obs_free = True
        if not is_list_empty(self.__polygon_obstacles):
            self.__polygon_obstacles_shapely = []
            for po in self.__polygon_obstacles:
                self.__polygon_obstacles_shapely.append(Polygon(po))
            self.obs_free = False

    def __border_contains(self, point: Point):
        """ Test if point is within the border polygon """
        return self.__polygon_border_shapely.contains(point)

    def __obstacles_contain(self, point: Point):
        """ Test if point is colliding with any obstacle polygons """
        obs = False
        for posi in self.__polygon_obstacles_shapely:
            if posi.contains(point):
                obs = True
                break
        return obs

    def __get_xy_limits(self):
        """ Get the xy limits for the bigger box """
        xb = self.__polygon_border[:, 0]
        yb = self.__polygon_border[:, 1]
        self.__xmin, self.__ymin = map(np.amin, [xb, yb])
        self.__xmax, self.__ymax = map(np.amax, [xb, yb])

    def __get_xy_gaps(self):
        """ Get the gap distance along each xy-axis """
        self.__ygap = self.__neighbour_distance * cos(radians(60)) * 2
        self.__xgap = self.__neighbour_distance * sin(radians(60))

    def construct_waypoints(self) -> None:
        """ Construct the waypoint graph based on the instruction given above.
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
        self.__get_xy_limits()
        self.__get_xy_gaps()

        gx = np.arange(self.__xmin, self.__xmax, self.__xgap)  # get [0, x_gap, 2*x_gap, ..., (n-1)*x_gap]
        gy = np.arange(self.__ymin, self.__ymax, self.__ygap)
        grid2d = []
        counter_grid2d = 0
        for i in range(len(gy)):
            for j in range(len(gx)):
                if j % 2 == 0:
                    x = gx[j]
                    y = gy[i] + self.__ygap / 2
                else:
                    x = gx[j]
                    y = gy[i]
                p = Point(x, y)
                if self.obs_free:
                    if self.__border_contains(p):
                        grid2d.append([x, y])
                        counter_grid2d += 1
                else:
                    if self.__border_contains(p) and not self.__obstacles_contain(p):
                        grid2d.append([x, y])
                        counter_grid2d += 1

        self.multiple_depth_layer = False
        self.no_depth_layers = len(self.__depths)
        if self.no_depth_layers > 1:
            self.multiple_depth_layer = True

        for i in range(self.no_depth_layers):
              for j in range(counter_grid2d):
                self.__waypoints = np.append(self.__waypoints,
                                             np.array([grid2d[j][0], grid2d[j][1],
                                                       self.__depths[i]]).reshape(1, -1), axis=0)
        self.__waypoints = np.array(self.__waypoints)

    def construct_hash_neighbours(self):
        """ Construct the hash table for containing neighbour indices around each waypoint.
        - Get the adjacent depth layers
            - find the current depth layer index, then find the upper and lower depth layer indices.
            - find the corresponding waypoints.
        - Get the lateral neighbour indices for each layer.
        - Append all the neighbour indices for each waypoint.
        """
        # check adjacent depth layers to determine the neighbouring waypoints.
        self.no_waypoint = self.__waypoints.shape[0]
        ERROR_BUFFER = .01 * self.__neighbour_distance
        for i in range(self.no_waypoint):
            # determine ind depth layer
            xy_c = self.__waypoints[i, 0:2].reshape(1, -1)
            d_c = self.__waypoints[i, 2]
            ind_d = np.where(self.__depths == d_c)[0][0]

            # determine ind adjacent layers
            ind_u = ind_d + 1 if ind_d < self.no_depth_layers - 1 else ind_d
            ind_l = ind_d - 1 if ind_d > 0 else 0

            # compute lateral distance
            id = np.unique([ind_d, ind_l, ind_u])
            ds = self.__depths[id]

            ind_n = []
            for ids in ds:
                ind_id = np.where(self.__waypoints[:, 2] == ids)[0]
                xy = self.__waypoints[ind_id, 0:2]
                dist = cdist(xy, xy_c)
                ind_n_temp = np.where((dist <= self.__neighbour_distance + ERROR_BUFFER) *
                                      (dist >= self.__neighbour_distance - ERROR_BUFFER))[0]
                for idt in ind_n_temp:
                    ind_n.append(ind_id[idt])
            self.__neighbour[i] = ind_n

    def get_waypoints(self):
        """
        Returns: waypoints
        """
        return self.__waypoints

    def get_hash_neighbour(self):
        """
        Returns: neighbour hash table
        """
        return self.__neighbour

    def get_polygon_border(self):
        """
        Returns: border polygon
        """
        return self.__polygon_border

    def get_polygon_obstacles(self):
        """
        Returns: obstacles' polygons.
        """
        return self.__polygon_obstacles

    def get_neighbour_distance(self):
        """
        Return neighbour distance.
        """
        return self.__neighbour_distance

    def get_waypoint_from_ind(self, ind: Union[int, list, np.ndarray]) -> np.ndarray:
        """
        Return waypoint locations using ind.
        """
        return self.__waypoints[ind, :]

    def get_ind_from_waypoint(self, waypoint: np.ndarray) -> Union[int, np.ndarray, None]:
        """
        Args:
            waypoint: np.array([xp, yp, zp])
        Returns: index of the closest waypoint.
        """

        if len(waypoint) > 0:
            dm = waypoint.ndim
            if dm == 1:
                d = cdist(self.__waypoints, waypoint.reshape(1, -1))
                return np.argmin(d, axis=0)[0]
            elif dm == 2:
                d = cdist(self.__waypoints, waypoint)
                return np.argmin(d, axis=0)
            else:
                return None
        else:
            return None

    def get_ind_neighbours(self, ind: int) -> list:
        """
        Return all the neighbouring indices close to the given index.
        Args:
            ind: waypoint index
        Returns: neighbour indices
        """
        return self.__neighbour[ind]

    @staticmethod
    def get_vector_between_two_waypoints(wp1: np.ndarray, wp2: np.ndarray) -> np.ndarray:
        """ Get a vector from wp1 to wp2.

        Args:
            wp1: np.array([x1, y1, z1])
            wp2: np.array([x2, y2, z2])

        Returns:
            vec: np.array([[x2 - x1],
                           [y2 - y1],
                           [z2 - z1]])

        """
        dx = wp2[0] - wp1[0]
        dy = wp2[1] - wp1[1]
        dz = wp2[2] - wp1[2]
        vec = np.vstack((dx, dy, dz))
        return vec




