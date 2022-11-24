"""
Budget module computes the penalty for each location within the field using the polynomial penalty function given
the polygon formed by the remaining budget ellipse.
The remaining budget ellipse is formed from the current location to the home location with the long-axis to be equal
double the distance of the budget and the vertical axis to be equal sqrt(a**2-c**2), where c is the distance from
current location to the home location.

This module has two main functions:
- Compute the remaining budget ellipse.
    - This is later used for the rrtstar planning or other planning algorithms as a filtering process to remove illegal
    waypoints.
- Compute the budget field.
    - This can be used to get the global minimum cost location so to set the next desired location to be wp_min_cost.

:param
- MARGIN: defines the minimum vertical distance to stop using rrt*
"""
from Config import Config
import numpy as np
from matplotlib.patches import Ellipse
from shapely.geometry import Polygon, LineString, Point
import math


class Budget:
    __config = Config

    # initial values
    __MARGIN = 100  # when ellipse b is smaller than this, should go home.
    __grid = None
    __budget = 100000  # metres for the operation in the sea.
    __goal = __config.get_loc_end()
    __x_now, __y_now = __config.get_loc_start()
    __x_prev, __y_prev = __config.get_loc_start()
    __budget_field = None

    # ellipse parameters
    __ellipse_a = .0
    __ellipse_b = .0
    __ellipse_c = .0
    __ellipse_middle_x = .0
    __ellipse_middle_y = .0
    __ellipse_angle = .0
    __ellipse = None

    # shapely object
    __polygon_ellipse = None
    __line_ellipse = None

    # practical matters
    __go_home = False
    __border_warning = False

    def __init__(self, grid):
        self.__grid = grid

    def get_budget_field(self, x_now: float, y_now: float) -> np.ndarray:
        self.__x_now = x_now
        self.__y_now = y_now
        dist = np.sqrt((self.__x_now - self.__x_prev)**2 +
                       (self.__y_now - self.__y_prev)**2)
        self.__budget -= dist
        self.__update_budget_ellipse()
        self.__budget_field = np.zeros_like(self.__grid[:, 0])

        xg = self.__grid[:, 0] - self.__ellipse_middle_x
        yg = self.__grid[:, 1] - self.__ellipse_middle_y
        xr = xg * np.cos(self.__ellipse_angle) - yg * np.sin(self.__ellipse_angle)
        yr = xg * np.sin(self.__ellipse_angle) + yg * np.cos(self.__ellipse_angle)
        if not self.__go_home:
            u = (xr / self.__ellipse_b) ** 2 + (yr / self.__ellipse_a) ** 2
        else:
            u = np.ones_like(xr) * np.inf

        for i in range(self.__grid.shape[0]):
            point = Point(self.__grid[i, 0], self.__grid[i, 1])
            if self.__polygon_ellipse.contains(point):
                self.__budget_field[i] = 0
            else:
                self.__budget_field[i] = u[i] ** 2

        if np.amax(self.__budget_field) > 1:  # update border warning so to generate trees within ellipse.
            self.__border_warning = True

        self.__x_prev = self.__x_now
        self.__y_prev = self.__y_now
        return self.__budget_field

    def __update_budget_ellipse(self):
        self.__ellipse_middle_x = (self.__x_now + self.__goal[0]) / 2
        self.__ellipse_middle_y = (self.__y_now + self.__goal[1]) / 2
        dx = self.__goal[0] - self.__x_now
        dy = self.__goal[1] - self.__y_now
        self.__ellipse_angle = np.math.atan2(dx, dy)  # dx: vertical increment, dy: lateral increment
        self.__ellipse_a = self.__budget / 2
        self.__ellipse_c = np.sqrt(dx ** 2 + dy ** 2) / 2
        if self.__ellipse_a > self.__ellipse_c + self.__MARGIN:
            self.__ellipse_b = np.sqrt(self.__ellipse_a ** 2 - self.__ellipse_c ** 2)
            self.__ellipse = Ellipse(xy=(self.__ellipse_middle_y, self.__ellipse_middle_x), width=2*self.__ellipse_a,
                                     height=2*self.__ellipse_b, angle=math.degrees(self.__ellipse_angle))
            self.vertices = self.__ellipse.get_verts()
            # self.__polygon_ellipse = Polygon(self.vertices)
            # self.__line_ellipse = LineString(self.vertices)
            self.__polygon_ellipse = Polygon(np.fliplr(self.vertices))  # when uses NED system.
            self.__line_ellipse = LineString(np.fliplr(self.vertices))
        else:  # TODO: remove budget calculation, when it is too small, should only use straight line planning.
            self.__ellipse_b = 0
            self.__ellipse = Ellipse(xy=(self.__ellipse_middle_y, self.__ellipse_middle_x), width=2*self.__ellipse_a,
                                     height=2*self.__ellipse_b, angle=math.degrees(self.__ellipse_angle))
            self.vertices = self.__ellipse.get_verts()
            self.__polygon_ellipse = Polygon([])
            self.__line_ellipse = LineString([])
            self.__go_home = True

    def set_budget(self, value: float) -> None:
        self.__budget = value

    def set_goal(self, loc: np.ndarray) -> None:
        self.__goal = loc

    def set_loc_prev(self, loc: np.ndarray) -> None:
        self.__x_prev, self.__y_prev = loc

    def get_loc_now(self) -> np.ndarray:
        return np.array([self.__x_now, self.__y_now])

    def get_loc_prev(self) -> np.ndarray:
        return np.array([self.__x_prev, self.__y_prev])

    def get_ellipse_a(self) -> float:
        return self.__ellipse_a

    def get_ellipse_b(self) -> float:
        return self.__ellipse_b

    def get_ellipse_c(self) -> float:
        return self.__ellipse_c

    def get_ellipse(self) -> 'Ellipse':
        return self.__ellipse

    def get_ellipse_middle_location(self) -> np.ndarray:
        return np.array([self.__ellipse_middle_x, self.__ellipse_middle_y])

    def get_ellipse_rotation_angle(self) -> float:
        return self.__ellipse_angle

    def get_budget(self) -> float:
        return self.__budget

    def get_goal(self) -> np.ndarray:
        return self.__goal

    def get_polygon_ellipse(self) -> 'Polygon':
        return self.__polygon_ellipse

    def get_line_ellipse(self) -> 'LineString':
        return self.__line_ellipse

    def get_go_home_alert(self) -> bool:
        return self.__go_home

    def get_border_warning(self) -> bool:
        return self.__border_warning


if __name__ == "__main__":
    grid = np.random.rand(10, 2)
    b = Budget(grid)

