"""
Myopic2D Planner plans the next waypoint according to Sense, Plan, Act process.
It wraps all the essential components together to ease the procedure for the agent during adaptive sampling.
Myopic2D path planner determines the next waypoint according to minimum cost criterion by using the cost valley.
It utilizes the three-waypoint system to smooth out the planned trajectory.
- Previous waypoint: the previous location.
- Current waypoint: contains the current location, used to filter illegal next waypoints.
- Next waypoint: contains the next waypoint, and the AUV should go to next waypoint once it arrives at the current one.
"""
from CostValley.CostValley import CostValley
from Config import Config
from usr_func.is_list_empty import is_list_empty
from shapely.geometry import Point, LineString
import numpy as np
import os


class Myopic2D:
    """
    Myopic2D planner determines the next waypoint according to minimum EIBV criterion.
    """
    def __init__(self, weight_eibv: float = 1., weight_ivr: float = 1.) -> None:
        # set the directional penalty
        self.__config = Config()
        self.__directional_penalty = False
        print("Directional penalty: ", self.__directional_penalty)

        # s0: set up default environment
        self.__cost_valley = CostValley(weight_eibv=weight_eibv, weight_ivr=weight_ivr)
        self.__grf = self.__cost_valley.get_grf_model()
        self.__waypoint_distance = self.__config.get_waypoint_distance()
        self.__candidates_angle = np.linspace(0, 2 * np.pi, 7)

        # s1: add polygon border and polygon obstacles.
        self.__polygon_border = self.__config.get_polygon_border()
        self.__polygon_border_shapely = self.__config.get_polygon_border_shapely()
        self.__line_border_shapely = self.__config.get_line_border_shapely()
        self.__polygon_obstacle = self.__config.get_polygon_obstacle()
        self.__polygon_obstacle_shapely = self.__config.get_polygon_obstacle_shapely()
        self.__line_obstacle_shapely = self.__config.get_line_obstacle_shapely()

        # s2: set up trackers
        self.__wp_curr = self.__config.get_loc_start()
        self.__wp_prev = self.__wp_curr
        self.__wp_next = self.__wp_curr
        self.__loc_cand = None
        self.__trajectory = []
        self.__trajectory.append([self.__wp_curr[0], self.__wp_curr[1]])

    def update_next_waypoint(self, ctd_data: np.ndarray = None) -> np.ndarray:
        """
        Get pioneer waypoint index according to minimum EIBV criterion, which is only an integer.

        If no possible candidate locations were found. Then a random location in the neighbourhood is selected.
        Also the pioneer index can be modified here.

        Para:
            ctd_data: np.array([[timestamp, x, y, sal]])
        Returns:
            id_pioneer: designed pioneer waypoint index.
        """
        # s0: update grf kernel
        self.__grf.assimilate_temporal_data(ctd_data)
        self.__cost_valley.update_cost_valley(self.__wp_curr)

        # s1: find candidate locations
        # id_smooth, id_neighbours = self.get_candidates_indices()
        wp_smooth, wp_neighbours = self.get_candidates_waypoints()

        if not is_list_empty(wp_smooth):
            # get cost associated with those valid candidate locations.
            costs = []
            self.__loc_cand = wp_smooth
            for loc in self.__loc_cand:
                costs.append(self.__cost_valley.get_cost_at_location(loc))
            wp_next = wp_smooth[np.argmin(costs)]
        else:
            angles = np.linspace(0, 2 * np.pi, 61)
            for angle in angles:
                wp_next = self.__wp_curr + self.__waypoint_distance * np.array([np.sin(angle), np.cos(angle)])
                if self.is_location_legal(wp_next) and self.is_path_legal(self.__wp_curr, wp_next):
                    break

        self.__wp_next = wp_next
        self.__wp_prev = self.__wp_curr
        self.__wp_curr = self.__wp_next
        self.__trajectory.append([self.__wp_curr[0], self.__wp_curr[1]])
        return self.__wp_next

    def get_candidates_waypoints(self) -> tuple:
        """
        Filter sharp turn, bottom up and dive down behaviours.

        It computes the dot product between two vectors. One is from the previous waypoint to the current waypoint.
        The other is from the current waypoint to the potential next waypoint.

        Example:
            >>> wp_prev = np.array([x1, y1])
            >>> wp_curr = np.array([x2, y2])
            >>> vec1 = wp_curr - wp_prev
            >>> for wp in wp_neighbours:
            >>>     wp = np.array([xi, yi])
            >>>     vec2 = wp - wp_curr
            >>>     if dot(vec1.T, vec2) >= 0:
            >>>         append(wp)

        Returns:
            wp_smooth: filtered candidate waypoints.
            wp_neighbours: all the neighbours at the current location.
        """
        # s1: get vec from previous wp to current wp.
        vec1 = self.get_vector_between_two_waypoints(self.__wp_prev, self.__wp_curr)

        # s2: get all neighbour waypoints
        wp_neighbours = []
        wp_smooth = []
        for angle in self.__candidates_angle:
            wp_temp = self.__wp_curr + self.__waypoint_distance * np.array([np.sin(angle), np.cos(angle)])
            # s3: filter out illegal locations
            if self.is_location_legal(wp_temp) and self.is_path_legal(self.__wp_curr, wp_temp):
                wp_neighbours.append(wp_temp)
                vec2 = self.get_vector_between_two_waypoints(self.__wp_curr, wp_temp)
                if self.__directional_penalty:
                    if vec1.T @ vec2 >= 0:
                        wp_smooth.append(wp_temp)
                else:
                    wp_smooth.append(wp_temp)
        return wp_smooth, wp_neighbours

    def is_location_legal(self, loc: np.ndarray) -> bool:
        """
        Check if the location is legal.
        Args:
            loc: np.array([x, y])

        Returns:
            True if legal, False if illegal.
        """
        point = Point(loc[0], loc[1])
        if self.__polygon_border_shapely.contains(point) and \
                not self.__polygon_obstacle_shapely.contains(point):
            return True
        else:
            return False

    def is_path_legal(self, loc_start: np.ndarray, loc_end: np.ndarray) -> bool:
        """
        Check if the path is legal.
        Args:
            loc_start: np.array([x1, y1])
            loc_end: np.array([x2, y2])

        Returns:
            True if legal, False if illegal.
        """
        line = LineString([loc_start, loc_end])
        if self.__line_border_shapely.intersects(line) or self.__line_obstacle_shapely.intersects(line):
            return False
        else:
            return True

    def get_previous_waypoint(self) -> np.ndarray:
        """ Previous waypoint. """
        return self.__wp_prev

    def get_current_waypoint(self) -> np.ndarray:
        """ Current waypoint. """
        return self.__wp_curr

    def get_next_waypoint(self) -> np.ndarray:
        """ Next waypoint. """
        return self.__wp_next

    def get_trajectory(self) -> np.ndarray:
        """ Trajectory for myopic2d path planner. """
        return np.array(self.__trajectory)

    def getCostValley(self):
        """ Get the cost valley used in path planning. """
        return self.__cost_valley

    def get_loc_cand(self) -> np.ndarray:
        return self.__loc_cand

    @staticmethod
    def get_vector_between_two_waypoints(wp_start: np.ndarray, wp_end: np.ndarray) -> np.ndarray:
        """ Get a vector from wp_start to wp_end.

        Args:
            wp_start: np.array([x1, y1])
            wp_end: np.array([x2, y2])

        Returns:
            vec: np.array([[x2 - x1],
                           [y2 - y1]])

        """
        dx = wp_end[0] - wp_start[0]
        dy = wp_end[1] - wp_start[1]
        vec = np.vstack((dx, dy))
        return vec


if __name__ == "__main__":
    m = Myopic2D()