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
from usr_func.is_list_empty import is_list_empty
import numpy as np
import os


class Myopic2D:
    """
    Myopic2D planner determines the next waypoint according to minimum EIBV criterion.
    """
    def __init__(self, loc_start: np.ndarray, weight_eibv: float = 1., weight_ivr: float = 1.,
                 sigma: float = .1, nugget: float = .01) -> None:
        # s0: set up default environment
        self.__cost_valley = CostValley(weight_eibv=weight_eibv, weight_ivr=weight_ivr, sigma=sigma, nugget=nugget)
        self.__grf = self.__cost_valley.get_grf_model()
        self.__field = self.__grf.field

        # s1: set up trackers
        self.__wp_curr = loc_start
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
            ctd_data: np.array([[x, y, sal]])
        Returns:
            id_pioneer: designed pioneer waypoint index.
        """
        # s0: update grf kernel
        self.__grf.assimilate_data(ctd_data)
        self.__cost_valley.update_cost_valley()

        # s1: find candidate locations
        id_smooth, id_neighbours = self.get_candidates_indices()

        if not is_list_empty(id_smooth):
            # get cost associated with those valid candidate locations.
            costs = []
            self.__loc_cand = self.__field.get_location_from_ind(id_smooth)
            for loc in self.__loc_cand:
                costs.append(self.__cost_valley.get_cost_at_location(loc))
            id_next = id_smooth[np.argmin(costs)]
        else:
            rng_ind = np.random.randint(0, len(id_neighbours))
            id_next = id_neighbours[rng_ind]

        self.__wp_next = self.__field.get_location_from_ind(id_next)
        self.__wp_prev = self.__wp_curr
        self.__wp_curr = self.__wp_next
        self.__trajectory.append([self.__wp_curr[0], self.__wp_curr[1]])
        return self.__wp_next

    def get_candidates_indices(self) -> tuple:
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
            id_smooth: filtered candidate indices in the waypointgraph.
            id_neighbours: all the neighbours at the current location.
        """
        # s1: get vec from previous wp to current wp.
        vec1 = self.get_vector_between_two_waypoints(self.__wp_prev, self.__wp_curr)

        # s2: get all neighbours.
        id_neighbours = self.__field.get_neighbour_indices(self.__field.get_ind_from_location(self.__wp_curr))

        # s3: smooth neighbour locations.
        id_smooth = []
        for iid in id_neighbours:
            wp_n = self.__field.get_location_from_ind(iid)
            vec2 = self.get_vector_between_two_waypoints(self.__wp_curr, wp_n)
            if vec1.T @ vec2 >= 0:
                id_smooth.append(iid)
        return id_smooth, id_neighbours

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

