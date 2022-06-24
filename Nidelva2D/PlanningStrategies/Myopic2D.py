"""
This script generates the next waypoint based on the current knowledge and previous path
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-06-22
"""

"""
Usage:
loc_next = MyopicPlanning2D(Knowledge).next_waypoint
"""


from usr_func import get_ibv, vectorise, time, np, pd, plt
import pickle
from GOOGLE.Nidelva2D.Config.Config import THRESHOLD, FILEPATH
from GOOGLE.Nidelva2D.GRF import GRF


class MyopicPlanning2D:

    def __init__(self, grf_model=None, waypoint_graph=None, hash_neighbours=None, hash_waypoint2grf=None, echo=False):
        self.grf_model = grf_model
        self.waypoint_graph = waypoint_graph
        self.hash_neighbours = hash_neighbours
        self.hash_waypoint2grf = hash_waypoint2grf
        self.echo = echo
        print("Myopic2D path planner is ready!")

    def update_legal_indices(self, ind_legal=None):
        if ind_legal is None:
            ind_legal = []
        self.ind_legal = ind_legal

    def find_next_waypoint_using_min_eibv(self, ind_current=None, ind_previous=None, ind_visited=None):
        self.ind_current = ind_current
        self.ind_previous = ind_previous
        self.ind_visited = ind_visited

        t1 = time.time()
        self.find_all_neighbours()
        t2 = time.time()
        if self.echo:
            print("Time consumed for neighbour: ", t2 - t1)
        t1 = time.time()
        self.smooth_filter_neighbours()
        t2 = time.time()
        if self.echo:
            print("Time consumed for smoothing: ", t2 - t1)

        self.EIBV = []
        t1 = time.time()
        for ind_candidate in self.ind_candidates:  # don't need parallel, since candidate number is small, too slow to run mp
            self.EIBV.append(self.get_eibv_from_grf_model(self.hash_waypoint2grf[ind_candidate]))
        if self.EIBV:
            self.ind_next = self.ind_candidates[np.argmin(self.EIBV)]
        else:
            if len(self.ind_neighbours) > 1:
                self.ind_next = self.ind_neighbours[np.random.randint(len(self.ind_neighbours))]
            else:
                print("WARN")

        t2 = time.time()
        print("Path planning takes: ", t2 - t1)
        return self.ind_next

    def find_all_neighbours(self):
        self.ind_neighbours = self.hash_neighbours[self.ind_current]
        self.ind_neighbours = list(set(self.ind_neighbours).intersection(self.ind_legal))
        print("legal neighbours: ", self.ind_neighbours)

    def smooth_filter_neighbours(self):
        vec1 = self.get_vec_from_indices(self.ind_previous, self.ind_current)
        self.ind_candidates = []
        for i in range(len(self.ind_neighbours)):
            ind_candidate = self.ind_neighbours[i]
            if not ind_candidate in self.ind_visited:
                vec2 = self.get_vec_from_indices(self.ind_current, ind_candidate)
                if np.dot(vec1.T, vec2) >= 0:
                    self.ind_candidates.append(ind_candidate)

    def get_vec_from_indices(self, ind_start, ind_end):
        x_start = self.waypoint_graph[ind_start, 0]
        y_start = self.waypoint_graph[ind_start, 1]
        x_end = self.waypoint_graph[ind_end, 0]
        y_end = self.waypoint_graph[ind_end, 1]

        dx = x_end - x_start
        dy = y_end - y_start
        return vectorise([dx, dy])

    def get_eibv_from_grf_model(self, ind_candidate):
        t1 = time.time()
        sigma_diag = self.grf_model.get_posterior_variance_at_ind(ind_candidate)
        t2 = time.time()
        if self.echo:
            print("get post variance take: ", t2-t1)
        t1 = time.time()
        eibv = get_ibv(self.grf_model.mu_cond, sigma_diag, THRESHOLD)
        t2 = time.time()
        if self.echo:
            print("eibv takes; ", t2 - t1)
        return eibv

    def check_mp(self):
        self.grf_model = GRF()
        self.waypoint_graph = pd.read_csv(FILEPATH + "Config/WaypointGraph.csv").to_numpy()
        f_neighbour = open(FILEPATH + "Config/HashNeighbours.p", 'rb')
        self.hash_neighbours = pickle.load(f_neighbour)
        f_neighbour.close()

        f2 = open(FILEPATH + "Config/HashWaypoint2GRF.p", 'rb')
        f2.close()
        self.hash_waypoint2grf = pickle.load(f2)

        i_now = int(self.waypoint_graph.shape[0]/2) - 190
        i_prev = i_now - 1
        ind_legal = np.arange(i_now)
        self.update_legal_indices(ind_legal=ind_legal)

        ind = self.find_next_waypoint_using_min_eibv(i_now, i_prev, [])
        print("ind_neighbour: ", self.ind_neighbours)

        x = self.waypoint_graph[:, 0]
        y = self.waypoint_graph[:, 1]
        plt.plot(x, y, 'k.', alpha=.3)
        plt.plot(x[i_now], y[i_now], 'b.', markersize=20)
        plt.plot(x[i_prev], y[i_prev], 'y.', markersize=20)
        plt.plot(x[self.ind_candidates], y[self.ind_candidates], 'g.', markersize=20)
        plt.plot(x[ind], y[ind], 'r.', markersize=20)
        plt.plot(x[ind_legal], y[ind_legal], 'c*', alpha=.2)
        plt.plot(x[self.ind_neighbours], y[self.ind_neighbours], 'c^', alpha=.1, markersize=30)
        plt.show()


if __name__ == "__main__":
    mp = MyopicPlanning2D()
    mp.check_mp()


