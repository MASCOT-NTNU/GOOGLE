

class RRTStar:

    obstacles = np.array(OBSTACLES)
    polygon_obstacles = []

    def __init__(self, config=None):
        self.nodes = []
        self.config = config
        self.path = []
        self.starting_node = TreeNode(self.config.starting_location, None, 0,
                                      self.config.GPKernel.mu_prior_vector, self.config.GPKernel.Sigma_prior)
        self.ending_node = TreeNode(self.config.ending_location, None, 0)
        self.starting_node.EIBV = 0
        self.setF(self.starting_node)
        self.setF(self.ending_node)
        self.counter_fig = 0
        pass

    def setF(self, node):
        node.F = self.config.GPKernel.getIndF(node.location.x, node.location.y)

    def expand_trees(self):
        self.nodes.append(self.starting_node)
        t1 = time.time()
        for i in range(MAXNUM):
            # print("Iteration: ", i)
            if np.random.rand() <= self.config.goal_sample_rate:
                new_location = self.config.ending_location
            else:
                new_location = self.get_new_location()

            nearest_node = self.get_nearest_node(self.nodes, new_location)
            next_node = self.get_next_node(nearest_node, new_location)

            if self.isWithin(next_node):
                # print("Collision zone")
                continue

            next_node, nearest_node = self.rewire_tree(next_node, nearest_node)

            if self.isIntersect(nearest_node, next_node):
                continue

            if self.isarrived(next_node):
                self.ending_node.parent = next_node
            else:
                self.nodes.append(next_node)
            pass

            # plt.figure()
            # for j in range(self.obstacles.shape[0]):
            #     obstacle = np.append(self.obstacles[j], self.obstacles[j][0, :].reshape(1, -1), axis=0)
            #     plt.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')
            #
            # for node in self.nodes:
            #     if node.parent is not None:
            #         plt.plot([node.location.x, node.parent.location.x],
            #                  [node.location.y, node.parent.location.y], "-g")
            # # path = np.array(self.path)
            # # plt.plot(path[:, 0], path[:, 1], "-r")
            # plt.plot(self.config.starting_location.x, self.config.starting_location.y, 'k*', ms=10)
            # plt.plot(self.config.ending_location.x, self.config.ending_location.y, 'g*', ms=10)
            # plt.grid()
            # plt.savefig(FIGPATH + "T_{:04d}.png".format(self.counter_fig))
            # self.counter_fig += 1
            # # plt.show()
            # plt.close("all")
        t2 = time.time()
        print("Time consumed: ", t2 - t1)
        pass

    @staticmethod
    def get_new_location():
        x = np.random.uniform(XLIM[0], XLIM[1])
        y = np.random.uniform(YLIM[0], YLIM[1])
        location = Location(x, y)
        return location

    def get_nearest_node(self, nodes, location):
        dist = []
        node_new = TreeNode(location)
        for node in nodes:
            dist.append(self.get_distance_between_nodes(node, node_new))
        return nodes[dist.index(min(dist))]

    def get_next_node(self, node, location):
        node_temp = TreeNode(location)
        if RRTStar.get_distance_between_nodes(node, node_temp) <= self.config.step:
            return TreeNode(location, node)
        else:
            angle = np.math.atan2(location.y - node.location.y, location.x - node.location.x)
            x = node.location.x + self.config.step * np.cos(angle)
            y = node.location.y + self.config.step * np.sin(angle)
            location_next = Location(x, y)
        return TreeNode(location_next, node)

    @staticmethod
    def get_distance_between_nodes(node1, node2):
        dist_x = node1.location.x - node2.location.x
        dist_y = node1.location.y - node2.location.y
        dist = np.sqrt(dist_x ** 2 + dist_y ** 2)
        return dist

    def rewire_tree(self, node_current, node_nearest):
        ind_neighbour_nodes = self.get_neighbour_nodes(node_current)

        for i in range(len(ind_neighbour_nodes)):
            if self.get_cost_between_nodes(self.nodes[ind_neighbour_nodes[i]], node_current) < \
                    self.get_cost_between_nodes(node_nearest, node_current):
                node_nearest = self.nodes[ind_neighbour_nodes[i]]

            node_current.parent = node_nearest
            node_current.cost = self.get_cost_between_nodes(node_nearest, node_current)

        for i in range(len(ind_neighbour_nodes)):
            cost_current_neighbour = self.get_cost_between_nodes(node_current, self.nodes[ind_neighbour_nodes[i]])
            if cost_current_neighbour < self.nodes[ind_neighbour_nodes[i]].cost:
                self.nodes[ind_neighbour_nodes[i]].cost = cost_current_neighbour
                self.nodes[ind_neighbour_nodes[i]].parent = node_current
        return node_current, node_nearest

    def get_neighbour_nodes(self, node_current):
        distance_between_nodes = []
        for i in range(len(self.nodes)):
            if self.isIntersect(self.nodes[i], node_current):
                distance_between_nodes.append(np.inf)
            else:
                distance_between_nodes.append(RRTStar.get_distance_between_nodes(self.nodes[i], node_current))
        ind_neighbours = np.where(np.array(distance_between_nodes) <= RADIUS_NEIGHBOUR)[0]
        return ind_neighbours

    def get_cost_between_nodes(self, node1, node2):
        cost = (node1.cost +
                self.get_distance_between_nodes(node1, node2) +
                self.get_reward_between_nodes(node1, node2) +
                self.get_cost_according_to_budget(node1, node2))
               # RRTStar.get_cost_along_path(node1.location, node2.location)
        # print("Cost: ", cost)
        return cost

    def get_reward_between_nodes(self, node1, node2):
        self.setF(node2)
        node1.EIBV = self.get_eibv_for_node(node1)
        node2.EIBV = self.get_eibv_for_node(node2)
        reward = (node1.EIBV + node2.EIBV) / 2 * RRTStar.get_distance_between_nodes(node1, node2)
        return reward

    def get_eibv_for_node(self, node):
        return self.config.GPKernel.eibv[node.F]

    def get_cost_according_to_budget(self, node1, node2):
        N = 10
        x = np.linspace(node1.location.x, node2.location.x, N)
        y = np.linspace(node1.location.y, node2.location.y, N)
        cost = []
        for i in range(N):
            cost.append(self.get_cost_from_budget_field(Location(x[i], y[i])))
        cost_total = np.trapz(cost) / N
        # print("Cost total: ", cost_total)
        # cost1 = self.get_cost_from_budget_field(node1.location)
        # cost2 = self.get_cost_from_budget_field(node2.location)
        # cost_total = (cost1 + cost2) / 2 * RRTStar.get_distance_between_nodes(node1, node2)
        return cost_total

    def get_cost_from_budget_field(self, location):
        dist = np.sqrt((location.x - self.config.goal_location.x) ** 2 +
                       (location.y - self.config.goal_location.y) ** 2)
        return dist

    @staticmethod
    def get_cost_along_path(location1, location2):
        N = 10
        x = np.linspace(location1.x, location2.x, N)
        y = np.linspace(location1.y, location2.y, N)
        cost = []
        for i in range(N):
            cost.append(RRTStar.get_cost_from_field(Location(x[i], y[i])))
        cost_total = np.trapz(cost) / N
        # cost_total = np.sum(cost) / RRTStar.get_distance_between_nodes(TreeNode(location1), TreeNode(location2))
        # print("cost total: ", cost_total)
        return cost_total

    @staticmethod
    def get_cost_from_field(location):
        # return 0
        return 2**2 * np.exp(-((location.x - .5) ** 2 + (location.y - .5) ** 2) / .09)


    def isarrived(self, node):
        dist = self.get_distance_between_nodes(self.ending_node, node)
        if dist < DISTANCE_TOLERANCE:
            return True
        else:
            return False

    def set_obstacles(self):
        for i in range(self.obstacles.shape[0]):
            self.polygon_obstacles.append(Polygon(list(map(tuple, self.obstacles[i]))))

    '''
    Collision detection
    '''
    def isWithin(self, node):
        point = Point(node.location.x, node.location.y)
        within = False
        for i in range(len(self.polygon_obstacles)):
            if self.polygon_obstacles[i].contains(point):
                within = True
        return within

    def isIntersect(self, node1, node2):
        line = LineString([(node1.location.x, node1.location.y),
                           (node2.location.x, node2.location.y)])
        intersect = False
        for i in range(len(self.polygon_obstacles)):
            if self.polygon_obstacles[i].intersects(line):
                intersect = True
        return intersect
    '''
    End of collision detection
    '''

    def get_shortest_path(self):
        self.path.append([self.ending_node.location.x, self.ending_node.location.y])
        pointer_node = self.ending_node
        while pointer_node.parent is not None:

            node = pointer_node.parent
            self.path.append([node.location.x, node.location.y])
            pointer_node = node

        # self.path.append([self.starting_node.location.x, self.starting_node.location.y])
        self.path = np.array(self.path)

    def plot_tree(self):

        # plt.figure()
        if np.any(self.obstacles):
            for i in range(len(self.obstacles)):
                obstacle = np.append(self.obstacles[i], self.obstacles[i][0, :].reshape(1, -1), axis=0)
                plt.plot(obstacle[:, 0], obstacle[:, 1], 'r-.')

        for node in self.nodes:
            if node.parent is not None:
                plt.plot([node.location.x, node.parent.location.x],
                         [node.location.y, node.parent.location.y], "-g")

        path = np.array(self.path)
        plt.plot(path[:, 0], path[:, 1], "-r")
        plt.plot(self.config.starting_location.x, self.config.starting_location.y, 'k*', ms=10)
        plt.plot(self.config.ending_location.x, self.config.ending_location.y, 'g*')
        plt.grid()
        plt.title("rrt_star")
        # plt.savefig(FIGPATH + "T_{:04d}.png".format(self.counter_fig))
        # plt.show()
