
class GOOGLE:

    def __init__(self):
        self.trajectory = []
        pass

    def pathplanner(self):
        self.gp = GPKernel()
        self.gp.setup()
        self.gp.getEIBVField()

        starting_loc = Location(1., .0)
        goal_loc = Location(.0, .0)
        self.gp.getBudgetField(goal_loc)

        ind_min_cost = np.argmin(self.gp.eibv)
        ending_loc = Location(self.gp.grid_vector[ind_min_cost, 0], self.gp.grid_vector[ind_min_cost, 1])
        # ending_loc = Location(.0, 1.)

        # plotf_vector(self.gp.grid_vector, self.gp.mu_truth, "Truth")
        # plt.show()
        #
        # plotf_vector(self.gp.grid_vector, self.gp.mu_prior_vector, "Prior")
        # plt.show()

        budget = BUDGET
        distance_travelled = 0
        current_loc = [starting_loc.x, starting_loc.y]
        self.trajectory.append([starting_loc.x, starting_loc.y])

        print("Total budget: ", budget)

        for i in range(NUM_STEPS):
            print("Step: ", i)

            # == path planning ==
            rrtconfig = RRTConfig(starting_location=starting_loc, ending_location=ending_loc,
                                  goal_sample_rate=GOAL_SAMPLE_RATE, step=STEP, GPKernel=self.gp,
                                  budget=budget, goal_location=goal_loc)
            self.rrt = RRTStar(rrtconfig)
            self.rrt.expand_trees()
            self.rrt.get_shortest_path()
            path = self.rrt.path


            # == plotting ==
            fig = plt.figure(figsize=(20, 5))
            gs = GridSpec(nrows=1, ncols=4)
            ax = fig.add_subplot(gs[0])
            cmap = get_cmap("RdBu", 10)
            plotf_vector(self.gp.grid_vector, self.gp.mu_truth, "Ground Truth", cmap=cmap)

            ax = fig.add_subplot(gs[1])
            plotf_vector(self.gp.grid_vector, self.gp.mu_cond, "Conditional Mean", cmap=cmap)
            plotf_trajectory(self.trajectory)

            ax = fig.add_subplot(gs[2])
            plotf_vector(self.gp.grid_vector, np.sqrt(np.diag(self.gp.Sigma_cond)), "Prediction Error", cmap=cmap)
            plotf_trajectory(self.trajectory)

            ax = fig.add_subplot(gs[3])
            self.rrt.plot_tree()
            plotf_vector(self.gp.grid_vector, .6 * self.gp.eibv + .45 * self.gp.budget_field, "EIBV cost valley + Budget Radar", alpha=.1, cmap=cmap)
            plotf_budget_radar([goal_loc.x, goal_loc.y], budget)
            # plt.show()
            plt.savefig(FIGPATH + "P_{:03d}.png".format(i))
            plt.close("all")
            # == end plotting ==


            next_loc = path[-2, :]
            self.trajectory.append(next_loc)
            distance_travelled = np.sqrt((current_loc[0] - next_loc[0]) ** 2 +
                                         (current_loc[1] - next_loc[1]) ** 2)
            current_loc = next_loc

            budget = budget - distance_travelled
            print("Budget left: ", budget)

            ind_F = self.gp.getIndF(next_loc[0], next_loc[1])
            F = np.zeros([1, self.gp.grid_vector.shape[0]])
            F[0, ind_F] = True
            self.gp.mu_cond, self.gp.Sigma_cond = self.gp.GPupd(self.gp.mu_cond, self.gp.Sigma_cond, F,
                                                                self.gp.R, F @ self.gp.mu_truth)
            self.gp.getEIBVField()
            self.gp.getBudgetField(goal_loc)

            starting_loc = Location(next_loc[0], next_loc[1])
            node_start = TreeNode(starting_loc)
            node_end = TreeNode(ending_loc)
            if RRTStar.get_distance_between_nodes(node_start, node_end) < DISTANCE_TOLERANCE:
                print("Arrived")

            ind_min_cost = np.argmin(.6 * self.gp.eibv + .45 * self.gp.budget_field)
            ending_loc = Location(self.gp.grid_vector[ind_min_cost, 0], self.gp.grid_vector[ind_min_cost, 1])

            #     starting_loc = Location(.0, 1.)
            #     ending_loc = Location(1., 1.)


        pass