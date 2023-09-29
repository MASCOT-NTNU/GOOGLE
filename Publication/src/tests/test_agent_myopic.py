"""
Unittest for the agent

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-09-04
"""

from unittest import TestCase
from Agents.AgentMyopic import Agent


class TestAgent(TestCase):
    def setUp(self) -> None:
        seed = 0
        debug = True
        self.agent1 = Agent(weight_eibv=1., weight_ivr=.0, random_seed=seed, debug=debug, name="EIBV")
        # self.agent2 = Agent(weight_eibv=.0, weight_ivr=2., random_seed=seed, debug=debug, name="IVR")
        # self.agent3 = Agent(weight_eibv=1., weight_ivr=1., random_seed=seed, debug=debug, name="Equal")

    def test_run(self) -> None:
        self.agent1.run()
        # self.agent2.run(num_steps)
        # self.agent3.run(num_steps)

        import matplotlib.pyplot as plt
        from matplotlib.pyplot import get_cmap

        agents = [self.agent1, self.agent2, self.agent3]
        grid = self.agent1.grf.grid
        for agent in agents:
            mu_data = agent.mu_data
            sigma_data = agent.sigma_data
            mu_truth_data = agent.mu_truth_data

            for i in range(num_steps):
                plt.figure(figsize=(30, 10))
                plt.subplot(131)
                plt.scatter(grid[:, 1], grid[:, 0], c=mu_data[i, :],
                            cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
                plt.colorbar()
                plt.title("mu_data")

                plt.subplot(132)
                plt.scatter(grid[:, 1], grid[:, 0], c=mu_truth_data[i, :],
                            cmap=get_cmap("BrBG", 10), vmin=10, vmax=30)
                plt.colorbar()
                plt.title("mu_truth_data")

                plt.subplot(133)
                plt.scatter(grid[:, 1], grid[:, 0], c=sigma_data[i, :],
                            cmap=get_cmap("RdBu", 10), vmin=0, vmax=1)
                plt.colorbar()
                plt.title("sigma_data")

                plt.show()

        print("h")

