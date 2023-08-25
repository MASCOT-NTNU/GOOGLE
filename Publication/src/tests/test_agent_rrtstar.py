"""
Unittest for the agent RRTStar

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-08-24
"""

from unittest import TestCase
from Agents.AgentRRTStar import Agent


class TestAgent(TestCase):
    def setUp(self) -> None:
        seed = 0
        debug = True
        approximate_eibv = False
        budget_mode = False
        fast_eibv = True
        sigma = .5
        nugget = .01
        neighbour_distance = 120
        self.agent1 = Agent(neighbour_distance=neighbour_distance, weight_eibv=1.99, weight_ivr=.01,
                            sigma=sigma, nugget=nugget, random_seed=seed,
                            debug=debug, name="EIBV", approximate_eibv=approximate_eibv, fast_eibv=fast_eibv)
        self.agent2 = Agent(neighbour_distance=neighbour_distance, weight_eibv=.01, weight_ivr=1.99,
                            budget_mode=budget_mode, sigma=sigma, nugget=nugget, random_seed=seed,
                            debug=debug, name="IVR", approximate_eibv=approximate_eibv, fast_eibv=fast_eibv)
        self.agent3 = Agent(neighbour_distance=neighbour_distance, weight_eibv=1., weight_ivr=1.,
                            budget_mode=budget_mode, sigma=sigma, nugget=nugget, random_seed=seed,
                            debug=debug, name="Equal", approximate_eibv=approximate_eibv, fast_eibv=fast_eibv)

    def test_run(self) -> None:
        num_steps = 50
        self.agent1.run(num_steps)
        self.agent2.run(num_steps)
        self.agent3.run(num_steps)

        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(self.agent1.ibv, label="EIBV")
        plt.plot(self.agent2.ibv, label="IVR")
        plt.plot(self.agent3.ibv, label="EQUAL")
        plt.legend()
        plt.xlabel("Steps")
        plt.ylabel("IBV")
        # plt.savefig("/Users/yaolin/Downloads/small_coef.png")
        plt.show()
        print("hello")
        plt.show()