"""
Unittest for the agent
"""

from unittest import TestCase
from Agents.AgentRRTStar import Agent


class TestAgent(TestCase):
    def setUp(self) -> None:
        seed = 0
        debug = True
        approximate_eibv = False
        sigma = 1.
        nugget = .4
        self.agent1 = Agent(weight_eibv=1.99, weight_ivr=.01, sigma=sigma, nugget=nugget,
                            random_seed=seed, debug=debug, name="EIBV", approximate_eibv=approximate_eibv)
        self.agent2 = Agent(weight_eibv=.01, weight_ivr=1.99, sigma=sigma, nugget=nugget,
                            random_seed=seed, debug=debug, name="IVR", approximate_eibv=approximate_eibv)
        self.agent3 = Agent(weight_eibv=1., weight_ivr=1., sigma=sigma, nugget=nugget,
                            random_seed=seed, debug=debug, name="Equal", approximate_eibv=approximate_eibv)

    def test_run(self) -> None:
        num_steps = 4
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