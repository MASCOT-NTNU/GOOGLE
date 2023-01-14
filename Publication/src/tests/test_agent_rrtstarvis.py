"""
Unittest for the agent
"""

from unittest import TestCase
from Agents.AgentRRTStarVis import Agent


class TestAgent(TestCase):
    def setUp(self) -> None:
        seed = 0
        debug = True
        sigma = .1
        nugget = .01
        self.agent = Agent(weight_eibv=1., weight_ivr=1., sigma=sigma, nugget=nugget,
                           random_seed=seed, debug=debug, name="Equal", budget_mode=True)

    def test_run(self) -> None:
        num_steps = 50
        self.agent.run(num_steps)
