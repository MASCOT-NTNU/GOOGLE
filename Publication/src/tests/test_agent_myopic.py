"""
Unittest for the agent
"""

from unittest import TestCase
from Agents.AgentMyopic import Agent


class TestAgent(TestCase):
    def setUp(self) -> None:
        seed = 0
        debug = True
        self.agent1 = Agent(1.99, .01, seed, debug, "EIBV")
        self.agent2 = Agent(.01, 1.99, seed, debug, "IVR")
        self.agent3 = Agent(1., 1., seed, debug, "Equal")

    def test_run(self) -> None:
        num_steps = 50
        self.agent1.run(num_steps)
        self.agent2.run(num_steps)
        self.agent3.run(num_steps)
