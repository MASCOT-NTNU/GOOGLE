"""
Unittest for simulator
"""
from unittest import TestCase
from Simulators.Simulator_Myopic2D import Simulator


class TestSimulator(TestCase):

    def setUp(self) -> None:
        self.s = Simulator(0, False)

    def test_agent_run(self):
        self.s.run_all(40)


