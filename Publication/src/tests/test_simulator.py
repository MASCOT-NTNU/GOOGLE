"""
Unittest for simulator
"""
from unittest import TestCase
from Simulators.Simulator import Simulator


class TestSimulator(TestCase):

    def setUp(self) -> None:
        self.s = Simulator(weight_eibv=.1, weight_ivr=1.9, case="ivr")

    def test_agent_run(self):
        self.s.run_replicates()
        # self.s.run_simulator()




