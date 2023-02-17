"""
Unittest for simulator
"""
from unittest import TestCase
from Simulators.Simulator_EIBV import SimulatorEIBV


class TestSimulator(TestCase):

    def setUp(self) -> None:
        sigma = 1.
        nugget = .4
        self.s = SimulatorEIBV(sigma=sigma, nugget=nugget, seed=0, debug=True)

    def test_agent_run(self):
        self.s.run_all(50)


