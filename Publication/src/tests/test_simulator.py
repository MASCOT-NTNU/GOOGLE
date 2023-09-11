"""
Unit tests for the simulator module.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-09-06
"""

from unittest import TestCase
from Simulators.Simulator import Simulator


class TestSimulator(TestCase):

    def setUp(self) -> None:
        seed = 0
        debug = True
        weight_eibv = 2.
        weight_ivr = 0.
        self.simulator = Simulator(weight_eibv=weight_eibv, weight_ivr=weight_ivr,
                                   random_seed=seed, replicate_id=0, debug=debug)

    def test_run(self) -> None:
        self.simulator.run_myopic()
        self.simulator.run_rrt()
        pass
