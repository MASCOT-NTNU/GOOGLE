"""
Unittest for simulator myopic

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-09-05
"""
from unittest import TestCase
from Simulators.Simulator_Myopic2D import SimulatorMyopic2D


class TestSimulator(TestCase):

    def setUp(self) -> None:
        sigma = 1.
        nugget = .25
        neighbour_distance = 120
        approximate_eibv = False
        fast_eibv = True
        debug = False
        directional_penalty = False
        self.s = SimulatorMyopic2D(neighbour_distance=neighbour_distance, sigma=sigma, nugget=nugget, seed=0,
                                   debug=debug, approximate_eibv=approximate_eibv, fast_eibv=fast_eibv,
                                   directional_penalty=directional_penalty)

    def test_agent_run(self):
        self.s.run_all(10)

        res_eibv = self.s.extract_data_for_agent(self.s.ag_eibv)
        res_ivr = self.s.extract_data_for_agent(self.s.ag_ivr)
        res_eq = self.s.extract_data_for_agent(self.s.ag_eq)

        self.s

