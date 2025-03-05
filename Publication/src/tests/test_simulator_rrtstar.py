"""
Unittest for simulator RRTStar

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-09-05
"""
from unittest import TestCase
from Simulators.Simulator_RRTStar import SimulatorRRTStar


class TestSimulator(TestCase):

    def setUp(self) -> None:
        seed = 0
        debug = False
        appxoimate_eibv = False
        sigma = 1.
        nugget = .25
        neighbour_distance = 120
        fast_eibv = True
        self.s = SimulatorRRTStar(neighbour_distance=neighbour_distance, sigma=sigma, nugget=nugget, seed=seed,
                                  debug=debug, approximate_eibv=appxoimate_eibv, fast_eibv=fast_eibv)

    def test_agent_run(self):
        self.s.run_all(10)
        res_eibv = self.s.extract_data_for_agent(self.s.ag_eibv)
        res_ivr = self.s.extract_data_for_agent(self.s.ag_ivr)
        res_eq = self.s.extract_data_for_agent(self.s.ag_eq)

        self.s


