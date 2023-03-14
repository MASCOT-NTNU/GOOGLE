"""
Unittest for simulator
"""
from unittest import TestCase
from Simulators.Simulator_Myopic2D import SimulatorMyopic2D


class TestSimulator(TestCase):

    def setUp(self) -> None:
        sigma = 1.
        nugget = .4
        neighbour_distance = 240
        approximate_eibv = False
        fast_eibv = True
        debug = False
        directional_penalty = False
        self.s = SimulatorMyopic2D(neighbour_distance=neighbour_distance, sigma=sigma, nugget=nugget, seed=0,
                                   debug=debug, approximate_eibv=approximate_eibv, fast_eibv=fast_eibv,
                                   directional_penalty=directional_penalty)

    def test_agent_run(self):
        self.s.run_all(10)

        import matplotlib.pyplot as plt
        res_eibv = self.s.extract_data_for_agent(self.s.ag_eibv)
        res_ivr = self.s.extract_data_for_agent(self.s.ag_ivr)
        res_eq = self.s.extract_data_for_agent(self.s.ag_eq)
        plt.plot(res_eibv[1], label="EIBV")
        plt.plot(res_ivr[1], label="IVR")
        plt.plot(res_eq[1], label="EQ")
        plt.legend()
        plt.show()

        self.s


