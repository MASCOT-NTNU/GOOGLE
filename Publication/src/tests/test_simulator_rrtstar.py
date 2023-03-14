"""
Unittest for simulator
"""
from unittest import TestCase
from Simulators.Simulator_RRTStar import SimulatorRRTStar


class TestSimulator(TestCase):

    def setUp(self) -> None:
        seed = 0
        debug = True
        appxoimate_eibv = False
        sigma = 1.
        nugget = .4
        neighbour_distance = 240
        fast_eibv = True
        self.s = SimulatorRRTStar(neighbour_distance=neighbour_distance, sigma=sigma, nugget=nugget, seed=seed,
                                  debug=debug, approximate_eibv=appxoimate_eibv, fast_eibv=fast_eibv)

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


