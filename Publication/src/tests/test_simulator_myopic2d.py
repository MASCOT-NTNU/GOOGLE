"""
Unittest for simulator
"""
from unittest import TestCase
from Simulators.Simulator_Myopic2D import SimulatorMyopic2D


class TestSimulator(TestCase):

    def setUp(self) -> None:
        sigma = .1
        nugget = .01
        self.s = SimulatorMyopic2D(sigma=sigma, nugget=nugget, seed=0, debug=False)

    def test_agent_run(self):
        self.s.run_all(80)

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


