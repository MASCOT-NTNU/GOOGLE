"""
Unittest for simulator
"""
from unittest import TestCase
from Simulators.Simulator import Simulator


class TestSimulator(TestCase):

    def setUp(self) -> None:
        self.s = Simulator(weight_eibv=.1, weight_ivr=1.9, case="ivr", debug=True)

    def test_agent_run(self):
        t, l = self.s.run_simulator(5)
        import matplotlib.pyplot as plt
        plt.plot(l.rmse); plt.title("rmse"); plt.show()
        plt.plot(l.ibv);
        plt.title("ibv")
        plt.show()
        plt.plot(l.vr);
        plt.title("vr")
        plt.show()
        plt.show()



