from unittest import TestCase
from Simulators.Simulator_Myopic2D import SimulatorMyopic2D
from Simulators.Simulator_RRTStar import SimulatorRRTStar


class TestSimulator(TestCase):

    def setUp(self) -> None:
        sigma = .1
        nugget = .01
        self.sm = SimulatorMyopic2D(sigma=sigma, nugget=nugget, seed=0, debug=False)
        self.sr = SimulatorRRTStar(sigma=sigma, nugget=nugget, seed=0, debug=False)

    def test_agent_run(self):
        self.sm.run_all(1)
        self.sr.run_all(1)

        res_eibv_myopic = self.sm.extract_data_for_agent(self.sm.ag_eibv)
        # res_ivr = self.sm.extract_data_for_agent(self.sm.ag_ivr)
        # res_eq = self.sm.extract_data_for_agent(self.sm.ag_eq)

        res_eibv_rrt = self.sr.extract_data_for_agent(self.sr.ag_eibv)
        # res_ivr = self.sm.extract_data_for_agent(self.sm.ag_ivr)
        # res_eq = self.sm.extract_data_for_agent(self.sm.ag_eq)


        print("hello")

        self.sm.run_all(1)



