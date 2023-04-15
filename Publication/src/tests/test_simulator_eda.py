from unittest import TestCase
from Simulators.EDA import EDA


class TestEDA(TestCase):

    def setUp(self) -> None:
        self.e = EDA()

    def test_all(self) -> None:
        # self.e.plot_metric_analysis()
        self.e.plot_cost_components()
