"""
Unit tests for the EDA simulator.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-09-05
"""

from unittest import TestCase
from Simulators.EDA_SIM import EDA


class TestEDA(TestCase):

    def setUp(self) -> None:
        self.e = EDA()

    def test_all(self) -> None:
        self.e.plot_metric_analysis()
        # self.e.plot_cost_components()
