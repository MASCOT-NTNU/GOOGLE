from unittest import TestCase
from Experiment.EDA import EDA


class TestEDA(TestCase):

    def setUp(self) -> None:
        self.e = EDA()
        pass

    # def test_mission_recap(self):
    #     self.e.run_mission_recap()

    # def test_refine_grid(self) -> None:
    #     self.e.refine_grid()
    #     pass

    # def test_get_fields(self) -> None:
    #     self.e.get_fields4gis()

    # def test_save_prior(self) -> None:
    #     self.e.save_prior()

    def test_get_cost_valley(self) -> None:
        self.e.get_costvalley4gis()
        pass