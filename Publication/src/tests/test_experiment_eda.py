from unittest import TestCase
from Experiment.EDA import EDA


class TestEDA(TestCase):

    def setUp(self) -> None:
        self.e = EDA()
        pass

    # def test_get_crossplot(self) -> None:
    #     self.e.get_crossplot_between_auv_and_sinmod()
    #     pass

    # def test_get_current_loc(self) -> None:
    #     self.e.get_current_location()

    # def test_refine_values(self) -> None:
    #     self.e.refine_values4gis()

    # def test_mission_recap(self):
    #     self.e.run_mission_recap()

    # def test_refine_grid(self) -> None:
    #     self.e.refine_grid()
    #     pass

    # def test_get_fields(self) -> None:
    #     self.e.get_fields4gis()

    # def test_save_prior(self) -> None:
    #     self.e.save_prior()

    # def test_get_cost_valley(self) -> None:
    #     self.e.get_3d_cost_valley()
    #     pass

    def test_get_trees_on_cost_valley(self) -> None:
        self.e.get_trees_on_cost_valley()
        pass

    # def test_plot_tiff(self) -> None:
    #     self.e.plot_tiff()
    #     pass

    # def test_simulation_result_discussion(self) -> None:
    #     pass
