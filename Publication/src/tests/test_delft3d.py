from unittest import TestCase
from Delft3D import Delft3D
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


class TestDelft3D(TestCase):
    def setUp(self) -> None:
        self.d = Delft3D()

    def test_wind_conditions(self) -> None:
        # c1: north moderate
        dd = self.d.get_dataset()
        plt.scatter(dd[:, 1], dd[:, 0], c=dd[:, 2], cmap=get_cmap("BrBG", 10), vmin=10, vmax=36)
        plt.colorbar()
        plt.show()

