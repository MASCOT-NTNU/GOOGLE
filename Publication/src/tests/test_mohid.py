from unittest import TestCase
from MOHID import MOHID
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


class TestMOHID(TestCase):

    def setUp(self) -> None:
        self.m = MOHID()

    def test_pass(self):
        md = self.m.get_mohid_dataset()

        plt.scatter(md[:, 1], md[:, 0], c=md[:, 2], cmap=get_cmap("BrBG", 10), vmin=10, vmax=36)
        plt.colorbar()
        plt.show()

        dd




