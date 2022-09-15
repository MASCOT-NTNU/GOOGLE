from unittest import TestCase

from CostValley.Obstacle import Obstacle
from Field import Field
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap


class TestObstacle(TestCase):
    def setUp(self) -> None:
        self.f = Field()
        self.grid = self.f.get_grid()
        self.obs = Obstacle(self.grid, self.f)

    def test_get_obstacle_field(self):
        obs = self.obs.get_obstacle_field()
        plt.scatter(self.grid[:, 0], self.grid[:, 1], c=obs, cmap=get_cmap("RdBu", 2), vmin=0, vmax=1, alpha=.4)
        plt.colorbar()
        plt.show()


