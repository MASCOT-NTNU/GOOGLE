"""
Unittest for directional filter
"""
from unittest import TestCase
from CostValley.Direction import Direction
from Field import Field
from Config import Config
import matplotlib.pyplot as plt
from Visualiser.Visualiser import plotf_vector
from matplotlib.cm import get_cmap
from usr_func.interpolate_2d import interpolate_2d
import numpy as np


class TestDirection(TestCase):

    def setUp(self) -> None:
        f = Field()
        self.grid = f.get_grid()
        self.c = Config()
        self.d = Direction(self.grid)
        self.polygon_border = self.c.get_polygon_border()
        self.polygon_obstacle = self.c.get_polygon_obstacle()
        self.xlim, self.ylim = f.get_border_limits()

    def test_get_direction_field(self):
        # s1: starting loc
        loc = [1000, -1000]
        xp, yp = self.d.get_previous_location()
        plt.figure(figsize=(15, 12))
        df = self.d.get_direction_field(loc[0], loc[1])
        plt.scatter(self.grid[:, 1], self.grid[:, 0], c=df, s=600, cmap="RdBu", vmin=0, vmax=1, alpha=.4)
        xn, yn = self.d.get_current_location()
        plt.plot([yp, yn], [xp, xn], 'k-', linewidth=2)
        plt.plot(yn, xn, 'r.', markersize=20, label="Current location")
        plt.plot(yp, xp, 'b.', markersize=20, label="Previous location")
        plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'k-.')
        plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'k-.')
        plt.colorbar()
        plt.legend()
        plt.show()

        loc = [2000, -1000]
        xp, yp = self.d.get_previous_location()
        plt.figure(figsize=(15, 12))
        df = self.d.get_direction_field(loc[0], loc[1])
        plt.scatter(self.grid[:, 1], self.grid[:, 0], c=df, s=600, cmap="RdBu", vmin=0, vmax=1, alpha=.4)
        xn, yn = self.d.get_current_location()
        plt.plot([yp, yn], [xp, xn], 'k-', linewidth=2)
        plt.plot(yn, xn, 'r.', markersize=20, label="Current location")
        plt.plot(yp, xp, 'b.', markersize=20, label="Previous location")
        plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'k-.')
        plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'k-.')
        plt.colorbar()
        plt.legend()
        plt.show()

        # s3: move east
        loc = [3000, -900]
        xp, yp = self.d.get_previous_location()
        plt.figure(figsize=(15, 12))
        df = self.d.get_direction_field(loc[0], loc[1])
        plt.scatter(self.grid[:, 1], self.grid[:, 0], c=df, s=600, cmap="RdBu", vmin=0, vmax=1, alpha=.4)
        xn, yn = self.d.get_current_location()
        plt.plot([yp, yn], [xp, xn], 'k-', linewidth=2)
        plt.plot(yn, xn, 'r.', markersize=20, label="Current location")
        plt.plot(yp, xp, 'b.', markersize=20, label="Previous location")
        plt.plot(self.polygon_border[:, 1], self.polygon_border[:, 0], 'k-.')
        plt.plot(self.polygon_obstacle[:, 1], self.polygon_obstacle[:, 0], 'k-.')
        plt.colorbar()
        plt.legend()
        plt.show()
