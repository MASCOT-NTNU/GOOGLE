from unittest import TestCase
from CostValley.Direction import Direction
from Field import Field

import matplotlib.pyplot as plt
from Visualiser.Visualiser import plotf_vector
from matplotlib.cm import get_cmap
from usr_func.interpolate_2d import interpolate_2d
import numpy as np


class TestDirection(TestCase):

    def setUp(self) -> None:
        f = Field()
        self.grid = f.get_grid()
        self.d = Direction(self.grid)
        self.polygon_border = f.get_polygon_border()
        self.polygon_border = np.append(self.polygon_border, self.polygon_border[0, :].reshape(1, -1), axis=0)
        # self.polygon_obstacle = f.get_polygon_obstacles()[0]
        # self.polygon_obstacle = np.append(self.polygon_obstacle, self.polygon_obstacle[0, :].reshape(1, -1), axis=0)

        self.xlim, self.ylim = f.get_border_limits()

    def test_get_direction_field(self):
        # s1: starting loc
        loc = [6000, 8000]
        xp, yp = self.d.get_previous_location()
        plt.figure(figsize=(15, 12))
        df = self.d.get_direction_field(loc[0], loc[1])
        plt.scatter(self.grid[:, 1], self.grid[:, 0], c=df, s=600, cmap="RdBu", vmin=0, vmax=1, alpha=.4)
        xn, yn = self.d.get_current_location()
        plt.plot([yp, yn], [xp, xn], 'k-', linewidth=2)
        plt.plot(yn, xn, 'r.', markersize=20, label="Current location")
        plt.plot(yp, xp, 'b.', markersize=20, label="Previous location")
        plt.colorbar()
        plt.legend()
        plt.show()

        # s2: move
        loc = [6360, 8360]
        xp, yp = self.d.get_previous_location()
        plt.figure(figsize=(15, 12))
        df = self.d.get_direction_field(loc[0], loc[1])
        x = self.grid[:, 1]
        y = self.grid[:, 0]
        # gx, gy, gv = interpolate_2d(x, y, 50, 50, df)
        # plotf_vector(self.grid[:, 0], self.grid[:, 1], df, cmap="RdBu", xlabel='x',
        #              ylabel='y', title='Direction', cbar_title="Cost", vmin=-1, vmax=1.1, stepsize=.4)
        plt.scatter(self.grid[:, 1], self.grid[:, 0], c=df, s=600, cmap="RdBu", vmin=0, vmax=1, alpha=.4)
        # plt.scatter(gx, gy, c=gv, cmap="RdBu", vmin=0, vmax=1, alpha=.4, s=20)
        xn, yn = self.d.get_current_location()
        plt.plot([yp, yn], [xp, xn], 'k-', linewidth=2)
        plt.plot(yn, xn, 'r.', markersize=20, label="Current location")
        plt.plot(yp, xp, 'b.', markersize=20, label="Previous location")
        # plt.plot(self.polygon_border[:, 0], self.polygon_border[:, 1], 'r-.')
        # plt.plot(self.polygon_obstacle[:, 0], self.polygon_obstacle[:, 1], 'r-.')
        # plt.plot(0, 0, 'r.', markersize=20)
        # plt.plot(0, 1, 'k*', markersize=20)
        plt.xlim(self.xlim)
        plt.ylim(self.ylim)
        # plt.colorbar()
        plt.legend()
        plt.show()

        # s3: move east
        loc = [6360, 8900]
        xp, yp = self.d.get_previous_location()
        plt.figure(figsize=(15, 12))
        df = self.d.get_direction_field(loc[0], loc[1])
        plt.scatter(self.grid[:, 1], self.grid[:, 0], c=df, s=600, cmap="RdBu", vmin=0, vmax=1, alpha=.4)
        xn, yn = self.d.get_current_location()
        plt.plot([yp, yn], [xp, xn], 'k-', linewidth=2)
        plt.plot(yn, xn, 'r.', markersize=20, label="Current location")
        plt.plot(yp, xp, 'b.', markersize=20, label="Previous location")
        plt.colorbar()
        plt.legend()
        plt.show()
