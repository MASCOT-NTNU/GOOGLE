"""
Log object logs the data generated during the RRT* simulation study.
"""

from Config import Config
import numpy as np
from scipy.stats import norm


class Log:
    """ Log """
    def __init__(self) -> None:
        self.distance = []

        pass

    def append_log(self, rrt) -> None:
        pass


if __name__ == "__main__":
    l = Log()
