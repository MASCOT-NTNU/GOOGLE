from Visualiser.AgentPlot import AgentPlot
from Agents.Agent import Agent
from unittest import TestCase
import numpy as np
import os
import time


class TestAgentPlot(TestCase):

    def setUp(self) -> None:
        """
        Set up the planning strategies and the AUV simulator for the operation.
        """
        # s3: setup Visualiser.
        self.agent = Agent()

        self.ap = AgentPlot(agent=self.agent, figpath=os.getcwd() + "/../../fig/OP2_LongHorizon/")
        # self.visualiser = Visualiser(self, figpath=os.getcwd() + "/../fig/Myopic3D/")

    def test_plot(self):
        self.ap.plot_agent()

