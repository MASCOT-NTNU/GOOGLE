

from usr_func import *
import matplotlib.pyplot as plt


class SimulationResultsPlot:

    def __init__(self, knowledges, filename):
        self.knowledges = knowledges
        self.filename = filename
        self.plot()

    def plot(self):
        fig = plt.figure(figsize=(20, 20))

        plt.subplot(221)
        for i in range(len(self.knowledges)):
            plt.plot(self.knowledges[i].intergrated_bernoulli_variance, label =str(i + 2) + "D")
        plt.title("ibv")
        plt.legend()
        plt.xlabel("iteration")

        plt.subplot(222)
        for i in range(len(self.knowledges)):
            plt.plot(self.knowledges[i].root_mean_squared_error, label =str(i + 2) + "D")
        plt.title("rmse")
        plt.legend()
        plt.xlabel("iteration")

        plt.subplot(223)
        for i in range(len(self.knowledges)):
            plt.plot(self.knowledges[i].expected_variance, label =str(i + 2) + "D")
        plt.title("ev")
        plt.legend()
        plt.xlabel("iteration")

        plt.subplot(224)
        for i in range(len(self.knowledges)):
            plt.plot(self.knowledges[i].distance_travelled, label=str(i + 2) + "D")
        plt.gca().set_yscale("log")
        plt.title("Distance travelled")
        plt.legend()
        plt.xlabel("iteration")
        plt.suptitle(self.filename)

        plt.show()
