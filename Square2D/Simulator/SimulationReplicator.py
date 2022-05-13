
import time

import pandas as pd

from usr_func import *
from GOOGLE.Simulation_2DSquare.Simulator.Simulator import Simulator
from GOOGLE.Simulation_2DSquare.Simulator.SimulationResultContainer import SimulationResultContainer
from matplotlib.gridspec import GridSpec
from GOOGLE.Simulation_2DSquare.Config.Config import *



class SimulationReplicator:

    def __init__(self):
        self.result_simulation_2d = SimulationResultContainer("Myopic 2D Strategy")
        self.result_simulation_rrtstar = SimulationResultContainer("RRTStar Strategy")
        self.result_simulation_lawnmower = SimulationResultContainer("Lawn Mower Strategy")
        self.NUMBER_STEPS = 20
        self.NUMBER_REPLICATES = 50
        self.seed = np.random.choice(np.arange(1000), self.NUMBER_REPLICATES, replace=False)
        self.run_replicate()
        self.save_simulation_result()
        self.plot_simulation_result()
        pass

    def run_replicate(self):
        for i in range(self.NUMBER_REPLICATES):
            print("Replicate: ", i)
            t1 = time.time()
            seed = self.seed[i]
            # try:
            blockPrint()
            self.simulation_2d = Simulator(steps=self.NUMBER_STEPS, random_seed=seed, replicates=True)
            self.simulation_2d.run_2d()
            self.result_simulation_2d.append(self.simulation_2d.knowledge)

            self.simulation_rrtstar = Simulator(steps=self.NUMBER_STEPS, random_seed=seed, replicates=True)
            self.simulation_rrtstar.run_rrtstar()
            self.result_simulation_rrtstar.append(self.simulation_rrtstar.knowledge)

            self.simulation_lawnmower = Simulator(steps=self.NUMBER_STEPS, random_seed=seed, replicates=True)
            self.simulation_lawnmower.run_lawn_mower()
            self.result_simulation_lawnmower.append(self.simulation_lawnmower.knowledge)
            # except:
            #     print("Jump over one")
                # pass
            enablePrint()
            t2 = time.time()
            print('Each replicate takes: ', t2 - t1)

    def plot_simulation_result(self):
        ibv_2d = np.array(self.result_simulation_2d.expected_integrated_bernoulli_variance)
        rmse_2d = np.array(self.result_simulation_2d.root_mean_squared_error)
        ev_2d = np.array(self.result_simulation_2d.expected_variance)
        crps_2d = np.array(self.result_simulation_2d.continuous_ranked_probability_score)

        ibv_rrtstar = np.array(self.result_simulation_rrtstar.expected_integrated_bernoulli_variance)
        rmse_rrtstar = np.array(self.result_simulation_rrtstar.root_mean_squared_error)
        ev_rrtstar = np.array(self.result_simulation_rrtstar.expected_variance)
        crps_rrtstar = np.array(self.result_simulation_rrtstar.continuous_ranked_probability_score)

        ibv_lawnmower = np.array(self.result_simulation_lawnmower.expected_integrated_bernoulli_variance)
        rmse_lawnmower = np.array(self.result_simulation_lawnmower.root_mean_squared_error)
        ev_lawnmower = np.array(self.result_simulation_lawnmower.expected_variance)
        crps_lawnmower = np.array(self.result_simulation_lawnmower.continuous_ranked_probability_score)

        fig = plt.figure(figsize=(20, 20))
        gs = GridSpec(nrows=2, ncols=2)
        ax1 = fig.add_subplot(gs[0, 0])
        # # ax1.plot(np.mean(self.result_simulation_2d.expectedIntegratedBernoulliVariance, axis=0))
        # # ax1.plot(np.mean(self.result_simulation_3d.expectedIntegratedBernoulliVariance, axis=0))
        # # ax1.plot(np.mean(self.result_simulation_lawnmower.expectedIntegratedBernoulliVariance, axis=0))

        ax1.errorbar(np.arange(ibv_2d.shape[1]), np.mean(ibv_2d, axis=0),
                     yerr=np.std(ibv_2d, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o",
                     capsize=5, label="Myopic 2D")
        ax1.errorbar(np.arange(ibv_rrtstar.shape[1]), np.mean(ibv_rrtstar, axis=0),
                     yerr=np.std(ibv_rrtstar, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o",
                     capsize=5, label="RRTStar")
        ax1.errorbar(np.arange(ibv_lawnmower.shape[1]), np.mean(ibv_lawnmower, axis=0),
                     yerr=np.std(ibv_lawnmower, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o", capsize=5,
                     label="Lawn mower")
        plt.xlabel('Steps')
        plt.ylabel('IBV')
        plt.legend()

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.errorbar(np.arange(rmse_2d.shape[1]), np.mean(rmse_2d, axis=0),
                     yerr=np.std(rmse_2d, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o",
                     capsize=5, label="Myopic 2D")
        ax2.errorbar(np.arange(rmse_rrtstar.shape[1]), np.mean(rmse_rrtstar, axis=0),
                     yerr=np.std(rmse_rrtstar, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o",
                     capsize=5, label="RRTStar")
        ax2.errorbar(np.arange(rmse_lawnmower.shape[1]), np.mean(rmse_lawnmower, axis=0),
                     yerr=np.std(rmse_lawnmower, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o", capsize=5,
                     label="Lawn mower")
        plt.xlabel('Steps')
        plt.ylabel('RMSE')
        plt.legend()

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.errorbar(np.arange(ev_2d.shape[1]), np.mean(ev_2d, axis=0),
                     yerr=np.std(ev_2d, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o", capsize=5,
                     label="Myopic 2D")
        ax3.errorbar(np.arange(ev_rrtstar.shape[1]), np.mean(ev_rrtstar, axis=0),
                     yerr=np.std(ev_rrtstar, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o", capsize=5,
                     label="RRTStar")
        ax3.errorbar(np.arange(ev_lawnmower.shape[1]), np.mean(ev_lawnmower, axis=0),
                     yerr=np.std(ev_lawnmower, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o", capsize=5,
                     label="Lawn mower")
        plt.xlabel('Steps')
        plt.ylabel('Variance reduction')
        plt.legend()

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.errorbar(np.arange(crps_2d.shape[1]), np.mean(crps_2d, axis=0),
                     yerr=np.std(crps_2d, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o",
                     capsize=5, label="Myopic 2D")
        ax4.errorbar(np.arange(crps_rrtstar.shape[1]), np.mean(crps_rrtstar, axis=0),
                     yerr=np.std(crps_rrtstar, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o",
                     capsize=5, label="RRTStar")
        ax4.errorbar(np.arange(crps_lawnmower.shape[1]), np.mean(crps_lawnmower, axis=0),
                     yerr=np.std(crps_lawnmower, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o", capsize=5,
                     label="Lawn mower")
        # ax4.plot(np.mean(dist_2d, axis=0), label="Myopic 2D")
        # ax4.plot(np.mean(dist_3d, axis=0), label="Myopic 3D")
        # ax4.plot(np.mean(dist_lawnmower, axis=0), label="Lawn mower")
        # ax4.set_yscale("log")
        # ax4.set_ylim(top=1e4)
        plt.xlabel('Steps')
        plt.ylabel('CRPS')
        plt.legend()
        print(FIGPATH + "Sim_2DSquare/Result.pdf")
        plt.savefig(FIGPATH + "Sim_2DSquare/Result.pdf")
        # plt.show()
        plt.close("all")
        pass

    def save_simulation_result(self):
        ibv_2d = np.array(self.result_simulation_2d.expected_integrated_bernoulli_variance)
        rmse_2d = np.array(self.result_simulation_2d.root_mean_squared_error)
        ev_2d = np.array(self.result_simulation_2d.expected_variance)
        crps_2d = np.array(self.result_simulation_2d.continuous_ranked_probability_score)

        ibv_rrtstar = np.array(self.result_simulation_rrtstar.expected_integrated_bernoulli_variance)
        rmse_rrtstar = np.array(self.result_simulation_rrtstar.root_mean_squared_error)
        ev_rrtstar = np.array(self.result_simulation_rrtstar.expected_variance)
        crps_rrtstar = np.array(self.result_simulation_rrtstar.continuous_ranked_probability_score)

        ibv_lawnmower = np.array(self.result_simulation_lawnmower.expected_integrated_bernoulli_variance)
        rmse_lawnmower = np.array(self.result_simulation_lawnmower.root_mean_squared_error)
        ev_lawnmower = np.array(self.result_simulation_lawnmower.expected_variance)
        crps_lawnmower = np.array(self.result_simulation_lawnmower.continuous_ranked_probability_score)

        print(ibv_2d.shape)
        print(rmse_2d.shape)
        print(ev_2d.shape)
        print(crps_2d.shape)

        print(ibv_rrtstar.shape)
        print(rmse_rrtstar.shape)
        print(ev_rrtstar.shape)
        print(crps_rrtstar.shape)

        print(ibv_lawnmower.shape)
        print(rmse_lawnmower.shape)
        print(ev_lawnmower.shape)
        print(crps_lawnmower.shape)

        index = ["Replicate_{:d}".format(i) for i in range(ibv_2d.shape[0])]
        columns = ["Iteration_{:d}".format(i) for i in range(ibv_2d.shape[1])]
        save_file2csv(ibv_2d, FILEPATH + "SimulationResult/ibv_2d.csv", index, columns)
        save_file2csv(rmse_2d, FILEPATH + "SimulationResult/rmse_2d.csv", index, columns)
        save_file2csv(ev_2d, FILEPATH + "SimulationResult/ev_2d.csv", index, columns)
        save_file2csv(crps_2d, FILEPATH + "SimulationResult/crps_2d.csv", index, columns)

        save_file2csv(ibv_rrtstar, FILEPATH + "SimulationResult/ibv_rrtstar.csv", index, columns)
        save_file2csv(rmse_rrtstar, FILEPATH + "SimulationResult/rmse_rrtstar.csv", index, columns)
        save_file2csv(ev_rrtstar, FILEPATH + "SimulationResult/ev_rrtstar.csv", index, columns)
        save_file2csv(crps_rrtstar, FILEPATH + "SimulationResult/crps_rrtstar.csv", index, columns)

        save_file2csv(ibv_lawnmower, FILEPATH + "SimulationResult/ibv_lawnmower.csv", index, columns)
        save_file2csv(rmse_lawnmower, FILEPATH + "SimulationResult/rmse_lawnmower.csv", index, columns)
        save_file2csv(ev_lawnmower, FILEPATH + "SimulationResult/ev_lawnmower.csv", index, columns)
        save_file2csv(crps_lawnmower, FILEPATH + "SimulationResult/crps_lawnmower.csv", index, columns)
        pass


replicator = SimulationReplicator()




