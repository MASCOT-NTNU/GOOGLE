
import time
from usr_func import *
from Nidelva.Simulation.Simulator.Simulator import Simulator
from Nidelva.Simulation.Simulator.SimulationResultContainer import SimulationResultContainer
from matplotlib.gridspec import GridSpec


class SimulationReplicator:

    def __init__(self):
        self.result_simulation_2d = SimulationResultContainer("YoYo-assisted 2D Myopic Strategy")
        self.result_simulation_3d = SimulationResultContainer("3D Myopic Strategy")
        self.result_simulation_lawnmower = SimulationResultContainer("Lawn Mower Strategy")
        self.NUMBER_STEPS = 20
        self.NUMBER_REPLICATES = 100
        self.seed = np.random.choice(np.arange(1000), self.NUMBER_REPLICATES, replace=False)
        self.run_replicate()
        self.plot_simulation_result()
        pass

    def run_replicate(self):

        for i in range(self.NUMBER_REPLICATES):
            print("Replicate: ", i)
            t1 = time.time()
            seed = self.seed[i]
            try:
                self.simulation_2d = Simulator(steps=self.NUMBER_STEPS, random_seed=seed)
                self.simulation_2d.run_2d()
                self.result_simulation_2d.append(self.simulation_2d.knowledge)

                self.simulation_3d = Simulator(steps=self.NUMBER_STEPS, random_seed=seed)
                self.simulation_3d.run_3d()
                self.result_simulation_3d.append(self.simulation_3d.knowledge)

                self.simulation_lawnmower = Simulator(steps=self.NUMBER_STEPS, random_seed=seed)
                self.simulation_lawnmower.run_lawn_mower()
                self.result_simulation_lawnmower.append(self.simulation_lawnmower.knowledge)
            except:
                print("Jump over one")
                pass

            t2 = time.time()
            print('Each replicate takes: ', t2 - t1)

    def plot_simulation_result(self):
        ibv_2d = np.array(self.result_simulation_2d.expectedIntegratedBernoulliVariance)
        rmse_2d = np.array(self.result_simulation_2d.root_mean_squared_error)
        ev_2d = np.array(self.result_simulation_2d.expected_variance)
        dist_2d = np.array(self.result_simulation_2d.distanceTravelled)[:, 2:]

        ibv_3d = np.array(self.result_simulation_3d.expectedIntegratedBernoulliVariance)
        rmse_3d = np.array(self.result_simulation_3d.root_mean_squared_error)
        ev_3d = np.array(self.result_simulation_3d.expected_variance)
        dist_3d = np.array(self.result_simulation_3d.distanceTravelled)[:, 2:]

        ibv_lawnmower = np.array(self.result_simulation_lawnmower.expectedIntegratedBernoulliVariance)
        rmse_lawnmower = np.array(self.result_simulation_lawnmower.root_mean_squared_error)
        ev_lawnmower = np.array(self.result_simulation_lawnmower.expected_variance)
        dist_lawnmower = np.array(self.result_simulation_lawnmower.distanceTravelled)[:, 2:]

        fig = plt.figure(figsize=(20, 20))
        gs = GridSpec(nrows=2, ncols=2)
        ax1 = fig.add_subplot(gs[0, 0])
        # # ax1.plot(np.mean(self.result_simulation_2d.expectedIntegratedBernoulliVariance, axis=0))
        # # ax1.plot(np.mean(self.result_simulation_3d.expectedIntegratedBernoulliVariance, axis=0))
        # # ax1.plot(np.mean(self.result_simulation_lawnmower.expectedIntegratedBernoulliVariance, axis=0))

        ax1.errorbar(np.arange(ibv_2d.shape[1]), np.mean(ibv_2d, axis=0),
                     yerr=np.std(ibv_2d, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o",
                     capsize=5, label="Myopic 2D")
        ax1.errorbar(np.arange(ibv_3d.shape[1]), np.mean(ibv_3d, axis=0),
                     yerr=np.std(ibv_3d, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o",
                     capsize=5, label="Myopic 3D")
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
        ax2.errorbar(np.arange(rmse_3d.shape[1]), np.mean(rmse_3d, axis=0),
                     yerr=np.std(rmse_3d, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o",
                     capsize=5, label="Myopic 3D")
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
        ax3.errorbar(np.arange(ev_3d.shape[1]), np.mean(ev_3d, axis=0),
                     yerr=np.std(ev_3d, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o", capsize=5,
                     label="Myopic 3D")
        ax3.errorbar(np.arange(ev_lawnmower.shape[1]), np.mean(ev_lawnmower, axis=0),
                     yerr=np.std(ev_lawnmower, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o", capsize=5,
                     label="Lawn mower")
        plt.xlabel('Steps')
        plt.ylabel('Variance reduction')
        plt.legend()

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.errorbar(np.arange(dist_2d.shape[1]), np.mean(dist_2d, axis=0),
                     yerr=np.std(dist_2d, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o",
                     capsize=5, label="Myopic 2D")
        ax4.errorbar(np.arange(dist_3d.shape[1]), np.mean(dist_3d, axis=0),
                     yerr=np.std(dist_3d, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o",
                     capsize=5, label="Myopic 3D")
        ax4.errorbar(np.arange(dist_lawnmower.shape[1]), np.mean(dist_lawnmower, axis=0),
                     yerr=np.std(dist_lawnmower, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o", capsize=5,
                     label="Lawn mower")
        # ax4.plot(np.mean(dist_2d, axis=0), label="Myopic 2D")
        # ax4.plot(np.mean(dist_3d, axis=0), label="Myopic 3D")
        # ax4.plot(np.mean(dist_lawnmower, axis=0), label="Lawn mower")
        # ax4.set_yscale("log")
        # ax4.set_ylim(top=1e4)
        plt.xlabel('Steps')
        plt.ylabel('Distance travelled [m]')
        plt.legend()
        plt.savefig(
            "/Users/yaoling/OneDrive - NTNU/MASCOT_PhD/Publication/Nidelva/fig/Simulation/Result_5layers_MS.pdf")
        # plt.show()
        plt.close("all")
        pass

replicator = SimulationReplicator()

