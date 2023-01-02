"""
This script analyses the simulation result
Author: Yaolin Ge
Contact: yaolin.ge@ntnu.no
Date: 2022-03-25
"""
import os

import pandas as pd

from GOOGLE.Simulation_2DSquare.Config.Config import *
from usr_func import *


filepath_simulation = FILEPATH + "SimulationResult/"


class Result:

    def __init__(self, steps=None, mean=None, err=None):
        self.steps = steps
        self.mean = mean
        self.err = err


class ResultAll:

    def __init__(self, ibv=None, rmse=None, ev=None, crps=None):
        self.ibv = ibv
        self.rmse = rmse
        self.ev = ev
        self.crps = crps


class ResultAnalyser:

    def __init__(self):
        self.result_2d = self.get_results_from_files(strategy='2d')
        self.result_rrtstar = self.get_results_from_files(strategy='rrtstar')
        self.result_lawnmower = self.get_results_from_files(strategy='lawnmower')
        pass

    def get_results_from_files(self, strategy=None):
        files = os.listdir(filepath_simulation)
        for file in files:
            # if file.startswith('ibv'):
            if strategy in file:
                if 'ibv' in file:
                    self.df_ibv = pd.read_csv(filepath_simulation + file, index_col=False).iloc[:, 1:].to_numpy()
                    self.mean_ibv = np.mean(self.df_ibv, axis=0)
                    self.std_ibv = np.std(self.df_ibv, axis=0)
                    self.err_ibv = self.std_ibv / np.sqrt(self.df_ibv.shape[0]) * 1.645
                    print(file + " is finished successfully!")
                if 'rmse' in file:
                    self.df_rmse = pd.read_csv(filepath_simulation + file, index_col=False).iloc[:, 1:].to_numpy()
                    self.mean_rmse = np.mean(self.df_rmse, axis=0)
                    self.std_rmse = np.std(self.df_rmse, axis=0)
                    self.err_rmse = self.std_rmse / np.sqrt(self.df_rmse.shape[0]) * 1.645
                    print(file + " is finished successfully!")
                if 'ev' in file:
                    self.df_ev = pd.read_csv(filepath_simulation + file, index_col=False).iloc[:, 1:].to_numpy()
                    self.mean_ev = np.mean(self.df_ev, axis=0)
                    self.std_ev = np.std(self.df_ev, axis=0)
                    self.err_ev = self.std_ev / np.sqrt(self.df_ev.shape[0]) * 1.645
                    print(file + " is finished successfully!")
                if 'crps' in file:
                    self.df_crps = pd.read_csv(filepath_simulation + file, index_col=False).iloc[:, 1:].to_numpy()
                    self.mean_crps = np.mean(self.df_crps, axis=0)
                    self.std_crps = np.std(self.df_crps, axis=0)
                    self.err_crps = self.std_crps / np.sqrt(self.df_crps.shape[0]) * 1.645
                    print(file + " is finished successfully!")
        result = ResultAll(Result(np.arange(self.df_ibv.shape[1]), self.mean_ibv, self.err_ibv),
                           Result(np.arange(self.df_rmse.shape[1]), self.mean_rmse, self.err_rmse),
                           Result(np.arange(self.df_ev.shape[1]), self.mean_ev, self.err_ev),
                           Result(np.arange(self.df_crps.shape[1]), self.mean_crps, self.err_crps))
        return result

    def plot_simulation_result(self):
        fig = plt.figure(figsize=(30, 8))
        gs = GridSpec(nrows=1, ncols=3)
        ax1 = fig.add_subplot(gs[0])
        ax1.errorbar(self.result_2d.ibv.steps, self.result_2d.ibv.mean, yerr=self.result_2d.ibv.err, fmt="-o", capsize=5,
                     label="Myopic 2D")
        ax1.errorbar(self.result_rrtstar.ibv.steps, self.result_rrtstar.ibv.mean, yerr=self.result_rrtstar.ibv.err,
                     fmt="-o", capsize=5, label="RRT*")
        ax1.errorbar(self.result_lawnmower.ibv.steps, self.result_lawnmower.ibv.mean, yerr=self.result_lawnmower.ibv.err,
                     fmt="-o", capsize=5, label="Lawnmower")
        plt.xlabel('Time steps')
        plt.ylabel('IBV')
        plt.legend()

        ax2 = fig.add_subplot(gs[1])
        ax2.errorbar(self.result_2d.rmse.steps, self.result_2d.rmse.mean, yerr=self.result_2d.rmse.err, fmt="-o", capsize=5,
                     label="Myopic 2D")
        ax2.errorbar(self.result_rrtstar.rmse.steps, self.result_rrtstar.rmse.mean, yerr=self.result_rrtstar.rmse.err,
                     fmt="-o", capsize=5, label="RRT*")
        ax2.errorbar(self.result_lawnmower.rmse.steps, self.result_lawnmower.rmse.mean, yerr=self.result_lawnmower.rmse.err,
                     fmt="-o", capsize=5, label="Lawnmower")
        plt.xlabel('Time steps')
        plt.ylabel('RMSE')
        plt.legend()

        ax3 = fig.add_subplot(gs[2])
        ax3.errorbar(self.result_2d.ev.steps, self.result_2d.ev.mean, yerr=self.result_2d.ev.err, fmt="-o", capsize=5,
                     label="Myopic 2D")
        ax3.errorbar(self.result_rrtstar.ev.steps, self.result_rrtstar.ev.mean, yerr=self.result_rrtstar.ev.err,
                     fmt="-o", capsize=5, label="RRT*")
        ax3.errorbar(self.result_lawnmower.ev.steps, self.result_lawnmower.ev.mean, yerr=self.result_lawnmower.ev.err,
                     fmt="-o", capsize=5, label="Lawnmower")
        plt.xlabel('Time steps')
        plt.ylabel('Variance reduction')
        plt.legend()

        # ax4 = fig.add_subplot(gs[1, 1])
        # ax4.errorbar(np.arange(crps_2d.shape[1]), np.mean(crps_2d, axis=0),
        #              yerr=np.std(crps_2d, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o",
        #              capsize=5, label="Myopic 2D")
        # ax4.errorbar(np.arange(crps_rrtstar.shape[1]), np.mean(crps_rrtstar, axis=0),
        #              yerr=np.std(crps_rrtstar, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o",
        #              capsize=5, label="RRTStar")
        # ax4.errorbar(np.arange(crps_lawnmower.shape[1]), np.mean(crps_lawnmower, axis=0),
        #              yerr=np.std(crps_lawnmower, axis=0) / np.sqrt(self.NUMBER_REPLICATES) * 1.645, fmt="-o", capsize=5,
        #              label="Lawn mower")
        # ax4.plot(np.mean(dist_2d, axis=0), label="Myopic 2D")
        # ax4.plot(np.mean(dist_3d, axis=0), label="Myopic 3D")
        # ax4.plot(np.mean(dist_lawnmower, axis=0), label="Lawn mower")
        # ax4.set_yscale("log")
        # ax4.set_ylim(top=1e4)
        # plt.xlabel('Steps')
        # plt.ylabel('CRPS')
        # plt.legend()

        print(FIGPATH + "Sim_2DSquare/Result_3.pdf")
        plt.savefig(FIGPATH + "Sim_2DSquare/Result_3.pdf")
        plt.show()
        plt.close("all")
        pass

if __name__ == "__main__":
    r = ResultAnalyser()
    r.plot_simulation_result()



