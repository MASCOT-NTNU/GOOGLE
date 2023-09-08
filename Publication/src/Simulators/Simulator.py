"""
Simulator launches the simulation for all the agents.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-09-06
"""

from Agents.AgentMyopic import Agent as AgentMyopic
from Agents.AgentRRTStar import Agent as AgentRRTStar
from Config import Config
from usr_func.checkfolder import checkfolder
import numpy as np
import os
from time import time


class Simulator:

    def __init__(self, weight_eibv: float = 1., weight_ivr: float = 1.,
                 random_seed: int = 0, replicate_id: int = 0, debug: bool = False) -> None:
        self.__random_seed = random_seed
        self.__debug = debug
        self.__config = Config()
        self.__num_steps = self.__config.get_num_steps()
        if weight_eibv > weight_ivr:
            self.__name = "EIBV"
        elif weight_eibv < weight_ivr:
            self.__name = "IVR"
        else:
            self.__name = "Equal"
        self.__agent_myopic = AgentMyopic(weight_eibv=weight_eibv, weight_ivr=weight_ivr,
                                          random_seed=self.__random_seed, debug=self.__debug, name=self.__name)
        self.__agent_rrtstar = AgentRRTStar(weight_eibv=weight_eibv, weight_ivr=weight_ivr,
                                            random_seed=self.__random_seed, debug=self.__debug, name=self.__name)

        self.__datapath = os.getcwd() + "/npy/temporal/R_{:03d}/".format(replicate_id) + self.__name + "/"
        checkfolder(self.__datapath)

    def run_myopic(self) -> None:
        """ Run the simulation for all the agents. """
        t0 = time()
        self.__agent_myopic.run()
        print("Myopic simulation takes {:.2f} seconds.".format(time() - t0))

        t0 = time()
        (traj_myopic, ibv_myopic, rmse_myopic, vr_myopic,
         mu_data_myopic, cov_myopic, sigma_data_myopic, mu_truth_data_myopic) = self.__agent_myopic.get_metrics()

        np.savez(self.__datapath + "myopic.npz", traj=traj_myopic, ibv=ibv_myopic, rmse=rmse_myopic, vr=vr_myopic,
                    mu=mu_data_myopic, cov=cov_myopic, sigma=sigma_data_myopic, truth=mu_truth_data_myopic)
        print("Saving data takes {:.2f} seconds.".format(time() - t0))

    def run_rrt(self) -> None:
        """ Run the simulation for all the agents. """
        t0 = time()
        self.__agent_rrtstar.run()
        print("RRT* simulation takes {:.2f} seconds.".format(time() - t0))

        (traj_rrtstar, ibv_rrtstar, rmse_rrtstar, vr_rrtstar,
         mu_data_rrtstar, cov_rrtstar, sigma_data_rrtstar, mu_truth_data_rrtstar) = self.__agent_rrtstar.get_metrics()

        np.savez(self.__datapath + "rrtstar.npz", traj=traj_rrtstar, ibv=ibv_rrtstar, rmse=rmse_rrtstar, vr=vr_rrtstar,
                    mu=mu_data_rrtstar, cov=cov_rrtstar, sigma=sigma_data_rrtstar, truth=mu_truth_data_rrtstar)
        print("Saving data takes {:.2f} seconds.".format(time() - t0))
        print("Mission completed.")


if __name__ == "__main__":
    s = Simulator()

