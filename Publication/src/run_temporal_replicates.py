"""
This script runs the replicate study using two different agents in the same field.

Author: Yaolin Ge
Email: geyaolin@gmail.com
Date: 2023-09-06
"""
from Simulators.Simulator import Simulator
from joblib import Parallel, delayed
from Config import Config
import numpy as np


config = Config()
num_cores = config.get_num_cores()
num_steps = config.get_num_steps()
num_replicates = config.get_num_replicates()

print("Number of cores: ", num_cores, " | Number of steps: ",
      num_steps, " | Number of replicates: ", num_replicates)

debug = False
seeds = np.random.choice(10000, num_replicates, replace=False)

weight_set = np.array([[2., 0.],
                       [0., 2.],
                       [1., 1.]])
seed_weight_set = []
counter = 0
for seed in seeds:
    for weight in weight_set:
        seed_weight_set.append([counter, seed, weight[0], weight[1]])
    counter += 1
seed_weight_set = np.array(seed_weight_set)

print("Weight set: ")
print(seed_weight_set)


def run_replicates(sws: np.ndarray = np.array([0, 102, 1., 1.])):
    """
    This function runs the replicate study using two different agents in the same field.
    """
    i = int(sws[0])
    seed = int(sws[1])
    weight_eibv = sws[2]
    weight_ivr = sws[3]
    print("Replicate: ", i, " | Seed: ", seed, " | Weight EIBV: ", weight_eibv, " | Weight IVR: ", weight_ivr)
    simulator = Simulator(weight_eibv=weight_eibv, weight_ivr=weight_ivr,
                          random_seed=seed, replicate_id=i, debug=debug)
    simulator.run_myopic()
    simulator.run_rrt()


if __name__ == "__main__":
    Parallel(n_jobs=num_cores)(delayed(run_replicates)(sws) for sws in seed_weight_set)
