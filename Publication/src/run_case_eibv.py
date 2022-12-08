"""
Simulator replicate study for EIBV case
"""
from Simulators.Simulator import Simulator
from Config import Config
from joblib import Parallel, delayed
from time import time
import numpy as np


weight_eibv = 1.9
weight_ivr = .1

if weight_eibv > weight_ivr:
    case = "EIBV"
elif weight_eibv < weight_ivr:
    case = "IVR"
else:
    case = "EQUAL"

config = Config()
num_steps = config.get_num_steps()
num_replicates = config.get_num_replicates()


def run_replicates():

    s = Simulator(weight_eibv=weight_eibv,
                  weight_ivr=weight_ivr,
                  case=case)
    steps = num_steps * np.ones(num_replicates).astype(int)
    res = Parallel(n_jobs=3)(delayed(s.run_simulator)(step) for step in steps)

    traj_sim = np.empty([0, num_steps+1, 2])

    for i in range(len(res)):
        traj_sim = np.append(traj_sim, res[0].reshape(1, num_steps+1, 2), axis=0)

    # t1 = time()
    # for i in range(num_replicates):
    #     t2 = time()
    #     print("Replicate: ", i, " takes ", t2 - t1, " seconds.")
    #     t1 = time()
    #     traj = run_simulator()
    #     traj_sim = np.append(traj_sim, traj.reshape(1, num_steps+1, 2), axis=0)

    np.save("npy/" + case + ".npy", traj_sim)
    return 0


if __name__ == "__main__":
    run_replicates()

