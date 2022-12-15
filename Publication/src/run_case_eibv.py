"""
Simulator replicate study for EIBV case
"""
from Simulators.R_Simulator import Simulator
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
print(case)

config = Config()
num_replicates = config.get_num_replicates()


def run_replicates():

    s = Simulator(weight_eibv=weight_eibv,
                  weight_ivr=weight_ivr,
                  case=case)
    steps = 1 * np.ones(num_replicates).astype(int)
    res = Parallel(n_jobs=15)(delayed(s.run)(step) for step in steps)  # The return values are tuple
    """
    Return values are tuple and hereby need careful check with smaller steps 
    of replicates to extract the result correctly. 
    """
    Nstepsizes = len(s.stepsizes)
    Nmaxi = len(s.max_iterations)

    distance_traj = np.empty([0, Nstepsizes, Nmaxi])
    cost_traj = np.empty([0, Nstepsizes, Nmaxi])
    time_traj = np.empty([0, Nstepsizes, Nmaxi])

    for i in range(len(res)):
        distance_traj = np.append(distance_traj, res[i][0].reshape(1, Nstepsizes, Nmaxi), axis=0)
        cost_traj = np.append(cost_traj, res[i][1].reshape(1, Nstepsizes, Nmaxi), axis=0)
        time_traj = np.append(time_traj, res[i][2].reshape(1, Nstepsizes, Nmaxi), axis=0)

    np.save("npy/RRT/" + case + "_distance.npy", distance_traj)
    np.save("npy/RRT/" + case + "_cost.npy", cost_traj)
    np.save("npy/RRT/" + case + "_time.npy", time_traj)
    return 0


if __name__ == "__main__":
    run_replicates()

