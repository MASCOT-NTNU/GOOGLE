"""
Simulator replicate study for EIBV case
"""
from Simulators.Simulator import Simulator
from Config import Config
from joblib import Parallel, delayed
from time import time
import numpy as np

config = Config()
num_steps = config.get_num_steps()
num_replicates = config.get_num_replicates()
num_cores = config.get_num_cores()

"""
Return values are tuple and hereby need careful check with smaller steps 
of replicates to extract the result correctly. 
"""


def run_replicates(weight_eibv=1., weight_ivr=1.):
    s = Simulator()
    steps = num_steps * np.ones(num_replicates).astype(int)

    if weight_eibv > weight_ivr:
        case = "EIBV"
    elif weight_eibv < weight_ivr:
        case = "IVR"
    else:
        case = "EQUAL"

    s.set_weights(weight_eibv, weight_ivr)

    res = Parallel(n_jobs=num_cores)(delayed(s.run_simulator)(step) for step in steps)  # The return values are tuple

    traj_sim = np.empty([0, num_steps+1, 2])
    ibv = np.empty([0, num_steps+1])
    rmse = np.empty([0, num_steps+1])
    vr = np.empty([0, num_steps+1])

    for i in range(len(res)):
        traj_sim = np.append(traj_sim, res[i][0].reshape(1, num_steps+1, 2), axis=0)
        rmse = np.append(rmse, np.array(res[i][1].rmse).reshape(1, -1), axis=0)
        ibv = np.append(ibv, np.array(res[i][1].ibv).reshape(1, -1), axis=0)
        vr = np.append(vr, np.array(res[i][1].vr).reshape(1, -1), axis=0)

    # t1 = time()
    # for i in range(num_replicates):
    #     t2 = time()
    #     print("Replicate: ", i, " takes ", t2 - t1, " seconds.")
    #     t1 = time()
    #     traj = run_simulator()
    #     traj_sim = np.append(traj_sim, traj.reshape(1, num_steps+1, 2), axis=0)

    np.save("npy/CV/" + case + ".npy", traj_sim)
    np.save("npy/CV/" + case + "_ibv.npy", ibv)
    np.save("npy/CV/" + case + "_vr.npy", vr)
    np.save("npy/CV/" + case + "_rmse.npy", rmse)
    return 0


if __name__ == "__main__":
    run_replicates(1.9, .1)
    run_replicates(1., 1.)
    run_replicates(.1, 1.9)

