"""
Simulator replicate study for EIBV case
"""
from Simulators.Simulator_Myopic2D import Simulator
from Config import Config
from joblib import Parallel, delayed
from time import time
import numpy as np

config = Config()
num_replicates = config.get_num_replicates()
num_cores = config.get_num_cores()
num_steps = config.get_num_steps()
reps = 1 * np.ones(num_replicates).astype(int)

"""
Return values are tuple and hereby need careful check with smaller steps 
of replicates to extract the result correctly. 
"""


def run_replicates(i):
    print("R: ", i)
    s = Simulator()
    traj, rmse, ibv, vr = s.run_simulator(num_steps)
    return traj, rmse, ibv, vr


if __name__ == "__main__":
    traj_sim = np.empty([0, 3, num_steps + 1, 2])
    ibv = np.empty([0, 3, num_steps + 1])
    rmse = np.empty([0, 3, num_steps + 1])
    vr = np.empty([0, 3, num_steps + 1])

    res = Parallel(n_jobs=num_cores)(delayed(run_replicates)(rep) for rep in reps)

    for i in range(len(res)):
        traj_sim = np.append(traj_sim, res[i][0].reshape(1, 3, num_steps + 1, 2), axis=0)
        rmse = np.append(rmse, res[i][1].reshape(1, 3, num_steps + 1), axis=0)
        ibv = np.append(ibv, res[i][2].reshape(1, 3, num_steps + 1), axis=0)
        vr = np.append(vr, res[i][3].reshape(1, 3, num_steps + 1), axis=0)

    np.save("npy/CV/Myopic/TRAJ.npy", traj_sim)
    np.save("npy/CV/Myopic/RMSE.npy", rmse)
    np.save("npy/CV/Myopic/IBV.npy", ibv)
    np.save("npy/CV/Myopic/VR.npy", vr)
