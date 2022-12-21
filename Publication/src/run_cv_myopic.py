"""
Simulator replicate study for EIBV case
"""
from Simulators.Simulator_Myopic2D import Simulator
from Config import Config
from Simulators.CTD import CTD
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
    ctd = CTD()

    s2 = Simulator(weight_eibv=1., weight_ivr=1., ctd=ctd)
    t2, l2 = s2.run_simulator(num_steps=num_steps)

    s3 = Simulator(weight_eibv=.1, weight_ivr=1.9, ctd=ctd)
    t3, l3 = s3.run_simulator(num_steps=num_steps)

    s1 = Simulator(weight_eibv=1.9, weight_ivr=.1, ctd=ctd)
    t1, l1 = s1.run_simulator(num_steps=num_steps)

    return t1, t2, t3, l1, l2, l3


if __name__ == "__main__":
    traj_sim = np.empty([0, 3, num_steps + 1, 2])
    ibv = np.empty([0, 3, num_steps + 1])
    rmse = np.empty([0, 3, num_steps + 1])
    vr = np.empty([0, 3, num_steps + 1])

    res = Parallel(n_jobs=num_cores)(delayed(run_replicates)(rep) for rep in reps)

    for i in range(len(res)):
        traj = np.stack((res[i][0], res[i][1], res[i][2]), axis=0)
        traj_sim = np.append(traj_sim, traj.reshape(1, 3, num_steps + 1, 2), axis=0)

        rmse_r = np.stack((res[i][3].rmse, res[i][4].rmse, res[i][5].rmse), axis=0)
        rmse = np.append(rmse, rmse_r.reshape(1, 3, num_steps + 1), axis=0)

        ibv_r = np.stack((res[i][3].ibv, res[i][4].ibv, res[i][5].ibv), axis=0)
        ibv = np.append(ibv, ibv_r.reshape(1, 3, num_steps + 1), axis=0)

        vr_r = np.stack((res[i][3].vr, res[i][4].vr, res[i][5].vr), axis=0)
        vr = np.append(vr, vr_r.reshape(1, 3, num_steps + 1), axis=0)

    np.save("npy/CV/Myopic/TRAJ.npy", traj_sim)
    np.save("npy/CV/Myopic/RMSE.npy", rmse)
    np.save("npy/CV/Myopic/IBV.npy", ibv)
    np.save("npy/CV/Myopic/VR.npy", vr)


