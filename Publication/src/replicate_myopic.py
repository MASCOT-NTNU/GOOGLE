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
# reps = 1 * np.ones(num_replicates).astype(int)
seeds = np.random.randint(0, 10000, num_replicates)

"""
Return values are tuple and hereby need careful check with smaller steps 
of replicates to extract the result correctly. 
"""

def run_replicates(i):
    print("Seed: ", seeds[i])
    s = Simulator(seed=seeds[i], debug=False)
    s.run_all(num_steps=num_steps)
    res_eibv = s.extract_data_for_agent(s.ag_eibv)
    res_ivr = s.extract_data_for_agent(s.ag_ivr)
    res_eq = s.extract_data_for_agent(s.ag_eq)
    return res_eibv, res_ivr, res_eq


if __name__ == "__main__":
    traj_sim = np.empty([0, 3, num_steps, 2])
    ibv = np.empty([0, 3, num_steps])
    rmse = np.empty([0, 3, num_steps])
    vr = np.empty([0, 3, num_steps])

    res = Parallel(n_jobs=num_cores)(delayed(run_replicates)(rep) for rep in range(num_replicates))

    for i in range(len(res)):
        res_eibv = res[i][0]
        res_ivr = res[i][1]
        res_eq = res[i][2]

        traj = np.stack((res_eibv[0], res_ivr[0], res_eq[0]), axis=0)
        traj_sim = np.append(traj_sim, traj.reshape(1, 3, num_steps, 2), axis=0)

        ibv_r = np.stack((res_eibv[1], res_ivr[1], res_eq[1]), axis=0)
        ibv = np.append(ibv, ibv_r.reshape(1, 3, num_steps), axis=0)

        vr_r = np.stack((res_eibv[2], res_ivr[2], res_eq[2]), axis=0)
        vr = np.append(vr, vr_r.reshape(1, 3, num_steps), axis=0)

        rmse_r = np.stack((res_eibv[3], res_ivr[3], res_eq[3]), axis=0)
        rmse = np.append(rmse, rmse_r.reshape(1, 3, num_steps), axis=0)

    np.save("npy/CV/Myopic/TRAJ.npy", traj_sim)
    np.save("npy/CV/Myopic/RMSE.npy", rmse)
    np.save("npy/CV/Myopic/IBV.npy", ibv)
    np.save("npy/CV/Myopic/VR.npy", vr)


