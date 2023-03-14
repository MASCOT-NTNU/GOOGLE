"""
Simulator replicate study for EIBV case
"""
"""
!!! How to run multiple input parallel processing. 
"""
# Parallel(n_jobs=6)(delayed(makeGraph)(graph_type=graph, nodes=vertex, edge_probability=prob, power_exponent=exponent) for vertex in vertices for prob in edge_probabilities for exponent in power_exponents for graph in graph_types)

from Simulators.Simulator_Myopic2D import SimulatorMyopic2D
from Simulators.Simulator_RRTStar import SimulatorRRTStar
from Visualiser.AgentPlotMyopic import AgentPlotMyopic
from Visualiser.AgentPlotRRTStar import AgentPlotRRTStar
from usr_func.checkfolder import checkfolder
from Config import Config
from joblib import Parallel, delayed
from time import time
import numpy as np
from tqdm import tqdm

config = Config()
num_replicates = config.get_num_replicates()
num_cores = config.get_num_cores()
num_steps = config.get_num_steps()
# reps = 1 * np.ones(num_replicates).astype(int)
# seeds = np.random.randint(0, 10000, num_replicates)
seeds = np.random.choice(10000, num_replicates, replace=False)  # to generate non-repetitive seeds.

neighbour_distance = 240

"""
Return values are tuple and hereby need careful check with smaller steps 
of replicates to extract the result correctly. 
"""
Simulators = [SimulatorRRTStar, SimulatorMyopic2D]
# sigmas = [1.5, 1., .5, .1]
# nuggets = [.4, .25, .1, .01]

# datapath = "npy/"

sigmas = [1.]
nuggets = [.4]

# sigmas = [.5]
# nuggets = [.25]


datapath = "npy/analytical/"

def run_replicates(i: int = 0):
    print("seed: ", seeds[i])
    folderpath = datapath + "R_{:03d}/".format(i)
    checkfolder(folderpath)
    for sigma in sigmas:
        print("sigma: ", sigma)
        sigpath = folderpath + "sigma_{:02d}/".format(int(10 * sigma))
        checkfolder(sigpath)
        for nugget in nuggets:
            print("nugget: ", nugget)
            nuggetpath = sigpath + "nugget_{:03d}/".format(int(100 * nugget))
            checkfolder(nuggetpath)
            for Simulator in Simulators:
                print("simulator: ", Simulator.__name__)
                simpath = nuggetpath + Simulator.__name__ + "/"
                checkfolder(simpath)

                s = Simulator(neighbour_distance=neighbour_distance, sigma=sigma, nugget=nugget,
                              seed=seeds[i], debug=False, approximate_eibv=False, fast_eibv=True)
                """ Save simulation figures. """
                # if "Myopic" in Simulator.__name__:
                #     ap = AgentPlotMyopic
                # else:
                #     ap = AgentPlotRRTStar
                # app = ap(s.ag_eq, simpath)
                # app.plot_ground_truth(title="truth", seed=seeds[i])
                """ End of plotting. """

                s.run_all(num_steps=num_steps)
                res_eibv = s.extract_data_for_agent(s.ag_eibv)
                res_ivr = s.extract_data_for_agent(s.ag_ivr)
                res_eq = s.extract_data_for_agent(s.ag_eq)

                """ Get each component. """
                traj = np.stack((res_eibv[0], res_ivr[0], res_eq[0]), axis=0)
                ibv = np.stack((res_eibv[1], res_ivr[1], res_eq[1]), axis=0)
                vr = np.stack((res_eibv[2], res_ivr[2], res_eq[2]), axis=0)
                rmse = np.stack((res_eibv[3], res_ivr[3], res_eq[3]), axis=0)

                np.save(simpath + "traj.npy", traj)
                np.save(simpath + "ibv.npy", ibv)
                np.save(simpath + "vr.npy", vr)
                np.save(simpath + "rmse.npy", rmse)


if __name__ == "__main__":
    t1 = time()
    res = Parallel(n_jobs=num_cores)(delayed(run_replicates)(i=rep) for rep in range(num_replicates))
    t2 = time()
    print("Replicate study takes ", t2 - t1)

