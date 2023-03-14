"""
Simulator generates the simulation result based on different weight set.
"""
from Agents.AgentMyopic import Agent
import numpy as np


class SimulatorMyopic2D:

    def __init__(self, neighbour_distance: float=120, sigma: float = .1, nugget: float = .01, seed: int = 0,
                 debug: bool = False, approximate_eibv: bool = False, fast_eibv: bool = True,
                 directional_penalty: bool = False) -> None:
        self.ag_eibv = Agent(neighbour_distance=neighbour_distance, weight_eibv=1.99, weight_ivr=.01,
                             sigma=sigma, nugget=nugget, random_seed=seed, debug=debug, name="EIBV",
                             approximate_eibv=approximate_eibv, fast_eibv=fast_eibv,
                             directional_penalty=directional_penalty)
        self.ag_ivr = Agent(neighbour_distance=neighbour_distance, weight_eibv=.01, weight_ivr=1.99,
                            sigma=sigma, nugget=nugget, random_seed=seed, debug=debug, name="IVR",
                            approximate_eibv=approximate_eibv, fast_eibv=fast_eibv,
                            directional_penalty=directional_penalty)
        self.ag_eq = Agent(neighbour_distance=neighbour_distance, weight_eibv=1., weight_ivr=1.,
                           sigma=sigma, nugget=nugget, random_seed=seed, debug=debug, name="Equal",
                           approximate_eibv=approximate_eibv, fast_eibv=fast_eibv,
                           directional_penalty=directional_penalty)

    def run_all(self, num_steps: int = 10) -> None:
        self.ag_eibv.run(num_steps=num_steps)
        self.ag_ivr.run(num_steps=num_steps)
        self.ag_eq.run(num_steps=num_steps)

    def extract_data_for_agent(self, agent: 'Agent' = None) -> tuple:
        traj = agent.trajectory
        ibv = np.array(agent.ibv).reshape(-1, 1)
        vr = np.array(agent.vr).reshape(-1, 1)
        rmse = np.array(agent.rmse).reshape(-1, 1)
        return traj, ibv, vr, rmse

