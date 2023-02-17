"""
Simulator runs the simulation based on two different EIBV mechanisms.

EIBV1 - simplified version which is only an approximation of the analytical one
EIBV2 - the analytical one.

The objective is to compare if EIBV1 and EIBV2 is actually producing different result.

"""
from Agents.AgentMyopic import Agent
import numpy as np


class SimulatorEIBV:

    def __init__(self, sigma: float = .1, nugget: float = .01, seed: int = 0, debug: bool = False) -> None:
        self.ag_eibv_approximate = Agent(weight_eibv=2., weight_ivr=.0, sigma=sigma, nugget=nugget,
                                         random_seed=seed, approximate_eibv=True, debug=debug, name="Approximate")
        # self.ag_eibv_analytical = Agent(weight_eibv=2., weight_ivr=.0, sigma=sigma, nugget=nugget,
        #                                 random_seed=seed, approximate_eibv=False, debug=debug, name="Analytical")

    def run_all(self, num_steps: int = 10) -> None:
        self.ag_eibv_approximate.run(num_steps=num_steps)
        # self.ag_eibv_analytical.run(num_steps=num_steps)

    # def extract_data_for_agent(self, agent: 'Agent' = None) -> tuple:
        # traj = agent.trajectory
        # ibv = np.array(agent.ibv).reshape(-1, 1)
        # vr = np.array(agent.vr).reshape(-1, 1)
        # rmse = np.array(agent.rmse).reshape(-1, 1)
        # return traj, ibv, vr, rmse

