"""
Simulator generates the simulation result based on different weight set.
"""
from Agents.AgentRRTStar import Agent
import numpy as np


class Simulator:

    def __init__(self, seed: int = 0, debug: bool = False) -> None:
        self.ag_eibv = Agent(weight_eibv=1.99, weight_ivr=.01, random_seed=seed, debug=debug, name="EIBV")
        self.ag_ivr = Agent(weight_eibv=.01, weight_ivr=1.99, random_seed=seed, debug=debug, name="IVR")
        self.ag_eq = Agent(weight_eibv=1., weight_ivr=1., random_seed=seed, debug=debug, name="Equal")

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

