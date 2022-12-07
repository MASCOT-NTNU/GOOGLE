"""
Simulator replicate study for equal case
"""
from Simulators.Simulator import Simulator


if __name__ == "__main__":
    weight_eibv = 1.
    weight_ivr = 1.

    s = Simulator(weight_eibv=weight_eibv,
                  weight_ivr=weight_ivr,
                  case="Equal")
    s.run_replicates()



