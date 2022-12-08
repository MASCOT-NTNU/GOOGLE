"""
Simulator replicate study for IVR case
"""
from Simulators.Simulator import Simulator


if __name__ == "__main__":
    weight_eibv = .1
    weight_ivr = 1.9

    s = Simulator(weight_eibv=weight_eibv,
                  weight_ivr=weight_ivr,
                  case="IVR")
    s.run_replicates()



