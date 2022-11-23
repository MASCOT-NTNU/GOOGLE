"""
This function will set the global resume state
"""

import numpy as np


def set_resume_state(state: bool = False) -> None:
    if state:
        a = np.array([10.0])
        np.savetxt("resume_flag.txt", a)
    else:
        a = np.array([.0])
        np.savetxt("resume_flag.txt", a)
        np.savetxt("counter.txt", np.array([.0]))  # reset the counter as well


if __name__ == "__main__":
    set_resume_state(False)

