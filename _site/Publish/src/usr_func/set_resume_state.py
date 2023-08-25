"""
This function is used to set the resume state of the simulation.
"""

import numpy as np


def set_resume_state(state: bool = False) -> None:
    """
    Args:
        state (bool): True if the simulation is to be resumed, False otherwise.

    Returns:
        None

    Examples:
        >>> set_resume_state(True)
        $ cat resume_flag.txt
        10.0
        $ cat counter.txt  # it comes from the previous run, so the number may vary.
        34.0
        >>> set_resume_state(False)
        $ cat resume_flag.txt
        0.0
        $ cat counter.txt
        0.0

    """
    if state:
        a = np.array([10.0])
        np.savetxt("resume_flag.txt", a)
    else:
        a = np.array([.0])
        np.savetxt("resume_flag.txt", a)
        np.savetxt("counter.txt", np.array([.0]))  # reset the counter as well


if __name__ == "__main__":
    set_resume_state(False)

