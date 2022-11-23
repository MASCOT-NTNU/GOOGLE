"""
This function will get the global resume state
"""

import numpy as np


def get_resume_state() -> bool:
    flag = np.loadtxt("resume_flag.txt")
    if flag == .0:
        return False
    else:
        return True

