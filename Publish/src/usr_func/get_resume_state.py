"""
This function is used to get the resume state of the system.
"""

import numpy as np


def get_resume_state() -> bool:
    """
    Returns:
        bool: True if the system is in the resume state, False otherwise.

    Examples:
        >>> get_resume_state()
        False

        >>> get_resume_state()
        True

    """
    flag = np.loadtxt("resume_flag.txt")
    if flag == .0:
        return False
    else:
        return True

