""" 
This function tests if a given integer is even or not. 
"""


def is_even(value: int) -> bool:
    """
    Checks if a given integer even or not.

    Args:
        value: integer value

    Returns: True if it is even, False or else.

    """
    if value % 2 == 0:
        return True
    else:
        return False

