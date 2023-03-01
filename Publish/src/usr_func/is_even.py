"""
Checks if a given integer even or not.
"""


def is_even(value: int) -> bool:
    """
    Args:
        value (int): The integer to check.

    Returns:
        bool: True if the integer is even, False otherwise.

    Examples:
        >>> is_even(2)
        True
        >>> is_even(3)
        False

    """
    if value % 2 == 0:
        return True
    else:
        return False

