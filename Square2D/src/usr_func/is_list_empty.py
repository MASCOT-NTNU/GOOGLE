"""
This function tests if a given list is empty or not.
"""

def is_list_empty(glist):
    """
    Tests if a given list is empty or not.

    Args:
        glist: can be nested list such as [], [[]], [[[]]] etc.

    Returns: True if it is empty, such as [], [[]], [[[]]], etc.

    """
    if isinstance(glist, list):
        return all(map(is_list_empty, glist))
    return False


