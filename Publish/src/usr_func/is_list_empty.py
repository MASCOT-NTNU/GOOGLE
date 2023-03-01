"""
Checks if a given list is empty or not.
"""

def is_list_empty(glist: list) -> bool:
    """
    Args:
        glist:  can be nested list such as [], [[]], [[[]]] etc.

    Returns:
        True if it is empty, such as [], [[]], [[[]]], etc.

    Examples:
        >>> is_list_empty([])
        True
        >>> is_list_empty([[]])
        True
        >>> is_list_empty([[[]]])
        True
        >>> is_list_empty([1, 2, 3])
        False
        >>> is_list_empty([1, 2, []])
        False

    """
    if isinstance(glist, list):
        return all(map(is_list_empty, glist))
    return False


