"""
The function is used to sort the polygon counter-clockwisely.
"""

import numpy as np
import math


def sort_polygon_vertices(polygon: np.ndarray) -> np.ndarray:
    """
    Args: vertices of the polygon is not organised in order.
        polygon: np.array([[x1, y1, z1],

    Returns:
        sorted polygon with the direction of counter-clockwise.

    Examples:
        >>> polygon = np.array([[0, 0, 0],
        ...                     [0, 1, 0],
        ...                     [1, 1, 0],
        ...                     [1, 0, 0]])
        >>> sort_polygon_vertices(polygon)
        array([[0, 0, 0],
                [1, 0, 0],
                [1, 1, 0],
                [0, 1, 0]])

    """
    polygon = list(polygon)
    cent = (sum([vertice[0] for vertice in polygon]) / len(polygon),
            sum([vertice[1] for vertice in polygon]) / len(polygon))
    polygon.sort(key=lambda vertice: math.atan2(vertice[1] - cent[1], vertice[0] - cent[0]))
    return np.array(polygon)
