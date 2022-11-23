"""
This function sorts the polygon with the counter-clockwise direction.
"""

import numpy as np
import math


def sort_polygon_vertices(polygon: np.ndarray) -> np.ndarray:
    """
    Sort the polygon counter-clockwisely.

    Args: vertices of the polygon is not organised in order.
        polygon: np.array([[x1, y1, z1],
                           [x2, y2, z2],
                           ...
                           [xn, yn, zn]])

    Returns: sorted polygon with the direction of counter-clockwise.
    """
    polygon = list(polygon)
    cent = (sum([vertice[0] for vertice in polygon]) / len(polygon),
            sum([vertice[1] for vertice in polygon]) / len(polygon))
    polygon.sort(key=lambda vertice: math.atan2(vertice[1] - cent[1], vertice[0] - cent[0]))
    return np.array(polygon)
