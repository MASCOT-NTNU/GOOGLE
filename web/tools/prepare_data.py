"""
Bake the static inputs of the RRTCV simulation into a single JSON asset for the
web port.

The Python simulation reads three things from disk at start-up that the browser
cannot reproduce on its own:

- ``data/csv/polygon_border.csv`` and ``polygon_obstacle.csv``: the operational
  area in WGS84.
- ``Publication/prior/sinmod.csv``: the SINMOD prior salinity field, in the same
  x/y (north/east) frame the simulation uses.

This script converts the polygons to x/y, rebuilds the hexagonal field grid with
exactly the same construction as ``src/common/Field.py``, samples the SINMOD
prior onto that grid with a nearest-neighbour lookup (what ``GRF`` does through
its KDTree), and writes ``web/public/data/field.json``.

Usage (from the repository root):
    python3 web/tools/prepare_data.py

Author: Yaolin Ge
Email: geyaolin@gmail.com
"""
import json
import os
from math import cos, sin, radians

import numpy as np
from shapely.geometry import Point, Polygon

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BORDER_CSV = os.path.join(REPO_ROOT, "RRTCV", "data", "csv", "polygon_border.csv")
OBSTACLE_CSV = os.path.join(REPO_ROOT, "RRTCV", "data", "csv", "polygon_obstacle.csv")
SINMOD_CSV = os.path.join(REPO_ROOT, "Publication", "prior", "sinmod.csv")
OUTPUT = os.path.join(REPO_ROOT, "web", "public", "data", "field.json")

# Mirrors src/common/WGS.py
CIRCUMFERENCE = 40075000.0
LATITUDE_ORIGIN = 63.42690974
LONGITUDE_ORIGIN = 10.3969373

# Mirrors src/common/Config.py
NEIGHBOUR_DISTANCE = 100.0   # GRF grid resolution, Field(neighbour_distance=100)
WAYPOINT_DISTANCE = 240.0    # distance between AUV waypoints, drives the RRT* step size
WGS_LOC_START = (63.44038447, 10.35675578)  # close to TBS
WGS_LOC_END = (63.440618, 10.355851)        # home location used in the field experiment


def latlon2xy(lat, lon):
    """ WGS84 to local north/east metres, identical to WGS.latlon2xy. """
    x = np.radians(lat - LATITUDE_ORIGIN) / 2 / np.pi * CIRCUMFERENCE
    y = np.radians(lon - LONGITUDE_ORIGIN) / 2 / np.pi * CIRCUMFERENCE * np.cos(np.radians(lat))
    return x, y


def read_polygon(filepath: str) -> np.ndarray:
    """ Read a lat/lon polygon csv and return it as x/y metres. """
    latlon = np.genfromtxt(filepath, delimiter=",", skip_header=1)
    x, y = latlon2xy(latlon[:, 0], latlon[:, 1])
    return np.stack((x, y), axis=1)


def construct_grid(polygon_border: np.ndarray, polygon_obstacle: np.ndarray,
                   neighbour_distance: float) -> np.ndarray:
    """ Rebuild the staggered hexagonal grid exactly as Field.__construct_grid does. """
    border_shapely = Polygon(polygon_border)
    obstacle_shapely = Polygon(polygon_obstacle)

    xmin, ymin = polygon_border.min(axis=0)
    xmax, ymax = polygon_border.max(axis=0)
    ygap = neighbour_distance * cos(radians(60)) * 2
    xgap = neighbour_distance * sin(radians(60))

    gx = np.arange(xmin, xmax, xgap)
    gy = np.arange(ymin, ymax, ygap)

    grid = []
    for i in range(len(gx)):
        for j in range(len(gy)):
            y = gy[j] + ygap / 2 if i % 2 == 0 else gy[j]
            x = gx[i]
            point = Point(x, y)
            if border_shapely.contains(point) and not obstacle_shapely.contains(point):
                grid.append([x, y])
    return np.array(grid)


def sample_prior(grid: np.ndarray, sinmod: np.ndarray) -> np.ndarray:
    """ Nearest-neighbour lookup of the SINMOD prior mean onto the field grid. """
    from scipy.spatial import cKDTree
    tree = cKDTree(sinmod[:, :2])
    _, ind = tree.query(grid)
    return sinmod[ind, 2]


def main() -> None:
    polygon_border = read_polygon(BORDER_CSV)
    polygon_obstacle = read_polygon(OBSTACLE_CSV)
    grid = construct_grid(polygon_border, polygon_obstacle, NEIGHBOUR_DISTANCE)

    sinmod = np.genfromtxt(SINMOD_CSV, delimiter=",", skip_header=1)
    sinmod = sinmod[~np.isnan(sinmod).any(axis=1)]
    mu_prior = sample_prior(grid, sinmod)

    x_start, y_start = latlon2xy(*WGS_LOC_START)
    x_end, y_end = latlon2xy(*WGS_LOC_END)

    payload = {
        "generatedBy": "web/tools/prepare_data.py",
        "wgsOrigin": [LATITUDE_ORIGIN, LONGITUDE_ORIGIN],
        "neighbourDistance": NEIGHBOUR_DISTANCE,
        "waypointDistance": WAYPOINT_DISTANCE,
        "locStart": [round(float(x_start), 3), round(float(y_start), 3)],
        "locEnd": [round(float(x_end), 3), round(float(y_end), 3)],
        "polygonBorder": [[round(float(x), 2), round(float(y), 2)] for x, y in polygon_border],
        "polygonObstacle": [[round(float(x), 2), round(float(y), 2)] for x, y in polygon_obstacle],
        "grid": [[round(float(x), 2), round(float(y), 2)] for x, y in grid],
        "muPrior": [round(float(v), 4) for v in mu_prior],
    }

    os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
    with open(OUTPUT, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print("grid cells:      ", len(grid))
    print("border vertices: ", len(polygon_border))
    print("prior range:      {:.2f} - {:.2f}".format(mu_prior.min(), mu_prior.max()))
    print("written:         ", OUTPUT, "({:.1f} kB)".format(os.path.getsize(OUTPUT) / 1024))


if __name__ == "__main__":
    main()
