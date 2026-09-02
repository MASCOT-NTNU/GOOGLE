"""
Bake the static inputs of the RRTCV simulation into a single JSON asset for the
web port.

The Python simulation reads two kinds of thing from disk at start-up that the
browser cannot reproduce on its own:

- the operational area: ``data/csv/polygon_border.csv`` and
  ``polygon_obstacle.csv``, in WGS84;
- the **prior field**: the mean salinity the GRF starts from, which in the
  Python comes from a SINMOD netCDF.

This script converts the polygons to x/y, rebuilds the hexagonal field grid with
exactly the same construction as ``src/common/Field.py``, samples a prior field
onto that grid with a nearest-neighbour lookup (what ``GRF`` does through its
KDTree), and writes ``web/public/data/field.json``.

The prior is a swappable input. With no arguments it bakes the time-averaged
SINMOD field in ``Publication/prior/sinmod.csv`` and marks the result as
placeholder data; point ``--prior`` at your own file to replace it.

Usage (from the repository root):

    # the default placeholder prior
    python3 web/tools/prepare_data.py

    # your own prior, written to a second asset the page can be pointed at
    python3 web/tools/prepare_data.py \\
        --prior path/to/survey.csv \\
        --label "Nidelva survey, 2024-06-14" \\
        --real \\
        --output web/public/data/field-2024.json

A prior file is a CSV with a header naming either cartesian or geographic
columns, plus one value column:

    x,y,salinity            # x north, y east, metres in the WGS frame below
    lat,lon,salinity        # decimal degrees

Column names are matched case-insensitively; the value column is whatever is
left over (``salinity``, ``value``, ``mean``, ...). Rows with missing values are
dropped. Points outside the operational area are fine - the lookup only ever
asks for the nearest one - but coverage gaps show up as flat patches in the
prior, so aim to cover the whole area.

Author: Yaolin Ge
Email: geyaolin@gmail.com
"""
import argparse
import csv
import json
import os
from datetime import datetime, timezone
from math import cos, radians, sin

import numpy as np
from shapely.geometry import Point, Polygon

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BORDER_CSV = os.path.join(REPO_ROOT, "RRTCV", "data", "csv", "polygon_border.csv")
OBSTACLE_CSV = os.path.join(REPO_ROOT, "RRTCV", "data", "csv", "polygon_obstacle.csv")
DEFAULT_PRIOR = os.path.join(REPO_ROOT, "Publication", "prior", "sinmod.csv")
DEFAULT_OUTPUT = os.path.join(REPO_ROOT, "web", "public", "data", "field.json")

DEFAULT_PRIOR_LABEL = "SINMOD, time-averaged surface salinity"
DEFAULT_PRIOR_SOURCE = "Publication/prior/sinmod.csv"

# The schema the web app expects. Bump it if the shape of the asset changes so a
# stale hand-built file fails loudly instead of rendering nonsense.
SCHEMA_VERSION = 1

# Mirrors src/common/WGS.py
CIRCUMFERENCE = 40075000.0
LATITUDE_ORIGIN = 63.42690974
LONGITUDE_ORIGIN = 10.3969373

# Mirrors src/common/Config.py
NEIGHBOUR_DISTANCE = 80.0    # GRF grid resolution; the Python uses Field(neighbour_distance=100)
WAYPOINT_DISTANCE = 240.0    # distance between AUV waypoints, drives the RRT* step size
WGS_LOC_START = (63.44038447, 10.35675578)  # close to TBS
WGS_LOC_END = (63.440618, 10.355851)        # home location used in the field experiment

X_NAMES = {"x", "north", "northing"}
Y_NAMES = {"y", "east", "easting"}
LAT_NAMES = {"lat", "latitude"}
LON_NAMES = {"lon", "long", "longitude"}


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


def read_prior(filepath: str) -> np.ndarray:
    """
    Read a prior field as an (n, 3) array of x, y, value in the local frame.

    Accepts either cartesian (x/y) or geographic (lat/lon) coordinate columns;
    see the module docstring for the accepted names.
    """
    with open(filepath, newline="") as f:
        reader = csv.reader(f)
        try:
            header = [name.strip().lower() for name in next(reader)]
        except StopIteration:
            raise SystemExit(f"{filepath}: file is empty")
        rows = [row for row in reader if any(cell.strip() for cell in row)]

    def find(names):
        for i, name in enumerate(header):
            if name in names:
                return i
        return None

    ix, iy = find(X_NAMES), find(Y_NAMES)
    ilat, ilon = find(LAT_NAMES), find(LON_NAMES)
    if ix is not None and iy is not None:
        coord_columns, geographic = (ix, iy), False
    elif ilat is not None and ilon is not None:
        coord_columns, geographic = (ilat, ilon), True
    else:
        raise SystemExit(
            f"{filepath}: header {header} has no x/y or lat/lon columns. "
            "Expected one of x,y or lat,lon plus a value column."
        )

    value_columns = [i for i in range(len(header)) if i not in coord_columns]
    if not value_columns:
        raise SystemExit(f"{filepath}: no value column beside the coordinates")
    ivalue = value_columns[0]

    data = []
    for row in rows:
        try:
            a = float(row[coord_columns[0]])
            b = float(row[coord_columns[1]])
            v = float(row[ivalue])
        except (ValueError, IndexError):
            continue  # header repeats, blank cells, NaN markers
        if not all(map(np.isfinite, (a, b, v))):
            continue
        data.append((a, b, v))

    if not data:
        raise SystemExit(f"{filepath}: no usable rows")

    array = np.array(data)
    if geographic:
        x, y = latlon2xy(array[:, 0], array[:, 1])
        array = np.stack((x, y, array[:, 2]), axis=1)
    return array


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


def sample_prior(grid: np.ndarray, prior: np.ndarray) -> tuple:
    """
    Nearest-neighbour lookup of the prior field onto the grid.

    Also returns the largest distance any grid cell had to reach for a value, so
    a prior that does not cover the operational area is reported rather than
    silently smeared across it.
    """
    from scipy.spatial import cKDTree
    tree = cKDTree(prior[:, :2])
    distance, ind = tree.query(grid)
    return prior[ind, 2], float(distance.max())


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bake the RRTCV field and its prior into web/public/data/field.json.",
    )
    parser.add_argument("--prior", default=DEFAULT_PRIOR,
                        help="CSV holding the prior field (x,y,value or lat,lon,value). "
                             f"Default: {os.path.relpath(DEFAULT_PRIOR, REPO_ROOT)}")
    parser.add_argument("--label", default=None,
                        help="Human-readable name for the prior, shown in the page.")
    parser.add_argument("--source", default=None,
                        help="Provenance string recorded in the asset. "
                             "Defaults to the --prior path relative to the repository.")
    parser.add_argument("--units", default="psu", help="Units of the prior values. Default: psu")
    parser.add_argument("--real", dest="placeholder", action="store_false", default=None,
                        help="Mark the prior as real data rather than a placeholder.")
    parser.add_argument("--placeholder", dest="placeholder", action="store_true",
                        help="Mark the prior as placeholder data (the default for the "
                             "bundled SINMOD average).")
    parser.add_argument("--resolution", type=float, default=NEIGHBOUR_DISTANCE,
                        help="Spacing of the GRF grid in metres. Cost grows fast: 100 m is "
                             "1076 cells and a 0.3 s start-up, 80 m is 1690 and 1.2 s, 60 m is "
                             "3009 and 6.8 s. Below ~80 m there is little to resolve anyway - "
                             "the kernel's lateral range is 700 m. "
                             f"Default: {NEIGHBOUR_DISTANCE:.0f}")
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"Where to write the asset. Default: "
                             f"{os.path.relpath(DEFAULT_OUTPUT, REPO_ROOT)}")
    args = parser.parse_args()

    is_default_prior = os.path.abspath(args.prior) == os.path.abspath(DEFAULT_PRIOR)
    placeholder = is_default_prior if args.placeholder is None else args.placeholder
    label = args.label or (DEFAULT_PRIOR_LABEL if is_default_prior
                           else os.path.basename(args.prior))
    source = args.source or (DEFAULT_PRIOR_SOURCE if is_default_prior
                             else os.path.relpath(os.path.abspath(args.prior), REPO_ROOT))

    polygon_border = read_polygon(BORDER_CSV)
    polygon_obstacle = read_polygon(OBSTACLE_CSV)
    grid = construct_grid(polygon_border, polygon_obstacle, args.resolution)

    prior = read_prior(args.prior)
    mu_prior, worst_distance = sample_prior(grid, prior)

    x_start, y_start = latlon2xy(*WGS_LOC_START)
    x_end, y_end = latlon2xy(*WGS_LOC_END)

    payload = {
        "schemaVersion": SCHEMA_VERSION,
        "generatedBy": "web/tools/prepare_data.py",
        "generatedAt": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "wgsOrigin": [LATITUDE_ORIGIN, LONGITUDE_ORIGIN],
        "neighbourDistance": args.resolution,
        "waypointDistance": WAYPOINT_DISTANCE,
        "locStart": [round(float(x_start), 3), round(float(y_start), 3)],
        "locEnd": [round(float(x_end), 3), round(float(y_end), 3)],
        "polygonBorder": [[round(float(x), 2), round(float(y), 2)] for x, y in polygon_border],
        "polygonObstacle": [[round(float(x), 2), round(float(y), 2)] for x, y in polygon_obstacle],
        "grid": [[round(float(x), 2), round(float(y), 2)] for x, y in grid],
        "prior": {
            "label": label,
            "source": source,
            "units": args.units,
            "placeholder": placeholder,
            "points": int(prior.shape[0]),
            "range": [round(float(mu_prior.min()), 4), round(float(mu_prior.max()), 4)],
        },
        "muPrior": [round(float(v), 4) for v in mu_prior],
    }

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print("grid cells:      ", len(grid), "at {:.0f} m spacing".format(args.resolution))
    print("border vertices: ", len(polygon_border))
    print("prior:           ", label, "(placeholder)" if placeholder else "(real data)")
    print("prior points:    ", prior.shape[0])
    print("prior range:      {:.2f} - {:.2f} {}".format(mu_prior.min(), mu_prior.max(), args.units))
    if worst_distance > 2 * args.resolution:
        print("WARNING:          a grid cell is {:.0f} m from the nearest prior point; "
              "the prior may not cover the whole operational area".format(worst_distance))
    print("written:         ", args.output,
          "({:.1f} kB)".format(os.path.getsize(args.output) / 1024))


if __name__ == "__main__":
    main()
