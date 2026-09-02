/**
 * Field holds the discretised operational area: the staggered hexagonal grid,
 * the border and obstacle polygons, and the legality predicates the planner
 * asks about.
 *
 * Port of `src/common/Field.py` plus the polygon parts of `src/common/Config.py`.
 * The grid itself is baked by `web/tools/prepare_data.py` so the browser does
 * not have to redo the point-in-polygon sweep on start-up.
 */
import { PolylineIndex, polygonContains, type Point, type Polygon } from "./geometry.js";

/** Uniform bucket grid standing in for the KDTree the Python code uses. */
class NearestLookup {
  private readonly buckets: Map<number, number[]>;
  private readonly cell: number;

  constructor(private readonly xs: Float64Array, private readonly ys: Float64Array, cell: number) {
    this.cell = cell;
    this.buckets = new Map();
    for (let i = 0; i < xs.length; i++) {
      const key = this.key(xs[i], ys[i]);
      const bucket = this.buckets.get(key);
      if (bucket) bucket.push(i);
      else this.buckets.set(key, [i]);
    }
  }

  private key(x: number, y: number): number {
    // Cantor-ish pairing on signed cell indices, fine for a Map key.
    const i = Math.floor(x / this.cell);
    const j = Math.floor(y / this.cell);
    return i * 100003 + j;
  }

  /** Index of the grid point closest to (x, y). */
  query(x: number, y: number): number {
    const ci = Math.floor(x / this.cell);
    const cj = Math.floor(y / this.cell);
    let best = -1;
    let bestDist = Infinity;
    for (let ring = 1; ring < 64; ring++) {
      for (let i = ci - ring; i <= ci + ring; i++) {
        for (let j = cj - ring; j <= cj + ring; j++) {
          // Only the newly added outer ring needs scanning.
          if (ring > 1 && Math.abs(i - ci) < ring && Math.abs(j - cj) < ring) continue;
          const bucket = this.buckets.get(i * 100003 + j);
          if (!bucket) continue;
          for (const k of bucket) {
            const dx = this.xs[k] - x;
            const dy = this.ys[k] - y;
            const d = dx * dx + dy * dy;
            if (d < bestDist) {
              bestDist = d;
              best = k;
            }
          }
        }
      }
      // One extra ring past the first hit guarantees the true nearest.
      if (best >= 0 && Math.sqrt(bestDist) <= (ring - 1) * this.cell) break;
    }
    return best;
  }
}

export class Field {
  readonly gridX: Float64Array;
  readonly gridY: Float64Array;
  readonly n: number;
  readonly xlim: [number, number];
  readonly ylim: [number, number];

  private readonly lookup: NearestLookup;
  private readonly borderIndex: PolylineIndex;
  private readonly obstacleIndex: PolylineIndex;

  constructor(
    grid: readonly Point[],
    readonly polygonBorder: Polygon,
    readonly polygonObstacle: Polygon,
    neighbourDistance: number,
  ) {
    this.n = grid.length;
    this.gridX = new Float64Array(this.n);
    this.gridY = new Float64Array(this.n);
    for (let i = 0; i < this.n; i++) {
      this.gridX[i] = grid[i][0];
      this.gridY[i] = grid[i][1];
    }
    this.lookup = new NearestLookup(this.gridX, this.gridY, neighbourDistance * 2);
    this.borderIndex = new PolylineIndex(polygonBorder);
    this.obstacleIndex = new PolylineIndex(polygonObstacle);

    let xmin = Infinity;
    let ymin = Infinity;
    let xmax = -Infinity;
    let ymax = -Infinity;
    for (const [x, y] of polygonBorder) {
      xmin = Math.min(xmin, x);
      xmax = Math.max(xmax, x);
      ymin = Math.min(ymin, y);
      ymax = Math.max(ymax, y);
    }
    this.xlim = [xmin, xmax];
    this.ylim = [ymin, ymax];
  }

  /** Index of the grid cell nearest to a location. */
  indexFromLocation(x: number, y: number): number {
    return this.lookup.query(x, y);
  }

  borderContains(x: number, y: number): boolean {
    return polygonContains(this.polygonBorder, x, y);
  }

  obstacleContains(x: number, y: number): boolean {
    return polygonContains(this.polygonObstacle, x, y);
  }

  isBorderInTheWay(ax: number, ay: number, bx: number, by: number): boolean {
    return this.borderIndex.intersects(ax, ay, bx, by);
  }

  isObstacleInTheWay(ax: number, ay: number, bx: number, by: number): boolean {
    return this.obstacleIndex.intersects(ax, ay, bx, by);
  }
}
