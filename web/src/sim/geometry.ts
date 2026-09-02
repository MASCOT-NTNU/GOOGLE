/**
 * Planar geometry helpers.
 *
 * The Python code leans on shapely for point-in-polygon and segment/boundary
 * intersection tests. These are the two predicates it actually uses, written
 * directly so the browser needs no geometry library.
 *
 * Coordinates are the simulation's local north/east metres: `[x, y]` with x
 * pointing north and y east, matching `WGS.latlon2xy`.
 */

export type Point = readonly [number, number];
export type Polygon = readonly Point[];

/** Ray-casting point-in-polygon test, matching `Polygon.contains`. */
export function polygonContains(polygon: Polygon, x: number, y: number): boolean {
  let inside = false;
  for (let i = 0, j = polygon.length - 1; i < polygon.length; j = i++) {
    const [xi, yi] = polygon[i];
    const [xj, yj] = polygon[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
      inside = !inside;
    }
  }
  return inside;
}

function orientation(ax: number, ay: number, bx: number, by: number, cx: number, cy: number): number {
  const v = (by - ay) * (cx - bx) - (bx - ax) * (cy - by);
  if (v > 0) return 1;
  if (v < 0) return -1;
  return 0;
}

function onSegment(ax: number, ay: number, bx: number, by: number, cx: number, cy: number): boolean {
  return (
    Math.min(ax, bx) <= cx && cx <= Math.max(ax, bx) &&
    Math.min(ay, by) <= cy && cy <= Math.max(ay, by)
  );
}

/** True if segments (a1,a2) and (b1,b2) intersect, endpoints included. */
export function segmentsIntersect(
  a1x: number, a1y: number, a2x: number, a2y: number,
  b1x: number, b1y: number, b2x: number, b2y: number,
): boolean {
  const o1 = orientation(a1x, a1y, a2x, a2y, b1x, b1y);
  const o2 = orientation(a1x, a1y, a2x, a2y, b2x, b2y);
  const o3 = orientation(b1x, b1y, b2x, b2y, a1x, a1y);
  const o4 = orientation(b1x, b1y, b2x, b2y, a2x, a2y);
  if (o1 !== o2 && o3 !== o4) return true;
  if (o1 === 0 && onSegment(a1x, a1y, a2x, a2y, b1x, b1y)) return true;
  if (o2 === 0 && onSegment(a1x, a1y, a2x, a2y, b2x, b2y)) return true;
  if (o3 === 0 && onSegment(b1x, b1y, b2x, b2y, a1x, a1y)) return true;
  if (o4 === 0 && onSegment(b1x, b1y, b2x, b2y, a2x, a2y)) return true;
  return false;
}

/**
 * True if the segment crosses the polygon's boundary, matching shapely's
 * `LineString(polygon).intersects(line)` on the closed ring.
 *
 * A uniform grid of the ring's segments keeps this cheap: RRT* runs the test
 * a few thousand times per planning call against a 319-vertex border.
 */
export class PolylineIndex {
  private readonly cells = new Map<number, number[]>();
  private readonly xs: Float64Array;
  private readonly ys: Float64Array;
  private readonly n: number;
  private readonly cell: number;
  private readonly x0: number;
  private readonly y0: number;
  private readonly nx: number;
  private readonly ny: number;

  constructor(ring: Polygon, cellSize = 200) {
    // Close the ring so the last segment back to the first vertex is included.
    const closed = ring[0][0] === ring[ring.length - 1][0] && ring[0][1] === ring[ring.length - 1][1]
      ? ring
      : [...ring, ring[0]];
    this.n = closed.length - 1;
    this.xs = new Float64Array(closed.length);
    this.ys = new Float64Array(closed.length);
    for (let i = 0; i < closed.length; i++) {
      this.xs[i] = closed[i][0];
      this.ys[i] = closed[i][1];
    }
    this.cell = cellSize;
    let xmin = Infinity;
    let ymin = Infinity;
    let xmax = -Infinity;
    let ymax = -Infinity;
    for (let i = 0; i < closed.length; i++) {
      xmin = Math.min(xmin, this.xs[i]);
      xmax = Math.max(xmax, this.xs[i]);
      ymin = Math.min(ymin, this.ys[i]);
      ymax = Math.max(ymax, this.ys[i]);
    }
    this.x0 = xmin - cellSize;
    this.y0 = ymin - cellSize;
    this.nx = Math.ceil((xmax - xmin) / cellSize) + 3;
    const ny = Math.ceil((ymax - ymin) / cellSize) + 3;
    for (let s = 0; s < this.n; s++) {
      const i0 = Math.max(0, Math.floor((Math.min(this.xs[s], this.xs[s + 1]) - this.x0) / cellSize));
      const i1 = Math.min(this.nx - 1, Math.floor((Math.max(this.xs[s], this.xs[s + 1]) - this.x0) / cellSize));
      const j0 = Math.max(0, Math.floor((Math.min(this.ys[s], this.ys[s + 1]) - this.y0) / cellSize));
      const j1 = Math.min(ny - 1, Math.floor((Math.max(this.ys[s], this.ys[s + 1]) - this.y0) / cellSize));
      for (let i = i0; i <= i1; i++) {
        for (let j = j0; j <= j1; j++) {
          const key = i * ny + j;
          const bucket = this.cells.get(key);
          if (bucket) bucket.push(s);
          else this.cells.set(key, [s]);
        }
      }
    }
    this.ny = ny;
  }

  intersects(ax: number, ay: number, bx: number, by: number): boolean {
    const c = this.cell;
    const i0 = Math.floor((Math.min(ax, bx) - this.x0) / c);
    const i1 = Math.floor((Math.max(ax, bx) - this.x0) / c);
    const j0 = Math.floor((Math.min(ay, by) - this.y0) / c);
    const j1 = Math.floor((Math.max(ay, by) - this.y0) / c);
    const seen = new Set<number>();
    for (let i = Math.max(0, i0); i <= Math.min(this.nx - 1, i1); i++) {
      for (let j = Math.max(0, j0); j <= Math.min(this.ny - 1, j1); j++) {
        const bucket = this.cells.get(i * this.ny + j);
        if (!bucket) continue;
        for (const s of bucket) {
          if (seen.has(s)) continue;
          seen.add(s);
          if (segmentsIntersect(ax, ay, bx, by, this.xs[s], this.ys[s], this.xs[s + 1], this.ys[s + 1])) {
            return true;
          }
        }
      }
    }
    return false;
  }
}
