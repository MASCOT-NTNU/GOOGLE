/**
 * Resampling from the simulation's staggered hexagonal grid onto a regular
 * lattice.
 *
 * The panels used to be drawn as one rectangle per grid cell, which made the
 * discretisation the most visible thing on screen. Interpolating instead is
 * also the more honest picture: the GRF's lateral range is 700 m, so the field
 * genuinely carries no structure at the scale of the block edges the rectangles
 * were drawing.
 *
 * Weights are Gaussian rather than inverse-distance, which avoids the bullseye
 * artefact IDW leaves at every data point, and they are computed once at
 * construction - only the values change from frame to frame, so redrawing a
 * panel is a weighted sum over a precomputed sparse matrix.
 */

export interface Bbox {
  xmin: number;
  xmax: number;
  ymin: number;
  ymax: number;
}

export class GridInterpolator {
  /** Lattice dimensions: `nx` columns of east, `ny` rows of north. */
  readonly nx: number;
  readonly ny: number;
  /** World coordinate of each lattice node, row-major, row 0 = north edge. */
  readonly worldX: Float64Array;
  readonly worldY: Float64Array;
  /** False where no grid cell is near enough for a value to mean anything. */
  readonly inside: Uint8Array;

  /** CSR-style contributions: node i draws on indices[offsets[i]..offsets[i+1]]. */
  private readonly offsets: Int32Array;
  private readonly indices: Int32Array;
  private readonly weights: Float64Array;

  constructor(
    gridX: Float64Array,
    gridY: Float64Array,
    bbox: Bbox,
    nx: number,
    ny: number,
    spacing: number,
    /** How far outside the grid's own footprint a node may still be filled. */
    maskRadius = spacing * 0.9,
    /**
     * Optional exact test for whether a node belongs to the field. The distance
     * mask alone scallops the outline around the outermost grid cells, which is
     * invisible behind a canvas clip path but shows up as a stepped silhouette
     * on the 3D surface, where there is no clip to hide it.
     */
    insideTest?: (x: number, y: number) => boolean,
  ) {
    this.nx = nx;
    this.ny = ny;
    const n = nx * ny;
    this.worldX = new Float64Array(n);
    this.worldY = new Float64Array(n);
    this.inside = new Uint8Array(n);

    const radius = spacing * 1.7;
    const sigma = spacing * 0.6;
    const twoSigmaSq = 2 * sigma * sigma;

    // Bucket the grid so each lattice node only tests nearby cells.
    const bucketSize = radius;
    const buckets = new Map<number, number[]>();
    const keyOf = (x: number, y: number) =>
      Math.floor(x / bucketSize) * 100003 + Math.floor(y / bucketSize);
    for (let i = 0; i < gridX.length; i++) {
      const key = keyOf(gridX[i], gridY[i]);
      const bucket = buckets.get(key);
      if (bucket) bucket.push(i);
      else buckets.set(key, [i]);
    }

    const offsets = new Int32Array(n + 1);
    const indices: number[] = [];
    const weights: number[] = [];
    const radiusSq = radius * radius;
    const maskRadiusSq = maskRadius * maskRadius;

    for (let row = 0; row < ny; row++) {
      // Row 0 is the north edge so the lattice maps straight onto screen rows.
      const x = bbox.xmax - ((bbox.xmax - bbox.xmin) * row) / (ny - 1);
      for (let col = 0; col < nx; col++) {
        const y = bbox.ymin + ((bbox.ymax - bbox.ymin) * col) / (nx - 1);
        const node = row * nx + col;
        this.worldX[node] = x;
        this.worldY[node] = y;
        offsets[node] = indices.length;

        let total = 0;
        let nearestSq = Infinity;
        const first = indices.length;
        const bi = Math.floor(x / bucketSize);
        const bj = Math.floor(y / bucketSize);
        for (let a = bi - 1; a <= bi + 1; a++) {
          for (let b = bj - 1; b <= bj + 1; b++) {
            const bucket = buckets.get(a * 100003 + b);
            if (!bucket) continue;
            for (const k of bucket) {
              const dx = gridX[k] - x;
              const dy = gridY[k] - y;
              const d2 = dx * dx + dy * dy;
              if (d2 > radiusSq) continue;
              if (d2 < nearestSq) nearestSq = d2;
              const w = Math.exp(-d2 / twoSigmaSq);
              indices.push(k);
              weights.push(w);
              total += w;
            }
          }
        }
        if (total > 0) {
          for (let p = first; p < indices.length; p++) weights[p] /= total;
        }
        const near = total > 0 && nearestSq <= maskRadiusSq;
        this.inside[node] = near && (!insideTest || insideTest(x, y)) ? 1 : 0;
      }
    }
    offsets[n] = indices.length;

    this.offsets = offsets;
    this.indices = Int32Array.from(indices);
    this.weights = Float64Array.from(weights);
  }

  /**
   * Resample `values` onto the lattice. Masked-out nodes come back as NaN so a
   * caller can leave them transparent or skip the quad that touches them.
   */
  apply(values: Float64Array, out: Float64Array): Float64Array {
    const { offsets, indices, weights, inside } = this;
    for (let node = 0; node < inside.length; node++) {
      if (!inside[node]) {
        out[node] = NaN;
        continue;
      }
      let sum = 0;
      for (let p = offsets[node]; p < offsets[node + 1]; p++) {
        sum += values[indices[p]] * weights[p];
      }
      out[node] = sum;
    }
    return out;
  }
}

/** Bounding box of a polygon given as [north, east] pairs. */
export function bboxOf(polygon: readonly (readonly [number, number])[]): Bbox {
  let xmin = Infinity;
  let xmax = -Infinity;
  let ymin = Infinity;
  let ymax = -Infinity;
  for (const [x, y] of polygon) {
    if (x < xmin) xmin = x;
    if (x > xmax) xmax = x;
    if (y < ymin) ymin = y;
    if (y > ymax) ymax = y;
  }
  return { xmin, xmax, ymin, ymax };
}
