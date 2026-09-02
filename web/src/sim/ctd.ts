/**
 * The simulated ocean: one realisation of the field the AUV is trying to map,
 * and the CTD samples it collects while flying between waypoints.
 *
 * Port of `src/simulation/CTD.py`: the truth is drawn from the same GRF prior
 * the agent starts with, `mu_truth = mu_prior + L z`, so the agent's model is
 * correctly specified and only its sampling choices are under test.
 */
import { cholesky, lowerMultiply } from "./linalg.js";
import type { Field } from "./field.js";
import type { Random } from "./random.js";

export class CTD {
  /** The true salinity at every grid cell. */
  readonly muTruth: Float64Array;

  private locX: number;
  private locY: number;

  constructor(
    private readonly field: Field,
    muPrior: Float64Array,
    /** Lower Cholesky factor of the prior covariance; see `CTD.factorise`. */
    choleskyFactor: Float64Array,
    random: Random,
    startX: number,
    startY: number,
  ) {
    const n = field.n;
    const z = random.normals(n);
    const noise = lowerMultiply(choleskyFactor, z, n);
    this.muTruth = new Float64Array(n);
    for (let i = 0; i < n; i++) this.muTruth[i] = muPrior[i] + noise[i];
    this.locX = startX;
    this.locY = startY;
  }

  /**
   * Cholesky factor of a prior covariance matrix. Kept separate so the (heavy)
   * factorisation is done once and reused every time the seed changes.
   */
  static factorise(sigmaPrior: Float64Array, n: number): Float64Array {
    return cholesky(Float64Array.from(sigmaPrior), n);
  }

  /** True salinity at a location, via the nearest grid cell. */
  salinityAt(x: number, y: number): number {
    return this.muTruth[this.field.indexFromLocation(x, y)];
  }

  /**
   * Fly to a location and return the CTD trace along the way as
   * `[x, y, salinity]` rows. The Python code samples 20 points per leg.
   */
  collect(x: number, y: number, samples = 20): number[][] {
    const x0 = this.locX;
    const y0 = this.locY;
    this.locX = x;
    this.locY = y;
    const data: number[][] = [];
    for (let i = 0; i < samples; i++) {
      const t = samples === 1 ? 1 : i / (samples - 1);
      const px = x0 + (x - x0) * t;
      const py = y0 + (y - y0) * t;
      data.push([px, py, this.salinityAt(px, py)]);
    }
    return data;
  }
}
