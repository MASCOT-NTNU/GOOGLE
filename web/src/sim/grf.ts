/**
 * Gaussian random field kernel: the prior belief about salinity over the field,
 * conditioned on everything the AUV has measured so far.
 *
 * Port of `src/common/GRF/GRF.py`. Two differences from the Python, both forced
 * by what is available in the browser:
 *
 * - The prior mean comes from the baked SINMOD field (`web/public/data/field.json`)
 *   rather than the SINMOD netCDF, so the static rather than the AR(1) temporal
 *   variant of the model is used - `assimilate_data` / `__update`, not
 *   `assimilate_temporal_data`.
 * - `getEiField` follows the `fast_eibv` path: the EIBV of a candidate is
 *   `sum_j Phi2(mur_j, -mur_j; rho_j)` with `mur = (threshold - mu) / sigma_post`
 *   and `rho = -variance_reduction / sigma_prior^2`, the default the published
 *   simulations ran with. The bivariate CDF is interpolated from `BvnTable`
 *   instead of the shipped `cdf.npz`.
 */
import { BvnTable } from "./normal.js";
import { solveSpd } from "./linalg.js";
import type { Field } from "./field.js";

export interface GrfParameters {
  /** Marginal standard deviation of the field. */
  sigma: number;
  /** Distance at which the Matern correlation has decayed to ~0.1, in metres. */
  lateralRange: number;
  /** Measurement noise variance. */
  nugget: number;
  /** Salinity threshold separating the two classes the survey cares about. */
  threshold: number;
}

export const DEFAULT_GRF_PARAMETERS: GrfParameters = {
  sigma: 1.0,
  lateralRange: 700,
  nugget: 0.25,
  threshold: 26.81189868,
};

export class GRF {
  readonly n: number;
  /** Conditional mean, length n. */
  readonly mu: Float64Array;
  /** Conditional covariance, n x n row-major. */
  readonly sigma: Float64Array;
  /** Prior covariance, kept for resampling the truth field. */
  readonly sigmaPrior: Float64Array;
  readonly muPrior: Float64Array;

  readonly eibvField: Float64Array;
  readonly ivrField: Float64Array;

  private readonly bvn: BvnTable;
  private readonly params: GrfParameters;

  constructor(
    private readonly field: Field,
    muPrior: Float64Array,
    params: GrfParameters,
    bvn: BvnTable,
    /** Prior covariance, if it has already been built for these parameters. */
    sigmaPrior?: Float64Array,
  ) {
    this.n = field.n;
    this.params = params;
    this.bvn = bvn;
    this.muPrior = muPrior;
    this.mu = Float64Array.from(muPrior);
    this.sigmaPrior = sigmaPrior ?? buildCovariance(field, params.sigma, params.lateralRange);
    this.sigma = Float64Array.from(this.sigmaPrior);
    this.eibvField = new Float64Array(this.n);
    this.ivrField = new Float64Array(this.n);
  }

  /** Reset the conditional field back to the prior. */
  reset(): void {
    this.mu.set(this.muPrior);
    this.sigma.set(this.sigmaPrior);
  }

  /**
   * Assimilate a batch of measurements, `[x, y, salinity]` triples.
   *
   * Measurements are averaged per grid cell first, exactly as
   * `GRF.assimilate_data` does, then applied as one conditional update.
   */
  assimilate(data: readonly number[][]): void {
    if (data.length === 0) return;
    const sums = new Map<number, { sum: number; count: number }>();
    for (const [x, y, salinity] of data) {
      const ind = this.field.indexFromLocation(x, y);
      const entry = sums.get(ind);
      if (entry) {
        entry.sum += salinity;
        entry.count += 1;
      } else {
        sums.set(ind, { sum: salinity, count: 1 });
      }
    }
    const indices = [...sums.keys()].sort((a, b) => a - b);
    const values = indices.map((i) => sums.get(i)!.sum / sums.get(i)!.count);
    this.update(indices, values);
  }

  /**
   * Conditional update at the measured cells.
   *
   * Same algebra as `GRF.__update`, written without the m x n selection matrix:
   *   mu    <- mu + S[:, I] C^-1 (z - mu[I])
   *   Sigma <- Sigma - S[:, I] C^-1 S[I, :]
   * with C = Sigma[I, I] + tau^2 I.
   */
  private update(indices: number[], values: number[]): void {
    const { n, sigma, mu } = this;
    const m = indices.length;
    const tau2 = this.params.nugget;

    // C = Sigma[I, I] + tau^2 I (a scratch copy, solveSpd factorises in place)
    const c = new Float64Array(m * m);
    for (let a = 0; a < m; a++) {
      for (let b = 0; b < m; b++) c[a * m + b] = sigma[indices[a] * n + indices[b]];
      c[a * m + a] += tau2;
    }

    // rhs = [innovation | Sigma[I, :]] solved together: (m x (1 + n))
    const k = 1 + n;
    const rhs = new Float64Array(m * k);
    for (let a = 0; a < m; a++) {
      rhs[a * k] = values[a] - mu[indices[a]];
      const row = indices[a] * n;
      for (let j = 0; j < n; j++) rhs[a * k + 1 + j] = sigma[row + j];
    }
    solveSpd(c, m, rhs, k);

    for (let i = 0; i < n; i++) {
      const row = i * n;
      let dmu = 0;
      for (let a = 0; a < m; a++) {
        const sia = sigma[row + indices[a]];
        if (sia === 0) continue;
        dmu += sia * rhs[a * k];
      }
      mu[i] += dmu;
    }
    // Sigma update needs the pre-update column values, so snapshot them first.
    const cols = new Float64Array(n * m);
    for (let i = 0; i < n; i++) {
      for (let a = 0; a < m; a++) cols[i * m + a] = sigma[i * n + indices[a]];
    }
    for (let i = 0; i < n; i++) {
      const row = i * n;
      const ci = i * m;
      for (let a = 0; a < m; a++) {
        const sia = cols[ci + a];
        if (sia === 0) continue;
        const rrow = a * k + 1;
        for (let j = 0; j < n; j++) sigma[row + j] -= sia * rhs[rrow + j];
      }
    }
  }

  /**
   * Recompute the EIBV and IVR reward fields.
   *
   * For every candidate cell i, sampling there would reduce the variance at
   * cell j by `VR_j = Sigma[j, i]^2 / (Sigma[i, i] + nugget)`. EIBV sums the
   * expected Bernoulli variance that remains; IVR sums the variance removed
   * and is flipped so that, like EIBV, low is good. Both are min-max
   * normalised, as `GRF.get_ei_field` does.
   */
  getEiField(): void {
    const { n, sigma, mu, eibvField, ivrField } = this;
    const { threshold, nugget } = this.params;
    const bvn = this.bvn;

    // Prior variance per cell and its reciprocal square root, hoisted out.
    const varJ = new Float64Array(n);
    for (let j = 0; j < n; j++) varJ[j] = sigma[j * n + j];

    for (let i = 0; i < n; i++) {
      const md = 1 / (varJ[i] + nugget);
      const row = i * n; // Sigma is symmetric, so row i is column i
      let eibv = 0;
      let ivr = 0;
      for (let j = 0; j < n; j++) {
        const sf = sigma[row + j];
        const vr = sf * sf * md;
        ivr += vr;
        const post = varJ[j] - vr;
        const mur = (threshold - mu[j]) / Math.sqrt(post > 1e-12 ? post : 1e-12);
        eibv += bvn.lookup(mur, -vr / varJ[j]);
      }
      eibvField[i] = eibv;
      ivrField[i] = ivr;
    }
    normalizeInPlace(eibvField);
    normalizeInPlace(ivrField);
    for (let i = 0; i < n; i++) ivrField[i] = 1 - ivrField[i];
  }

  /** Marginal standard deviation per cell. */
  standardDeviation(out: Float64Array): Float64Array {
    for (let i = 0; i < this.n; i++) out[i] = Math.sqrt(this.sigma[i * this.n + i]);
    return out;
  }

  /** Total remaining variance, the `vr` metric logged by the Python agent. */
  totalVariance(): number {
    let s = 0;
    for (let i = 0; i < this.n; i++) s += this.sigma[i * this.n + i];
    return s;
  }

  get threshold(): number {
    return this.params.threshold;
  }
}

/**
 * Matern (nu = 3/2) covariance over the field grid, `src/common/GRF/GRF.py`'s
 * `__construct_grf_field`. Exported so the worker can build it once and share
 * it between the agent and the Cholesky factorisation of the truth field.
 */
export function buildCovariance(field: Field, sigma: number, lateralRange: number): Float64Array {
  const n = field.n;
  const eta = 4.5 / lateralRange;
  const s2 = sigma * sigma;
  const cov = new Float64Array(n * n);
  const x = field.gridX;
  const y = field.gridY;
  for (let i = 0; i < n; i++) {
    cov[i * n + i] = s2;
    for (let j = i + 1; j < n; j++) {
      const dx = x[i] - x[j];
      const dy = y[i] - y[j];
      const ed = eta * Math.sqrt(dx * dx + dy * dy);
      const c = s2 * (1 + ed) * Math.exp(-ed);
      cov[i * n + j] = c;
      cov[j * n + i] = c;
    }
  }
  return cov;
}

/** Min-max normalisation to [0, 1], matching `usr_func/normalize.py`. */
export function normalizeInPlace(v: Float64Array): void {
  let min = Infinity;
  let max = -Infinity;
  for (let i = 0; i < v.length; i++) {
    if (v[i] < min) min = v[i];
    if (v[i] > max) max = v[i];
  }
  const span = max - min;
  if (span <= 0) {
    v.fill(0);
    return;
  }
  for (let i = 0; i < v.length; i++) v[i] = (v[i] - min) / span;
}
