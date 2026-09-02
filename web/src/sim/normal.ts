/**
 * Normal and bivariate-normal distribution functions.
 *
 * The EIBV of a candidate sampling location is a sum of bivariate normal
 * probabilities over every cell of the field, so the browser evaluates this
 * O(N^2) times per step. The Python code solves that by looking the values up
 * in a precomputed `cdf.npz` table; we do the same, except the table is built
 * on the fly in `BvnTable` from an accurate `bvnd` so no 11 MB asset has to be
 * shipped.
 */

/** Complementary error function, Numerical Recipes 6.2.2 (|eps| < 1.2e-7). */
export function erfc(x: number): number {
  const z = Math.abs(x);
  const t = 2 / (2 + z);
  const ty = 4 * t - 2;
  const cof = [
    -1.3026537197817094, 6.4196979235649026e-1, 1.9476473204185836e-2,
    -9.561514786808631e-3, -9.46595344482036e-4, 3.66839497852761e-4,
    4.2523324806907e-5, -2.0278578112534e-5, -1.624290004647e-6,
    1.303655835580e-6, 1.5626441722e-8, -8.5238095915e-8, 6.529054439e-9,
    5.059343495e-9, -9.91364156e-10, -2.27365122e-10, 9.6467911e-11,
    2.394038e-12, -6.886027e-12, 8.94487e-13, 3.13092e-13, -1.12708e-13,
    3.81e-16, 7.106e-15,
  ];
  let d = 0;
  let dd = 0;
  for (let j = cof.length - 1; j > 0; j--) {
    const tmp = d;
    d = ty * d - dd + cof[j];
    dd = tmp;
  }
  const ans = t * Math.exp(-z * z + 0.5 * (cof[0] + ty * d) - dd);
  return x >= 0 ? ans : 2 - ans;
}

/** Standard normal cumulative distribution function. */
export function normalCdf(x: number): number {
  return 0.5 * erfc(-x * Math.SQRT1_2);
}

/* Gauss-Legendre abscissae and weights used by `bvnd`, 6/12/20 points. */
const GL_W: number[][] = [
  [0.1713244923791705, 0.3607615730481384, 0.4679139345726904],
  [
    0.04717533638651177, 0.1069393259953183, 0.1600783285433464,
    0.2031674267230659, 0.2334925365383547, 0.2491470458134029,
  ],
  [
    0.01761400713915212, 0.04060142980038694, 0.06267204833410906,
    0.08327674157670475, 0.1019301198172404, 0.1181945319615184,
    0.1316886384491766, 0.1420961093183821, 0.1491729864726037,
    0.1527533871307259,
  ],
];
const GL_X: number[][] = [
  [-0.9324695142031522, -0.6612093864662647, -0.238619186083197],
  [
    -0.9815606342467191, -0.904117256370475, -0.769902674194305,
    -0.5873179542866171, -0.3678314989981802, -0.1252334085114692,
  ],
  [
    -0.9931285991850949, -0.9639719272779138, -0.9122344282513259,
    -0.8391169718222188, -0.7463319064601508, -0.636053680726515,
    -0.5108670019508271, -0.3737060887154196, -0.2277858511416451,
    -0.07652652113349733,
  ],
];
const TWO_PI = 2 * Math.PI;

/**
 * P(X > dh, Y > dk) for a standard bivariate normal with correlation r.
 *
 * Port of Alan Genz's BVND (Drezner & Wesolowsky's method with Genz's
 * refinement for |r| near 1), accurate to about 1e-15.
 */
export function bvnd(dh: number, dk: number, r: number): number {
  const ar = Math.abs(r);
  const g = ar < 0.3 ? 0 : ar < 0.75 ? 1 : 2;
  const w = GL_W[g];
  const x = GL_X[g];
  const lg = w.length;

  let h = dh;
  let k = dk;
  let hk = h * k;
  let bvn = 0;

  if (ar < 0.925) {
    const hs = (h * h + k * k) / 2;
    const asr = Math.asin(r);
    for (let i = 0; i < lg; i++) {
      for (const sign of [-1, 1]) {
        const sn = Math.sin((asr * (sign * x[i] + 1)) / 2);
        bvn += w[i] * Math.exp((sn * hk - hs) / (1 - sn * sn));
      }
    }
    bvn = (bvn * asr) / (2 * TWO_PI) + normalCdf(-h) * normalCdf(-k);
    return bvn;
  }

  if (r < 0) {
    k = -k;
    hk = -hk;
  }
  if (ar < 1) {
    const as = (1 - r) * (1 + r);
    let a = Math.sqrt(as);
    const bs = (h - k) * (h - k);
    const c = (4 - hk) / 8;
    const d = (12 - hk) / 16;
    let asr = -(bs / as + hk) / 2;
    if (asr > -100) {
      bvn = a * Math.exp(asr) * (1 - (c * (bs - as) * (1 - (d * bs) / 5)) / 3 + (c * d * as * as) / 5);
    }
    if (-hk < 100) {
      const b = Math.sqrt(bs);
      bvn -= Math.exp(-hk / 2) * Math.sqrt(TWO_PI) * normalCdf(-b / a) * b *
        (1 - (c * bs * (1 - (d * bs) / 5)) / 3);
    }
    a = a / 2;
    for (let i = 0; i < lg; i++) {
      for (const sign of [-1, 1]) {
        const xs = Math.pow(a * (sign * x[i] + 1), 2);
        const rs = Math.sqrt(1 - xs);
        asr = -(bs / xs + hk) / 2;
        if (asr > -100) {
          bvn += a * w[i] * Math.exp(asr) *
            (Math.exp((-hk * xs) / (2 * Math.pow(1 + rs, 2))) / rs - (1 + c * xs * (1 + d * xs)));
        }
      }
    }
    bvn = -bvn / TWO_PI;
  }
  if (r > 0) return bvn + normalCdf(-Math.max(h, k));
  bvn = -bvn;
  return k > h ? bvn + normalCdf(k) - normalCdf(h) : bvn;
}

/** P(X <= a, Y <= b) for a standard bivariate normal with correlation rho. */
export function bvnCdf(a: number, b: number, rho: number): number {
  return bvnd(-a, -b, rho);
}

/**
 * Lookup table for `bvnCdf(h, -h, rho)`, the only shape the EIBV integral needs.
 *
 * Because the two arguments are always antithetic the table is two-dimensional
 * rather than the three-dimensional one the Python code ships, so it is small
 * enough to build at start-up (a couple hundred milliseconds) and is read with
 * bilinear interpolation.
 *
 * The correlation axis is stored as `theta = asin(-rho)` instead of `rho`:
 * along `h = 0` the exact value is `1/4 + asin(rho) / (2 pi)`, which is linear
 * in theta but has an infinite slope in rho as rho approaches -1, so the
 * substitution is what keeps the interpolation error near 1e-6 instead of 1e-3.
 */
export class BvnTable {
  private static readonly H_MAX = 8;
  private static readonly NH = 801;
  private static readonly NTHETA = 401;

  private readonly table: Float64Array;
  private readonly dh: number;
  private readonly dtheta: number;

  constructor() {
    const { NH, NTHETA, H_MAX } = BvnTable;
    this.dh = (2 * H_MAX) / (NH - 1);
    this.dtheta = Math.PI / 2 / (NTHETA - 1);
    this.table = new Float64Array(NH * NTHETA);
    for (let i = 0; i < NH; i++) {
      const h = -H_MAX + i * this.dh;
      for (let j = 0; j < NTHETA; j++) {
        // rho spans [-1, 0]; clamp away from the degenerate endpoints.
        const rho = Math.min(-1e-12, Math.max(-1 + 1e-12, -Math.sin(j * this.dtheta)));
        this.table[i * NTHETA + j] = bvnCdf(h, -h, rho);
      }
    }
  }

  /** Interpolated `bvnCdf(h, -h, rho)`; rho is expected in [-1, 0]. */
  lookup(h: number, rho: number): number {
    const { NH, NTHETA, H_MAX } = BvnTable;
    let fi = (Math.min(H_MAX, Math.max(-H_MAX, h)) + H_MAX) / this.dh;
    let fj = Math.asin(Math.min(1, Math.max(0, -rho))) / this.dtheta;
    const i0 = Math.min(NH - 2, Math.floor(fi));
    const j0 = Math.min(NTHETA - 2, Math.floor(fj));
    fi -= i0;
    fj -= j0;
    const t = this.table;
    const a = t[i0 * NTHETA + j0];
    const b = t[i0 * NTHETA + j0 + 1];
    const c = t[(i0 + 1) * NTHETA + j0];
    const d = t[(i0 + 1) * NTHETA + j0 + 1];
    return (a * (1 - fj) + b * fj) * (1 - fi) + (c * (1 - fj) + d * fj) * fi;
  }
}
