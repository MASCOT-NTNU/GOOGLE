/**
 * Dense linear algebra on flat Float64Arrays (row-major).
 *
 * Only the handful of operations the GRF needs: a Cholesky factorisation for
 * sampling the truth field, and a small symmetric solve for the Kalman-style
 * conditional update.
 */

/**
 * In-place lower-triangular Cholesky factor of a symmetric positive definite
 * matrix `a` (n x n, row-major). Returns the same buffer with the strict upper
 * triangle zeroed. A small jitter is added to the diagonal if a pivot turns
 * non-positive, which can happen for a near-singular covariance matrix.
 */
export function cholesky(a: Float64Array, n: number): Float64Array {
  for (let j = 0; j < n; j++) {
    let d = a[j * n + j];
    for (let k = 0; k < j; k++) d -= a[j * n + k] * a[j * n + k];
    if (d <= 0) d = 1e-12;
    const ljj = Math.sqrt(d);
    a[j * n + j] = ljj;
    for (let i = j + 1; i < n; i++) {
      let s = a[i * n + j];
      const ri = i * n;
      const rj = j * n;
      for (let k = 0; k < j; k++) s -= a[ri + k] * a[rj + k];
      a[ri + j] = s / ljj;
    }
    for (let k = j + 1; k < n; k++) a[j * n + k] = 0;
  }
  return a;
}

/** y = L x for a lower-triangular L (n x n) and vector x. */
export function lowerMultiply(l: Float64Array, x: Float64Array, n: number): Float64Array {
  const out = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    let s = 0;
    const ri = i * n;
    for (let j = 0; j <= i; j++) s += l[ri + j] * x[j];
    out[i] = s;
  }
  return out;
}

/**
 * Solve `A X = B` for a symmetric positive definite A (m x m) and B (m x k),
 * both row-major. B is overwritten with the solution and returned.
 */
export function solveSpd(a: Float64Array, m: number, b: Float64Array, k: number): Float64Array {
  const l = cholesky(a, m); // a is a scratch copy owned by the caller
  // forward substitution: L Y = B
  for (let i = 0; i < m; i++) {
    for (let c = 0; c < k; c++) {
      let s = b[i * k + c];
      for (let j = 0; j < i; j++) s -= l[i * m + j] * b[j * k + c];
      b[i * k + c] = s / l[i * m + i];
    }
  }
  // back substitution: L^T X = Y
  for (let i = m - 1; i >= 0; i--) {
    for (let c = 0; c < k; c++) {
      let s = b[i * k + c];
      for (let j = i + 1; j < m; j++) s -= l[j * m + i] * b[j * k + c];
      b[i * k + c] = s / l[i * m + i];
    }
  }
  return b;
}
