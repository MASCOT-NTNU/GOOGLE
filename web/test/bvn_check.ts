/**
 * Numerics check for the bivariate normal CDF and its lookup table.
 *
 * The expected values are `scipy.stats.multivariate_normal.cdf` evaluated on the
 * same arguments, so a regression here means the EIBV field has drifted from the
 * Python reference.
 */
import { bvnCdf, BvnTable, normalCdf } from "../src/sim/normal.js";

const REFERENCE: [number, number, number, number][] = [
  // a, b, rho, scipy value
  [0, 0, -0.5, 1.666666666667e-1],
  [1, -1, -0.5, 9.614115922179e-2],
  [-2, 2, -0.9, 9.38887582116e-3],
  [2.5, -2.5, -0.99, 9.846047414356e-4],
  [0.3, -0.3, -0.01, 2.346416551822e-1],
  [-1.5, 1.5, -0.3, 5.547728694503e-2],
  [3, -3, -0.7, 1.119924358071e-3],
  [0, 0, -0.999, 7.11821870312e-3],
];

let failures = 0;
const check = (label: string, actual: number, expected: number, tolerance: number) => {
  const error = Math.abs(actual - expected);
  const ok = error <= tolerance;
  if (!ok) failures++;
  console.log(`${ok ? "ok  " : "FAIL"} ${label}: ${actual.toExponential(6)} ` +
    `(expected ${expected.toExponential(6)}, error ${error.toExponential(2)})`);
};

for (const [a, b, rho, expected] of REFERENCE) {
  check(`bvnCdf(${a}, ${b}, ${rho})`, bvnCdf(a, b, rho), expected, 1e-12);
}
check("normalCdf(0)", normalCdf(0), 0.5, 1e-12);
check("normalCdf(1.96)", normalCdf(1.96), 0.9750021048517796, 1e-12);
check("normalCdf(-3)", normalCdf(-3), 0.0013498980316300946, 1e-12);

const t0 = Date.now();
const table = new BvnTable();
console.log(`ok   table built in ${Date.now() - t0} ms`);

// The table is what the EIBV sweep actually reads, so hold it to a tolerance
// well below the differences the cost valley cares about.
let worst = 0;
for (let i = 0; i < 50000; i++) {
  const h = -6 + 12 * ((i * 0.61803398875) % 1);
  const rho = -((i * 0.31830988618) % 1);
  worst = Math.max(worst, Math.abs(table.lookup(h, rho) - bvnCdf(h, -h, rho)));
}
check("worst table interpolation error", worst, 0, 1e-4);

if (failures > 0) {
  console.error(`${failures} check(s) failed`);
  process.exit(1);
}
console.log("OK");
