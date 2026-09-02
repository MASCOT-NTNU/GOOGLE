/**
 * Headless smoke test: runs a short mission in node and prints the metrics, so
 * the numerics can be checked without a browser.
 *
 *   npm test
 */
import { readFileSync } from "node:fs";
import { Agent } from "../src/sim/agent.js";
import { CTD } from "../src/sim/ctd.js";
import { Field } from "../src/sim/field.js";
import { buildCovariance, DEFAULT_GRF_PARAMETERS } from "../src/sim/grf.js";
import { BvnTable } from "../src/sim/normal.js";
import type { FieldAsset, SimSettings } from "../src/sim/types.js";

const asset = JSON.parse(readFileSync("public/data/field.json", "utf8")) as FieldAsset;
const settings: SimSettings = {
  weightEibv: 1, weightIvr: 1, seed: 42,
  sigma: DEFAULT_GRF_PARAMETERS.sigma, nugget: DEFAULT_GRF_PARAMETERS.nugget,
  lateralRange: DEFAULT_GRF_PARAMETERS.lateralRange, rrtIterations: 1200,
};

const time = <T>(label: string, fn: () => T): T => {
  const t0 = Date.now();
  const out = fn();
  console.log(`${label}: ${Date.now() - t0} ms`);
  return out;
};

const bvn = time("bvn table", () => new BvnTable());
const field = time("field", () =>
  new Field(asset.grid, asset.polygonBorder, asset.polygonObstacle, asset.neighbourDistance));
console.log("grid cells:", field.n);

const params = { ...DEFAULT_GRF_PARAMETERS, sigma: settings.sigma, nugget: settings.nugget };
const sigmaPrior = time("covariance", () => buildCovariance(field, params.sigma, params.lateralRange));
const chol = time("cholesky", () => CTD.factorise(sigmaPrior, field.n));
const agent = time("agent", () =>
  new Agent(field, Float64Array.from(asset.muPrior), bvn, chol, params, settings,
    asset.locStart, asset.waypointDistance, sigmaPrior));

console.log("\nstep   ibv      rmse    variance  tree   sec");
for (let i = 0; i < 12; i++) {
  agent.advance();
  const m = agent.metrics[agent.metrics.length - 1];
  console.log(
    String(m.step).padStart(4),
    m.ibv.toFixed(2).padStart(8),
    m.rmse.toFixed(4).padStart(7),
    m.variance.toFixed(1).padStart(9),
    String(agent.rrt.nodes.length).padStart(6),
    m.elapsed.toFixed(2).padStart(6),
  );
}
const first = agent.metrics[0];
const last = agent.metrics[agent.metrics.length - 1];
console.log("\nvariance %.1f -> %.1f".replace("%.1f", first.variance.toFixed(1)).replace("%.1f", last.variance.toFixed(1)));
console.log("trajectory length:", agent.trajectory.length, "waypoints");
if (!(last.variance < first.variance)) throw new Error("variance did not decrease");
if (!Number.isFinite(last.rmse)) throw new Error("rmse is not finite");
console.log("OK");
