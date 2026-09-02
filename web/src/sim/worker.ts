/**
 * Simulation worker.
 *
 * Everything numerical - the 1076 x 1076 covariance matrix, the Cholesky
 * factorisation of the prior, the EI fields, RRT* - runs here so the main
 * thread only ever paints. The UI drives it with `step`/`reset` messages and
 * gets a snapshot of the state back after each one.
 */
import { Agent } from "./agent.js";
import { CTD } from "./ctd.js";
import { Field } from "./field.js";
import { buildCovariance, DEFAULT_GRF_PARAMETERS, type GrfParameters } from "./grf.js";
import { BvnTable } from "./normal.js";
import type { FieldAsset, SimSettings, SimSnapshot, WorkerRequest, WorkerResponse } from "./types.js";

let field: Field | null = null;
let asset: FieldAsset | null = null;
let muPrior: Float64Array | null = null;
let bvn: BvnTable | null = null;
let agent: Agent | null = null;

/** Prior covariance and its Cholesky factor, both cached per parameter set. */
let sigmaPrior: Float64Array | null = null;
let choleskyFactor: Float64Array | null = null;
let priorKey = "";

function post(message: WorkerResponse, transfer: Transferable[] = []): void {
  (self as unknown as Worker).postMessage(message, transfer);
}

function grfParametersFrom(settings: SimSettings): GrfParameters {
  return {
    ...DEFAULT_GRF_PARAMETERS,
    sigma: settings.sigma,
    nugget: settings.nugget,
    lateralRange: settings.lateralRange,
  };
}

function buildAgent(settings: SimSettings): Agent {
  if (!field || !muPrior || !bvn || !asset) throw new Error("worker not initialised");
  const params = grfParametersFrom(settings);
  const key = `${params.sigma}|${params.lateralRange}`;
  if (priorKey !== key || !sigmaPrior || !choleskyFactor) {
    post({ type: "progress", label: "Building the prior covariance" });
    sigmaPrior = buildCovariance(field, params.sigma, params.lateralRange);
    post({ type: "progress", label: "Factorising the prior covariance" });
    choleskyFactor = CTD.factorise(sigmaPrior, field.n);
    priorKey = key;
  }
  post({ type: "progress", label: "Sampling the truth field" });
  return new Agent(
    field, muPrior, bvn, choleskyFactor, params, settings,
    asset.locStart, asset.waypointDistance, sigmaPrior,
  );
}

function snapshot(a: Agent): SimSnapshot {
  const edges: number[] = [];
  for (const node of a.rrt.nodes) {
    if (node.parent < 0) continue;
    const parent = a.rrt.nodes[node.parent];
    edges.push(parent.x, parent.y, node.x, node.y);
  }
  return {
    step: a.step,
    mu: Float64Array.from(a.grf.mu),
    sd: Float64Array.from(a.standardDeviation()),
    eibv: Float64Array.from(a.grf.eibvField),
    ivr: Float64Array.from(a.grf.ivrField),
    cost: Float64Array.from(a.costValley.costField),
    truth: Float64Array.from(a.ctd.muTruth),
    trajectory: a.trajectory.map((p) => [...p]),
    plannedPath: a.rrt.trajectory.map((p) => [...p]),
    treeEdges: Float64Array.from(edges),
    wpNow: [...a.wpNow] as [number, number],
    wpNext: [...a.wpNext] as [number, number],
    wpPioneer: [...a.wpPioneer] as [number, number],
    target: [...a.target] as [number, number],
    metrics: a.metrics.map((m) => ({ ...m })),
  };
}

function transferablesOf(s: SimSnapshot): Transferable[] {
  return [s.mu.buffer, s.sd.buffer, s.eibv.buffer, s.ivr.buffer, s.cost.buffer,
    s.truth.buffer, s.treeEdges.buffer];
}

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const message = event.data;
  try {
    switch (message.type) {
      case "init": {
        asset = message.asset;
        muPrior = Float64Array.from(asset.muPrior);
        post({ type: "progress", label: "Building the bivariate normal table" });
        bvn = new BvnTable();
        post({ type: "progress", label: "Discretising the field" });
        field = new Field(
          asset.grid, asset.polygonBorder, asset.polygonObstacle, asset.neighbourDistance,
        );
        const t0 = performance.now();
        agent = buildAgent(message.settings);
        post({ type: "ready", n: field.n, truth: Float64Array.from(agent.ctd.muTruth), buildMs: performance.now() - t0 });
        const s = snapshot(agent);
        post({ type: "snapshot", snapshot: s }, transferablesOf(s));
        break;
      }
      case "reset": {
        agent = buildAgent(message.settings);
        const s = snapshot(agent);
        post({ type: "snapshot", snapshot: s }, transferablesOf(s));
        break;
      }
      case "step": {
        if (!agent) return;
        agent.advance();
        const s = snapshot(agent);
        post({ type: "snapshot", snapshot: s }, transferablesOf(s));
        break;
      }
      case "weights": {
        if (!agent) return;
        agent.costValley.weightEibv = message.weightEibv;
        agent.costValley.weightIvr = message.weightIvr;
        agent.costValley.blend();
        agent.target = agent.costValley.minimumCostLocation();
        const s = snapshot(agent);
        post({ type: "snapshot", snapshot: s }, transferablesOf(s));
        break;
      }
      case "iterations": {
        if (!agent) return;
        agent.rrt.params.maxIterations = message.value;
        break;
      }
    }
  } catch (error) {
    post({ type: "error", message: error instanceof Error ? error.message : String(error) });
  }
};
