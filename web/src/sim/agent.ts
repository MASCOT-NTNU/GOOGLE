/**
 * The agent: sense, plan, act.
 *
 * Port of `Agents/AgentRRTStar.py` together with `Planner/Planner.py`. One step is
 *
 *   1. log the metrics of the current belief,
 *   2. shift the waypoint trackers forward (now <- next <- pioneer),
 *   3. fly to the waypoint and collect a CTD trace,
 *   4. assimilate the trace, rebuild the cost valley, and plan the new pioneer
 *      waypoint with RRT* from the next waypoint to the valley's minimum.
 *
 * The planner is always one waypoint ahead of the vehicle, which is what lets
 * the AUV keep moving while the next leg is being computed.
 */
import { CostValley } from "./costValley.js";
import { CTD } from "./ctd.js";
import { Field } from "./field.js";
import { GRF, type GrfParameters } from "./grf.js";
import { BvnTable, normalCdf } from "./normal.js";
import { Random } from "./random.js";
import { RRTStarCV, defaultRrtParameters } from "./rrtStarCV.js";
import type { SimSettings, StepMetrics } from "./types.js";

export class Agent {
  readonly grf: GRF;
  readonly costValley: CostValley;
  readonly rrt: RRTStarCV;
  readonly ctd: CTD;

  readonly metrics: StepMetrics[] = [];
  readonly trajectory: number[][] = [];

  step = 0;
  wpNow: [number, number];
  wpNext: [number, number];
  wpPioneer: [number, number];
  target: [number, number];

  private readonly sd: Float64Array;

  constructor(
    field: Field,
    muPrior: Float64Array,
    bvn: BvnTable,
    choleskyFactor: Float64Array,
    grfParams: GrfParameters,
    settings: SimSettings,
    locStart: [number, number],
    waypointDistance: number,
    sigmaPrior: Float64Array,
  ) {
    const random = new Random(settings.seed);
    this.grf = new GRF(field, muPrior, grfParams, bvn, sigmaPrior);
    this.costValley = new CostValley(field, this.grf, settings.weightEibv, settings.weightIvr);
    const rrtParams = defaultRrtParameters(waypointDistance);
    rrtParams.maxIterations = settings.rrtIterations;
    this.rrt = new RRTStarCV(field, this.costValley, rrtParams, random);
    this.ctd = new CTD(field, muPrior, choleskyFactor, random, locStart[0], locStart[1]);
    this.sd = new Float64Array(field.n);

    // Planner initialisation: aim one and two steps toward the valley minimum.
    this.wpNow = [...locStart] as [number, number];
    this.target = this.costValley.minimumCostLocation();
    const angle = Math.atan2(this.target[0] - this.wpNow[0], this.target[1] - this.wpNow[1]);
    const s = waypointDistance;
    this.wpNext = [this.wpNow[0] + s * Math.sin(angle), this.wpNow[1] + s * Math.cos(angle)];
    this.wpPioneer = [this.wpNext[0] + s * Math.sin(angle), this.wpNext[1] + s * Math.cos(angle)];
  }

  /** Advance the mission by one waypoint. */
  advance(): void {
    const t0 = performance.now();

    // s0: log the belief before it is updated, matching `Agent.update_metrics`.
    this.trajectory.push([...this.wpNow]);

    // s1: shift the trackers.
    const flying: [number, number] = [...this.wpNow] as [number, number];
    this.wpNow = this.wpNext;
    this.wpNext = this.wpPioneer;

    // s2: fly there and sense.
    const ctdData = this.ctd.collect(flying[0], flying[1]);

    // s3: assimilate, rebuild the valley, replan.
    this.grf.assimilate(ctdData);
    this.costValley.update();
    this.target = this.costValley.minimumCostLocation();
    this.wpPioneer = this.rrt.getNextWaypoint(
      this.wpNext[0], this.wpNext[1], this.target[0], this.target[1],
    );

    this.step += 1;
    this.metrics.push(this.computeMetrics(performance.now() - t0));
  }

  private computeMetrics(elapsedMs: number): StepMetrics {
    const { grf, ctd } = this;
    const n = grf.n;
    grf.standardDeviation(this.sd);
    let ibv = 0;
    let se = 0;
    let variance = 0;
    for (let i = 0; i < n; i++) {
      const p = normalCdf((grf.threshold - grf.mu[i]) / this.sd[i]);
      ibv += p * (1 - p);
      const d = ctd.muTruth[i] - grf.mu[i];
      se += d * d;
      variance += this.sd[i] * this.sd[i];
    }
    return {
      step: this.step,
      ibv,
      rmse: Math.sqrt(se / n),
      variance,
      elapsed: elapsedMs / 1000,
    };
  }

  /** Marginal standard deviation of the current belief. */
  standardDeviation(): Float64Array {
    return this.grf.standardDeviation(this.sd);
  }
}
