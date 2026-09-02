/** Shared shapes between the simulation worker and the UI. */

/** The baked field asset, `web/public/data/field.json`. */
export interface FieldAsset {
  generatedBy: string;
  wgsOrigin: [number, number];
  neighbourDistance: number;
  waypointDistance: number;
  locStart: [number, number];
  locEnd: [number, number];
  polygonBorder: [number, number][];
  polygonObstacle: [number, number][];
  grid: [number, number][];
  muPrior: number[];
}

export interface SimSettings {
  weightEibv: number;
  weightIvr: number;
  seed: number;
  sigma: number;
  nugget: number;
  lateralRange: number;
  rrtIterations: number;
}

export interface StepMetrics {
  step: number;
  /** Integrated Bernoulli variance over the field: how misclassified it still is. */
  ibv: number;
  /** RMSE of the conditional mean against the truth. */
  rmse: number;
  /** Total remaining variance. */
  variance: number;
  /** Seconds spent in the last step. */
  elapsed: number;
}

/** One frame of simulation state, transferred to the UI after each step. */
export interface SimSnapshot {
  step: number;
  mu: Float64Array;
  sd: Float64Array;
  eibv: Float64Array;
  ivr: Float64Array;
  cost: Float64Array;
  truth: Float64Array;
  trajectory: number[][];
  plannedPath: number[][];
  treeEdges: Float64Array; // flat [x1, y1, x2, y2, ...]
  wpNow: [number, number];
  wpNext: [number, number];
  wpPioneer: [number, number];
  target: [number, number];
  metrics: StepMetrics[];
}

export type WorkerRequest =
  | { type: "init"; asset: FieldAsset; settings: SimSettings }
  | { type: "step" }
  | { type: "reset"; settings: SimSettings }
  | { type: "weights"; weightEibv: number; weightIvr: number }
  | { type: "iterations"; value: number };

export type WorkerResponse =
  | { type: "ready"; n: number; truth: Float64Array; buildMs: number }
  | { type: "progress"; label: string }
  | { type: "snapshot"; snapshot: SimSnapshot }
  | { type: "error"; message: string };
