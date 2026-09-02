/** Shared shapes between the simulation worker and the UI. */

/** Provenance of the prior mean field baked into an asset. */
export interface PriorMetadata {
  /** Human-readable name, shown in the page. */
  label: string;
  /** Where the values came from, as passed to `prepare_data.py --source`. */
  source: string;
  units: string;
  /** True while the asset ships stand-in data rather than a real survey. */
  placeholder: boolean;
  points: number;
  range: [number, number];
}

/**
 * The baked field asset, `web/public/data/field.json`.
 *
 * Regenerate it - or generate an alternative one to point the page at with
 * `?data=` - with `web/tools/prepare_data.py`.
 */
export interface FieldAsset {
  /** Bumped when the shape of this file changes; see `SCHEMA_VERSION`. */
  schemaVersion?: number;
  generatedBy: string;
  generatedAt?: string;
  prior?: PriorMetadata;
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
