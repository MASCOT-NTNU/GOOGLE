/**
 * The field panels: what each one shows, and how it is coloured.
 */
import { RAMPS, type Ramp } from "./colormap.js";
import type { SimSnapshot } from "../sim/types.js";

export type LayerId = "mu" | "truth" | "sd" | "eibv" | "ivr" | "cost";

export interface Layer {
  id: LayerId;
  label: string;
  caption: string;
  ramp: Ramp;
  /** Diverging layers put the classification threshold on the ramp's neutral. */
  diverging: boolean;
  unit: string;
  /**
   * Layers sharing a scale key are drawn against one common value range, so
   * their colours mean the same thing in both panels.
   */
  scaleKey: string;
}

export const LAYERS: Layer[] = [
  {
    id: "mu", label: "Conditional mean", caption: "Best estimate of salinity",
    ramp: RAMPS.BrBG, diverging: true, unit: "psu", scaleKey: "salinity",
  },
  {
    id: "truth", label: "Simulated truth", caption: "Hidden from the agent",
    ramp: RAMPS.BrBG, diverging: true, unit: "psu", scaleKey: "salinity",
  },
  {
    id: "sd", label: "Standard deviation", caption: "Remaining uncertainty",
    ramp: RAMPS.RdBuReversed, diverging: false, unit: "psu", scaleKey: "sd",
  },
  {
    id: "eibv", label: "EIBV reward", caption: "Ambiguity left after sampling here",
    ramp: RAMPS.YlGn, diverging: false, unit: "", scaleKey: "eibv",
  },
  {
    id: "ivr", label: "IVR reward", caption: "Variance removed, flipped so low is good",
    ramp: RAMPS.YlGnBu, diverging: false, unit: "", scaleKey: "ivr",
  },
  {
    id: "cost", label: "Cost valley", caption: "Weighted EIBV + IVR; the planner descends it",
    ramp: RAMPS.RdBuReversed, diverging: false, unit: "", scaleKey: "cost",
  },
];

export const LAYER_BY_ID = new Map(LAYERS.map((l) => [l.id, l]));

export interface Extent {
  min: number;
  max: number;
}

/**
 * Value ranges for every layer in a snapshot, with layers that share a
 * `scaleKey` collapsed onto one range.
 */
export function extentsOf(snapshot: SimSnapshot): Map<string, Extent> {
  const extents = new Map<string, Extent>();
  for (const layer of LAYERS) {
    const values = snapshot[layer.id] as Float64Array;
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < values.length; i++) {
      if (values[i] < min) min = values[i];
      if (values[i] > max) max = values[i];
    }
    const existing = extents.get(layer.scaleKey);
    if (existing) {
      existing.min = Math.min(existing.min, min);
      existing.max = Math.max(existing.max, max);
    } else {
      extents.set(layer.scaleKey, { min, max });
    }
  }
  for (const extent of extents.values()) {
    if (!Number.isFinite(extent.min)) {
      extent.min = 0;
      extent.max = 1;
    } else if (extent.max - extent.min < 1e-9) {
      extent.max = extent.min + 1;
    }
  }
  return extents;
}
