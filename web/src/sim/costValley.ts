/**
 * The cost valley: one scalar field over the grid that the planner descends.
 *
 * Port of `src/common/CostValley/CostValley.py`. Budget mode is off in the
 * simulation study (`Config.__budget_mode = False`), so the valley is the
 * weighted sum of the EIBV (exploitation) and IVR (exploration) rewards.
 */
import type { Field } from "./field.js";
import type { GRF } from "./grf.js";

export class CostValley {
  readonly costField: Float64Array;

  constructor(
    private readonly field: Field,
    private readonly grf: GRF,
    public weightEibv: number,
    public weightIvr: number,
  ) {
    this.costField = new Float64Array(field.n);
    this.update();
  }

  /** Recompute the reward fields and blend them with the current weights. */
  update(): void {
    this.grf.getEiField();
    this.blend();
  }

  /** Re-blend from the cached reward fields, e.g. after a weight change. */
  blend(): void {
    const { eibvField, ivrField } = this.grf;
    for (let i = 0; i < this.costField.length; i++) {
      this.costField[i] = eibvField[i] * this.weightEibv + ivrField[i] * this.weightIvr;
    }
  }

  costAtLocation(x: number, y: number): number {
    return this.costField[this.field.indexFromLocation(x, y)];
  }

  /** Trapezoidal cost of travelling in a straight line between two points. */
  costAlongPath(ax: number, ay: number, bx: number, by: number): number {
    const dx = ax - bx;
    const dy = ay - by;
    const dist = Math.sqrt(dx * dx + dy * dy);
    return ((this.costAtLocation(ax, ay) + this.costAtLocation(bx, by)) / 2) * dist;
  }

  /** The global minimum of the valley, which becomes the RRT* target. */
  minimumCostLocation(): [number, number] {
    let best = 0;
    for (let i = 1; i < this.costField.length; i++) {
      if (this.costField[i] < this.costField[best]) best = i;
    }
    return [this.field.gridX[best], this.field.gridY[best]];
  }
}
