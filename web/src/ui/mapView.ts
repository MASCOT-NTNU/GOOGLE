/**
 * The map: a field panel with the planner's state drawn on top.
 *
 * The field is painted as a brick-tiled cell grid (the simulation's staggered
 * hexagonal lattice, one rectangle per cell), clipped to the operational area
 * with the obstacle punched out. Overlays - the RRT* tree, the trajectory it
 * chose, the track flown so far, the waypoints - sit above it, each carrying a
 * surface-coloured ring so they stay legible over any field colour.
 *
 * Axes follow the Python plotting convention: east (y) runs across, north (x)
 * runs up.
 */
import { divergingRamp, salinityToUnit, sequentialRamp, type Ramp } from "./colormap.js";
import type { Theme } from "./theme.js";
import type { FieldAsset, SimSnapshot } from "../sim/types.js";

export type LayerId = "cost" | "mu" | "sd" | "eibv" | "ivr" | "truth";

export interface Layer {
  id: LayerId;
  label: string;
  short: string;
  description: string;
  diverging: boolean;
  unit: string;
}

export const LAYERS: Layer[] = [
  {
    id: "cost", label: "Cost valley", short: "Cost",
    description: "Weighted EIBV + IVR. The planner steers toward its minimum.",
    diverging: false, unit: "",
  },
  {
    id: "mu", label: "Conditional mean", short: "Mean",
    description: "The GRF's current best guess at salinity, given everything measured.",
    diverging: true, unit: "psu",
  },
  {
    id: "sd", label: "Standard deviation", short: "Std dev",
    description: "How uncertain the estimate still is, cell by cell.",
    diverging: false, unit: "psu",
  },
  {
    id: "eibv", label: "EIBV reward", short: "EIBV",
    description: "Expected integrated Bernoulli variance after sampling here. Low is informative.",
    diverging: false, unit: "",
  },
  {
    id: "ivr", label: "IVR reward", short: "IVR",
    description: "Integrated variance reduction from sampling here, flipped so low is informative.",
    diverging: false, unit: "",
  },
  {
    id: "truth", label: "Simulated truth", short: "Truth",
    description: "The realisation of the field the AUV is trying to map. Hidden from the agent.",
    diverging: true, unit: "psu",
  },
];

export interface MapOverlays {
  tree: boolean;
  plannedPath: boolean;
  trajectory: boolean;
}

interface Extent {
  min: number;
  max: number;
}

export class MapView {
  private ctx: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private scale = 1;
  private originY = 0; // east coordinate at the left edge
  private originX = 0; // north coordinate at the bottom edge
  private sequential: Ramp;
  private diverging: Ramp;

  /** Half-extent of one grid cell, in metres: east (across) and north (up). */
  private readonly cellEast: number;
  private readonly cellNorth: number;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly asset: FieldAsset,
    private theme: Theme,
    private readonly threshold: number,
  ) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d canvas context unavailable");
    this.ctx = ctx;
    this.sequential = sequentialRamp(theme.dark);
    this.diverging = divergingRamp(theme.dark);
    // The lattice steps `d * cos(60) * 2` east and `d * sin(60)` north.
    this.cellEast = asset.neighbourDistance * Math.cos(Math.PI / 3);
    this.cellNorth = (asset.neighbourDistance * Math.sin(Math.PI / 3)) / 2;
  }

  setTheme(theme: Theme): void {
    this.theme = theme;
    this.sequential = sequentialRamp(theme.dark);
    this.diverging = divergingRamp(theme.dark);
  }

  /** Recompute the canvas backing store and the world-to-pixel transform. */
  resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.width = Math.max(1, Math.round(rect.width));
    this.height = Math.max(1, Math.round(rect.height));
    this.canvas.width = Math.round(this.width * dpr);
    this.canvas.height = Math.round(this.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = Infinity;
    let ymax = -Infinity;
    for (const [x, y] of this.asset.polygonBorder) {
      xmin = Math.min(xmin, x);
      xmax = Math.max(xmax, x);
      ymin = Math.min(ymin, y);
      ymax = Math.max(ymax, y);
    }
    const pad = 80;
    const spanEast = ymax - ymin + 2 * pad;
    const spanNorth = xmax - xmin + 2 * pad;
    this.scale = Math.min(this.width / spanEast, this.height / spanNorth);
    this.originY = (ymin + ymax) / 2 - this.width / (2 * this.scale);
    this.originX = (xmin + xmax) / 2 - this.height / (2 * this.scale);
  }

  /** World (north, east) -> canvas pixels. East across, north up. */
  private px(x: number, y: number): [number, number] {
    return [(y - this.originY) * this.scale, this.height - (x - this.originX) * this.scale];
  }

  /** Canvas pixels -> world (north, east). */
  worldAt(clientX: number, clientY: number): [number, number] {
    const rect = this.canvas.getBoundingClientRect();
    const y = (clientX - rect.left) / this.scale + this.originY;
    const x = (rect.bottom - clientY) / this.scale + this.originX;
    return [x, y];
  }

  /**
   * Add a polygon as a closed sub-path. `fresh` starts a new path; pass false to
   * append a second ring to the current one, which is how the obstacle is
   * punched out of the border for an even-odd clip.
   */
  private tracePolygon(points: readonly (readonly [number, number])[], fresh = true): void {
    if (fresh) this.ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const [px, py] = this.px(points[i][0], points[i][1]);
      if (i === 0) this.ctx.moveTo(px, py);
      else this.ctx.lineTo(px, py);
    }
    this.ctx.closePath();
  }

  /** Value range of a field, used for the colour scale and the legend. */
  extentOf(values: Float64Array): Extent {
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < values.length; i++) {
      if (values[i] < min) min = values[i];
      if (values[i] > max) max = values[i];
    }
    if (!Number.isFinite(min)) return { min: 0, max: 1 };
    if (max - min < 1e-9) return { min, max: min + 1 };
    return { min, max };
  }

  colorFor(layer: Layer, value: number, extent: Extent): string {
    if (layer.diverging) {
      return this.diverging(salinityToUnit(value, this.threshold, extent.min, extent.max));
    }
    return this.sequential((value - extent.min) / (extent.max - extent.min));
  }

  draw(snapshot: SimSnapshot, layer: Layer, overlays: MapOverlays, extent: Extent): void {
    const { ctx, theme } = this;
    ctx.clearRect(0, 0, this.width, this.height);
    // The plate takes the panel's own surface colour so the map has no visible
    // frame; the operational area is set apart by its outline, not a backdrop.
    ctx.fillStyle = theme.surface;
    ctx.fillRect(0, 0, this.width, this.height);

    const values = snapshot[layer.id] as Float64Array;

    // Clip to the operational area with the obstacle punched out.
    ctx.save();
    this.tracePolygon(this.asset.polygonBorder);
    this.tracePolygon(this.asset.polygonObstacle, false);
    ctx.clip("evenodd");

    const w = 2 * this.cellEast * this.scale + 1;
    const h = 2 * this.cellNorth * this.scale + 1;
    const grid = this.asset.grid;
    for (let i = 0; i < grid.length; i++) {
      const [cx, cy] = this.px(grid[i][0], grid[i][1]);
      ctx.fillStyle = this.colorFor(layer, values[i], extent);
      ctx.fillRect(cx - w / 2, cy - h / 2, w, h);
    }
    ctx.restore();

    // Operational area outline and the obstacle.
    ctx.lineWidth = 1.5;
    ctx.strokeStyle = theme.axis;
    this.tracePolygon(this.asset.polygonBorder);
    ctx.stroke();
    ctx.fillStyle = theme.outside;
    this.tracePolygon(this.asset.polygonObstacle);
    ctx.fill();
    ctx.stroke();

    if (overlays.tree) this.drawTree(snapshot);
    if (overlays.plannedPath) this.drawPlannedPath(snapshot);
    if (overlays.trajectory) this.drawTrajectory(snapshot);
    this.drawWaypoints(snapshot);
  }

  private drawTree(snapshot: SimSnapshot): void {
    const { ctx } = this;
    const edges = snapshot.treeEdges;
    if (edges.length === 0) return;
    ctx.save();
    ctx.globalAlpha = 0.5;
    ctx.strokeStyle = this.theme.series3;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < edges.length; i += 4) {
      const [ax, ay] = this.px(edges[i], edges[i + 1]);
      const [bx, by] = this.px(edges[i + 2], edges[i + 3]);
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
    }
    ctx.stroke();
    ctx.restore();
  }

  private strokePath(points: readonly number[][], color: string, dash: number[]): void {
    if (points.length < 2) return;
    const { ctx } = this;
    ctx.save();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const [px, py] = this.px(points[i][0], points[i][1]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    // A surface-coloured casing keeps the line readable over any field colour.
    ctx.setLineDash([]);
    ctx.strokeStyle = this.theme.surface;
    ctx.lineWidth = 6;
    ctx.stroke();
    ctx.setLineDash(dash);
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
  }

  private drawPlannedPath(snapshot: SimSnapshot): void {
    this.strokePath(snapshot.plannedPath, this.theme.series2, [7, 5]);
  }

  private drawTrajectory(snapshot: SimSnapshot): void {
    this.strokePath(snapshot.trajectory, this.theme.series1, []);
  }

  private dot(x: number, y: number, radius: number, fill: string): void {
    const { ctx } = this;
    const [px, py] = this.px(x, y);
    ctx.beginPath();
    ctx.arc(px, py, radius + 2, 0, 2 * Math.PI);
    ctx.fillStyle = this.theme.surface;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(px, py, radius, 0, 2 * Math.PI);
    ctx.fillStyle = fill;
    ctx.fill();
  }

  private drawWaypoints(snapshot: SimSnapshot): void {
    const { ctx, theme } = this;

    // Target: a crosshair in ink rather than a fourth hue - it is a location,
    // not another series.
    const [tx, ty] = this.px(snapshot.target[0], snapshot.target[1]);
    ctx.save();
    ctx.strokeStyle = theme.surface;
    ctx.lineWidth = 4;
    for (const pass of [0, 1]) {
      if (pass === 1) {
        ctx.strokeStyle = theme.ink;
        ctx.lineWidth = 1.75;
      }
      ctx.beginPath();
      ctx.arc(tx, ty, 7, 0, 2 * Math.PI);
      ctx.moveTo(tx - 11, ty);
      ctx.lineTo(tx + 11, ty);
      ctx.moveTo(tx, ty - 11);
      ctx.lineTo(tx, ty + 11);
      ctx.stroke();
    }
    ctx.restore();

    // The pioneer waypoint terminates the leg RRT* just planned.
    this.dot(snapshot.wpPioneer[0], snapshot.wpPioneer[1], 4, theme.series2);

    // The vehicle sits at the end of the track it has actually flown; the
    // planner's `wpNow`/`wpNext` run ahead of it and are not the AUV.
    const flown = snapshot.trajectory;
    const auv = flown.length > 0 ? flown[flown.length - 1] : snapshot.wpNow;
    this.dot(auv[0], auv[1], 6, theme.series1);
  }
}
