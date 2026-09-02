/**
 * The planner map: a field panel with the planner's state drawn over it.
 *
 * The field comes from `FieldRaster`; this adds the RRT* tree, the trajectory
 * it chose, the track flown so far, and the waypoints. Every overlay carries a
 * surface-coloured casing so it stays legible over any field colour.
 */
import { FieldRaster, fitProjection, type Projection } from "./fieldRaster.js";
import type { Extent, Layer } from "./layers.js";
import type { Theme } from "./theme.js";
import type { FieldAsset, SimSnapshot } from "../sim/types.js";

export interface MapOverlays {
  tree: boolean;
  plannedPath: boolean;
  trajectory: boolean;
}

export class MapView {
  private ctx: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private projection: Projection;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly asset: FieldAsset,
    private readonly raster: FieldRaster,
    private theme: Theme,
    private readonly threshold: number,
  ) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d canvas context unavailable");
    this.ctx = ctx;
    this.projection = fitProjection(raster.bbox, 1, 1, 60);
  }

  setTheme(theme: Theme): void {
    this.theme = theme;
  }

  resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.width = Math.max(1, Math.round(rect.width));
    this.height = Math.max(1, Math.round(rect.height));
    this.canvas.width = Math.round(this.width * dpr);
    this.canvas.height = Math.round(this.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.projection = fitProjection(this.raster.bbox, this.width, this.height, 60);
  }

  /** Canvas pixels to world (north, east). */
  worldAt(clientX: number, clientY: number): [number, number] {
    const rect = this.canvas.getBoundingClientRect();
    const { scale } = this.projection;
    const [originPxX] = this.projection.px(0, 0);
    const y = (clientX - rect.left - originPxX) / scale;
    const [, originPxY] = this.projection.px(0, 0);
    const x = (originPxY - (clientY - rect.top)) / scale;
    return [x, y];
  }

  draw(snapshot: SimSnapshot, layer: Layer, overlays: MapOverlays, extent: Extent): void {
    const { ctx, theme } = this;
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.fillStyle = theme.surface;
    ctx.fillRect(0, 0, this.width, this.height);

    const values = snapshot[layer.id] as Float64Array;
    this.raster.draw(ctx, this.projection, values, layer, extent, this.threshold);
    this.raster.drawOutline(ctx, this.projection, theme.axis, theme.outside, 1.5);

    if (overlays.tree) this.drawTree(snapshot);
    if (overlays.plannedPath) this.strokePath(snapshot.plannedPath, theme.series2, [7, 5]);
    if (overlays.trajectory) this.strokePath(snapshot.trajectory, theme.series1, []);
    this.drawWaypoints(snapshot);
  }

  private drawTree(snapshot: SimSnapshot): void {
    const { ctx } = this;
    const edges = snapshot.treeEdges;
    if (edges.length === 0) return;
    ctx.save();
    ctx.globalAlpha = 0.45;
    ctx.strokeStyle = this.theme.series3;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let i = 0; i < edges.length; i += 4) {
      const [ax, ay] = this.projection.px(edges[i], edges[i + 1]);
      const [bx, by] = this.projection.px(edges[i + 2], edges[i + 3]);
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
      const [px, py] = this.projection.px(points[i][0], points[i][1]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
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

  private dot(x: number, y: number, radius: number, fill: string): void {
    const { ctx } = this;
    const [px, py] = this.projection.px(x, y);
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

    // The target is a location, not another series, so it gets a shape in ink
    // rather than a fourth hue.
    const [tx, ty] = this.projection.px(snapshot.target[0], snapshot.target[1]);
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

    this.dot(snapshot.wpPioneer[0], snapshot.wpPioneer[1], 4, theme.series2);

    // The vehicle sits at the end of the track it has flown; the planner's
    // `wpNow`/`wpNext` run ahead of it and are not the AUV.
    const flown = snapshot.trajectory;
    const auv = flown.length > 0 ? flown[flown.length - 1] : snapshot.wpNow;
    this.dot(auv[0], auv[1], 6, theme.series1);
  }

  /** Grid index nearest a world point, for the hover readout. */
  nearestCell(x: number, y: number): number {
    const grid = this.asset.grid;
    let best = -1;
    let bestDist = Infinity;
    for (let i = 0; i < grid.length; i++) {
      const dx = grid[i][0] - x;
      const dy = grid[i][1] - y;
      const d = dx * dx + dy * dy;
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }
    return Math.sqrt(bestDist) <= this.asset.neighbourDistance * 1.5 ? best : -1;
  }
}
