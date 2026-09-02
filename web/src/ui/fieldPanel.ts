/**
 * One small multiple: a field drawn on its own, with no planner overlay.
 *
 * All six panels share the same projection and the same `FieldRaster`, so they
 * are directly comparable and the interpolation weights are computed once for
 * the whole row of them.
 */
import { FieldRaster, fitProjection, type Projection } from "./fieldRaster.js";
import type { Extent, Layer } from "./layers.js";
import type { Theme } from "./theme.js";
import type { SimSnapshot } from "../sim/types.js";

export class FieldPanel {
  private ctx: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private projection: Projection;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly raster: FieldRaster,
    readonly layer: Layer,
    private theme: Theme,
    private readonly threshold: number,
  ) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d canvas context unavailable");
    this.ctx = ctx;
    this.projection = fitProjection(raster.bbox, 1, 1, 20);
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
    this.projection = fitProjection(this.raster.bbox, this.width, this.height, 20);
  }

  draw(snapshot: SimSnapshot, extent: Extent): void {
    const { ctx, theme } = this;
    ctx.clearRect(0, 0, this.width, this.height);
    ctx.fillStyle = theme.surface;
    ctx.fillRect(0, 0, this.width, this.height);
    const values = snapshot[this.layer.id] as Float64Array;
    this.raster.draw(ctx, this.projection, values, this.layer, extent, this.threshold);
    this.raster.drawOutline(ctx, this.projection, theme.axis, theme.outside, 1);
  }
}
