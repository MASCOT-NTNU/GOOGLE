/**
 * Painting an interpolated field onto a canvas.
 *
 * The lattice is resampled into an ImageData at roughly a few pixels per grid
 * cell, then blown up to the canvas with the browser's own bilinear smoothing
 * and clipped to the operational area. That keeps the per-frame work
 * proportional to the lattice rather than to the canvas, and the exact
 * coastline comes from the clip path rather than from the raster, so the
 * lattice can spill past the border without it showing.
 *
 * Axes follow the Python plotting convention: east (y) across, north (x) up.
 */
import { divergingUnit } from "./colormap.js";
import { GridInterpolator, bboxOf, type Bbox } from "./interpolator.js";
import type { Extent, Layer } from "./layers.js";
import type { Point, Polygon } from "../sim/geometry.js";

export interface Projection {
  /** World (north, east) to canvas pixels. */
  px(x: number, y: number): [number, number];
  scale: number;
}

/** Fits the operational area into a canvas, preserving aspect ratio. */
export function fitProjection(bbox: Bbox, width: number, height: number, pad: number): Projection {
  const spanEast = bbox.ymax - bbox.ymin + 2 * pad;
  const spanNorth = bbox.xmax - bbox.xmin + 2 * pad;
  const scale = Math.min(width / spanEast, height / spanNorth);
  const originY = (bbox.ymin + bbox.ymax) / 2 - width / (2 * scale);
  const originX = (bbox.xmin + bbox.xmax) / 2 - height / (2 * scale);
  return {
    scale,
    px: (x, y) => [(y - originY) * scale, height - (x - originX) * scale],
  };
}

export class FieldRaster {
  readonly bbox: Bbox;
  private readonly interpolator: GridInterpolator;
  private readonly lattice: Float64Array;
  private readonly image: ImageData;
  private readonly scratch: HTMLCanvasElement;

  constructor(
    gridX: Float64Array,
    gridY: Float64Array,
    readonly polygonBorder: Polygon,
    readonly polygonObstacle: Polygon,
    spacing: number,
    /** Lattice nodes per grid spacing. 2 for thumbnails, 4 for the big panel. */
    density: number,
  ) {
    this.bbox = bboxOf(polygonBorder);
    const nx = Math.max(8, Math.round(((this.bbox.ymax - this.bbox.ymin) / spacing) * density));
    const ny = Math.max(8, Math.round(((this.bbox.xmax - this.bbox.xmin) / spacing) * density));
    this.interpolator = new GridInterpolator(gridX, gridY, this.bbox, nx, ny, spacing);
    this.lattice = new Float64Array(nx * ny);
    this.scratch = document.createElement("canvas");
    this.scratch.width = nx;
    this.scratch.height = ny;
    const ctx = this.scratch.getContext("2d");
    if (!ctx) throw new Error("2d canvas context unavailable");
    this.image = ctx.createImageData(nx, ny);
  }

  /** Trace a polygon as a sub-path; `fresh` starts a new path. */
  static tracePolygon(
    ctx: CanvasRenderingContext2D,
    projection: Projection,
    points: readonly Point[],
    fresh = true,
  ): void {
    if (fresh) ctx.beginPath();
    for (let i = 0; i < points.length; i++) {
      const [px, py] = projection.px(points[i][0], points[i][1]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.closePath();
  }

  /** Clip the context to the operational area, obstacle punched out. */
  clipToArea(ctx: CanvasRenderingContext2D, projection: Projection): void {
    FieldRaster.tracePolygon(ctx, projection, this.polygonBorder);
    FieldRaster.tracePolygon(ctx, projection, this.polygonObstacle, false);
    ctx.clip("evenodd");
  }

  /** Resample the layer and paint it across the whole canvas, clipped to the area. */
  draw(
    ctx: CanvasRenderingContext2D,
    projection: Projection,
    values: Float64Array,
    layer: Layer,
    extent: Extent,
    threshold: number,
  ): void {
    const { nx, ny } = this.interpolator;
    this.interpolator.apply(values, this.lattice);

    const data = this.image.data;
    const span = extent.max - extent.min;
    for (let i = 0; i < this.lattice.length; i++) {
      const v = this.lattice[i];
      if (Number.isNaN(v)) {
        data[i * 4 + 3] = 0;
        continue;
      }
      const t = layer.diverging
        ? divergingUnit(v, threshold, extent.min, extent.max)
        : (v - extent.min) / span;
      layer.ramp.writeRgba(t, data, i * 4);
    }
    const scratchCtx = this.scratch.getContext("2d");
    if (!scratchCtx) return;
    scratchCtx.putImageData(this.image, 0, 0);

    const [left, top] = projection.px(this.bbox.xmax, this.bbox.ymin);
    const [right, bottom] = projection.px(this.bbox.xmin, this.bbox.ymax);

    ctx.save();
    this.clipToArea(ctx, projection);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    // The lattice spans node centres, so stretch by half a node either way to
    // cover the full extent rather than leaving a seam at the edges.
    const dx = (right - left) / (nx - 1) / 2;
    const dy = (bottom - top) / (ny - 1) / 2;
    ctx.drawImage(this.scratch, left - dx, top - dy, right - left + 2 * dx, bottom - top + 2 * dy);
    ctx.restore();
  }

  /** Outline the operational area and fill the obstacle. */
  drawOutline(
    ctx: CanvasRenderingContext2D,
    projection: Projection,
    stroke: string,
    obstacleFill: string,
    lineWidth = 1.25,
  ): void {
    ctx.lineWidth = lineWidth;
    ctx.strokeStyle = stroke;
    FieldRaster.tracePolygon(ctx, projection, this.polygonBorder);
    ctx.stroke();
    ctx.fillStyle = obstacleFill;
    FieldRaster.tracePolygon(ctx, projection, this.polygonObstacle);
    ctx.fill();
    ctx.stroke();
  }

  /** The lattice the 3D surface reuses, so both views resample identically. */
  get grid(): GridInterpolator {
    return this.interpolator;
  }
}
