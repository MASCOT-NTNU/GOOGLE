/**
 * Colour-scale strip for a field panel.
 *
 * Values encoded as colour need a readable scale; on the diverging salinity
 * panels the classification threshold is ticked explicitly, because which side
 * of it a cell falls on is the question the mission exists to answer.
 */
import { divergingUnit } from "./colormap.js";
import type { Extent, Layer } from "./layers.js";

export function renderColorbar(
  canvas: HTMLCanvasElement,
  layer: Layer,
  extent: Extent,
  threshold: number,
  ink: string,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  if (width < 2) return;
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  const span = extent.max - extent.min;
  for (let px = 0; px < width; px++) {
    const value = extent.min + (span * px) / (width - 1);
    const t = layer.diverging
      ? divergingUnit(value, threshold, extent.min, extent.max)
      : (value - extent.min) / span;
    ctx.fillStyle = layer.ramp.css(t);
    ctx.fillRect(px, 0, 1, height);
  }

  if (layer.diverging && threshold > extent.min && threshold < extent.max) {
    const px = ((threshold - extent.min) / span) * width;
    ctx.strokeStyle = ink;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, height);
    ctx.stroke();
  }
}
