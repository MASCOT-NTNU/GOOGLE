/**
 * Colour-scale strip for the active field panel.
 *
 * Values the map encodes as colour need a readable scale; for the diverging
 * salinity panels the classification threshold is ticked explicitly, since that
 * boundary is what the mission is about.
 */
import type { Layer, MapView } from "./mapView.js";

export function renderColorbar(
  canvas: HTMLCanvasElement,
  map: MapView,
  layer: Layer,
  extent: { min: number; max: number },
  threshold: number,
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.round(rect.width));
  const height = Math.max(1, Math.round(rect.height));
  canvas.width = Math.round(width * dpr);
  canvas.height = Math.round(height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, width, height);

  for (let px = 0; px < width; px++) {
    const value = extent.min + ((extent.max - extent.min) * px) / (width - 1);
    ctx.fillStyle = map.colorFor(layer, value, extent);
    ctx.fillRect(px, 0, 1, height);
  }

  if (layer.diverging && threshold > extent.min && threshold < extent.max) {
    const px = ((threshold - extent.min) / (extent.max - extent.min)) * width;
    ctx.strokeStyle = getComputedStyle(document.documentElement)
      .getPropertyValue("--text-primary").trim();
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(px, 0);
    ctx.lineTo(px, height);
    ctx.stroke();
  }
}
