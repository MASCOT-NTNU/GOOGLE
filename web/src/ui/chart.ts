/**
 * A single-series sparkline of one mission metric against step number.
 *
 * One metric per chart, never two scales on one pair of axes. Each is a single
 * series, so the tile's label carries the identity and no legend box is needed;
 * the current value is the headline beside it and every step is available in
 * the table view.
 */
import type { Theme } from "./theme.js";

export class Sparkline {
  private ctx: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private values: number[] = [];

  constructor(private readonly canvas: HTMLCanvasElement, private theme: Theme) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d canvas context unavailable");
    this.ctx = ctx;
  }

  setTheme(theme: Theme): void {
    this.theme = theme;
  }

  update(values: number[]): void {
    this.values = values;
    this.render();
  }

  resize(): void {
    const dpr = window.devicePixelRatio || 1;
    const rect = this.canvas.getBoundingClientRect();
    this.width = Math.max(1, Math.round(rect.width));
    this.height = Math.max(1, Math.round(rect.height));
    this.canvas.width = Math.round(this.width * dpr);
    this.canvas.height = Math.round(this.height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    this.render();
  }

  private render(): void {
    const { ctx, theme, width, height } = this;
    ctx.clearRect(0, 0, width, height);
    const values = this.values;
    if (values.length === 0) return;

    let min = Infinity;
    let max = -Infinity;
    for (const v of values) {
      if (v < min) min = v;
      if (v > max) max = v;
    }
    if (max - min < 1e-9) {
      min -= 0.5;
      max += 0.5;
    }
    const pad = 5;
    const plotH = height - 2 * pad;
    const xAt = (i: number) =>
      values.length === 1 ? width / 2 : (i / (values.length - 1)) * (width - 6) + 3;
    const yAt = (v: number) => pad + plotH - ((v - min) / (max - min)) * plotH;

    // A hairline baseline is enough chrome at this size; the tile carries the value.
    ctx.strokeStyle = theme.gridline;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, Math.round(height - 0.5));
    ctx.lineTo(width, Math.round(height - 0.5));
    ctx.stroke();

    ctx.strokeStyle = theme.series1;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    values.forEach((v, i) => (i === 0 ? ctx.moveTo(xAt(i), yAt(v)) : ctx.lineTo(xAt(i), yAt(v))));
    ctx.stroke();

    const lastX = xAt(values.length - 1);
    const lastY = yAt(values[values.length - 1]);
    ctx.beginPath();
    ctx.arc(lastX, lastY, 4.5, 0, 2 * Math.PI);
    ctx.fillStyle = theme.surface;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(lastX, lastY, 2.5, 0, 2 * Math.PI);
    ctx.fillStyle = theme.series1;
    ctx.fill();
  }
}
