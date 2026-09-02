/**
 * A single-series line chart of one mission metric against step number.
 *
 * One metric per chart, never two scales on one pair of axes. Each chart is one
 * series, so its title carries the identity and no legend box is needed; the
 * value at the current step is direct-labelled at the line's end and everything
 * else is available through the crosshair tooltip and the table view.
 */
import type { Theme } from "./theme.js";

export interface ChartSeries {
  label: string;
  caption: string;
  values: number[];
  format: (v: number) => string;
}

export class Chart {
  private ctx: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;
  private points: { x: number; y: number; step: number; value: number }[] = [];
  private hover: number | null = null;
  private series: ChartSeries | null = null;

  private readonly padLeft = 46;
  private readonly padRight = 12;
  private readonly padTop = 10;
  private readonly padBottom = 20;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly tooltip: HTMLElement,
    private theme: Theme,
  ) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d canvas context unavailable");
    this.ctx = ctx;

    canvas.addEventListener("pointermove", (event) => {
      const rect = canvas.getBoundingClientRect();
      this.hover = this.nearest(event.clientX - rect.left);
      this.render();
      this.showTooltip(event.clientX, event.clientY);
    });
    canvas.addEventListener("pointerleave", () => {
      this.hover = null;
      this.tooltip.hidden = true;
      this.render();
    });
  }

  setTheme(theme: Theme): void {
    this.theme = theme;
  }

  update(series: ChartSeries): void {
    this.series = series;
    this.resize();
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

  private nearest(px: number): number | null {
    if (this.points.length === 0) return null;
    let best = 0;
    let bestDist = Infinity;
    for (let i = 0; i < this.points.length; i++) {
      const d = Math.abs(this.points[i].x - px);
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }
    return best;
  }

  private showTooltip(clientX: number, clientY: number): void {
    if (this.hover === null || !this.series) {
      this.tooltip.hidden = true;
      return;
    }
    const point = this.points[this.hover];
    this.tooltip.hidden = false;
    this.tooltip.innerHTML =
      `<strong>Step ${point.step}</strong><span>${this.series.label}: ` +
      `${this.series.format(point.value)}</span>`;
    this.tooltip.style.left = `${clientX + 14}px`;
    this.tooltip.style.top = `${clientY + 14}px`;
  }

  private render(): void {
    const { ctx, theme, width, height } = this;
    ctx.clearRect(0, 0, width, height);
    const series = this.series;
    if (!series || series.values.length === 0) {
      ctx.fillStyle = theme.muted;
      ctx.font = "12px system-ui, -apple-system, 'Segoe UI', sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No steps yet", width / 2, height / 2);
      this.points = [];
      return;
    }

    const values = series.values;
    let min = Math.min(...values);
    let max = Math.max(...values);
    if (max - min < 1e-9) {
      min -= 0.5;
      max += 0.5;
    }
    const span = max - min;
    min -= span * 0.08;
    max += span * 0.08;

    const plotW = width - this.padLeft - this.padRight;
    const plotH = height - this.padTop - this.padBottom;
    const xAt = (i: number) =>
      this.padLeft + (values.length === 1 ? plotW / 2 : (i / (values.length - 1)) * plotW);
    const yAt = (v: number) => this.padTop + plotH - ((v - min) / (max - min)) * plotH;

    // Recessive gridlines and tick labels.
    ctx.font = "10px system-ui, -apple-system, 'Segoe UI', sans-serif";
    ctx.strokeStyle = theme.gridline;
    ctx.lineWidth = 1;
    ctx.fillStyle = theme.muted;
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    for (const frac of [0, 0.5, 1]) {
      const value = min + (max - min) * (1 - frac);
      const y = Math.round(this.padTop + plotH * frac) + 0.5;
      ctx.beginPath();
      ctx.moveTo(this.padLeft, y);
      ctx.lineTo(width - this.padRight, y);
      ctx.stroke();
      ctx.fillText(series.format(value), this.padLeft - 6, y);
    }

    ctx.textAlign = "left";
    ctx.textBaseline = "alphabetic";
    ctx.fillStyle = theme.muted;
    ctx.fillText("step 1", this.padLeft, height - 6);
    ctx.textAlign = "right";
    ctx.fillText(`${values.length}`, width - this.padRight, height - 6);

    this.points = values.map((v, i) => ({ x: xAt(i), y: yAt(v), step: i + 1, value: v }));

    ctx.strokeStyle = theme.series1;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    this.points.forEach((p, i) => (i === 0 ? ctx.moveTo(p.x, p.y) : ctx.lineTo(p.x, p.y)));
    ctx.stroke();

    // End marker with a surface ring, plus the one direct label.
    const last = this.points[this.points.length - 1];
    ctx.beginPath();
    ctx.arc(last.x, last.y, 6, 0, 2 * Math.PI);
    ctx.fillStyle = theme.surface;
    ctx.fill();
    ctx.beginPath();
    ctx.arc(last.x, last.y, 4, 0, 2 * Math.PI);
    ctx.fillStyle = theme.series1;
    ctx.fill();

    if (this.hover !== null) {
      const point = this.points[this.hover];
      ctx.strokeStyle = theme.axis;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(Math.round(point.x) + 0.5, this.padTop);
      ctx.lineTo(Math.round(point.x) + 0.5, this.padTop + plotH);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(point.x, point.y, 6, 0, 2 * Math.PI);
      ctx.fillStyle = theme.surface;
      ctx.fill();
      ctx.beginPath();
      ctx.arc(point.x, point.y, 4, 0, 2 * Math.PI);
      ctx.fillStyle = theme.series1;
      ctx.fill();
    }
  }
}
