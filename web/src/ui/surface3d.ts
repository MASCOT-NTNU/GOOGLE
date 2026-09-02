/**
 * The cost valley as a 3D surface, the view `ValleyPlotter.plot_trees_on_3d_valley`
 * produces in the Python.
 *
 * Drawn straight onto a 2D canvas rather than through a 3D library: the surface
 * is a single-valued height field, so a painter's-algorithm pass over
 * depth-sorted quads is exact, needs no dependency, and keeps the same drawing
 * style as the rest of the page. Relief comes from Lambert shading on each
 * quad's normal, which is what makes the valley read as a valley rather than as
 * a flat heat map seen at an angle.
 *
 * Drag to orbit.
 */
import { GridInterpolator } from "./interpolator.js";
import { polygonContains, type Polygon } from "../sim/geometry.js";
import type { Ramp } from "./colormap.js";
import type { Theme } from "./theme.js";

interface Quad {
  depth: number;
  screen: [number, number][];
  value: number;
  light: number;
}

export interface SurfaceOverlay {
  /** World-space polyline, drawn draped on the surface. */
  points: number[][];
  color: string;
  dash: number[];
}

export class Surface3D {
  private ctx: CanvasRenderingContext2D;
  private width = 0;
  private height = 0;

  private readonly interpolator: GridInterpolator;
  private readonly lattice: Float64Array;
  private readonly quads: Quad[] = [];

  /** Orbit angles, radians. */
  private yaw = -0.62;
  private pitch = 0.92;
  private dragging = false;
  /**
   * Quads are stepped by this while orbiting. Filling and depth-sorting the
   * full lattice costs ~55 ms a frame, which drags badly; halving it in each
   * direction quarters the work, and the full lattice comes back on release.
   */
  private stride = 1;
  private lastX = 0;
  private lastY = 0;

  private values: Float64Array | null = null;
  private min = 0;
  private max = 1;
  private overlays: SurfaceOverlay[] = [];
  private marker: [number, number] | null = null;

  constructor(
    private readonly canvas: HTMLCanvasElement,
    gridX: Float64Array,
    gridY: Float64Array,
    bbox: { xmin: number; xmax: number; ymin: number; ymax: number },
    spacing: number,
    polygonBorder: Polygon,
    polygonObstacle: Polygon,
    private theme: Theme,
    private ramp: Ramp,
    /** Lattice side; 96 is ~9k quads, which sorts and fills comfortably and
     *  keeps the coastline silhouette from looking stepped. */
    resolution = 96,
  ) {
    const ctx = canvas.getContext("2d");
    if (!ctx) throw new Error("2d canvas context unavailable");
    this.ctx = ctx;
    this.interpolator = new GridInterpolator(
      gridX, gridY, bbox, resolution, resolution, spacing, spacing * 1.4,
      (x, y) => polygonContains(polygonBorder, x, y) && !polygonContains(polygonObstacle, x, y),
    );
    this.lattice = new Float64Array(resolution * resolution);

    canvas.addEventListener("pointerdown", (event) => {
      this.dragging = true;
      this.lastX = event.clientX;
      this.lastY = event.clientY;
      this.stride = 2;
      canvas.setPointerCapture(event.pointerId);
    });
    canvas.addEventListener("pointermove", (event) => {
      if (!this.dragging) return;
      this.yaw += (event.clientX - this.lastX) * 0.008;
      this.pitch = Math.min(1.45, Math.max(0.12, this.pitch - (event.clientY - this.lastY) * 0.006));
      this.lastX = event.clientX;
      this.lastY = event.clientY;
      this.render();
    });
    const stop = (event: PointerEvent) => {
      if (!this.dragging) return;
      this.dragging = false;
      this.stride = 1;
      if (canvas.hasPointerCapture(event.pointerId)) canvas.releasePointerCapture(event.pointerId);
      this.render();
    };
    canvas.addEventListener("pointerup", stop);
    canvas.addEventListener("pointercancel", stop);
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
    this.render();
  }

  update(
    values: Float64Array,
    extent: { min: number; max: number },
    overlays: SurfaceOverlay[],
    marker: [number, number] | null,
  ): void {
    this.values = values;
    this.min = extent.min;
    this.max = extent.max;
    this.overlays = overlays;
    this.marker = marker;
    this.interpolator.apply(values, this.lattice);
    this.render();
  }

  /** Normalised height in [0, 1] at a lattice node. */
  private unit(v: number): number {
    return (v - this.min) / Math.max(1e-9, this.max - this.min);
  }

  /** Height exaggeration, in units of the footprint's half-width. */
  private static readonly RELIEF = 0.42;

  /** Fit computed each frame so the surface fills the panel at any orbit angle. */
  private viewScale = 1;
  private viewX = 0;
  private viewY = 0;

  /**
   * Model space - u, v across the footprint, h up - to unscaled view space.
   * Orthographic, which keeps the surface readable and makes depth a plain
   * linear ordering.
   */
  private raw(u: number, v: number, h: number): [number, number, number] {
    const cy = Math.cos(this.yaw);
    const sy = Math.sin(this.yaw);
    const cp = Math.cos(this.pitch);
    const sp = Math.sin(this.pitch);
    const x = u - 0.5;
    const y = v - 0.5;
    const rx = x * cy - y * sy;
    const ry = x * sy + y * cy;
    // Depth grows toward the viewer, so an ascending sort paints back to front.
    return [rx, (ry * cp - h * Surface3D.RELIEF) * 0.62, ry * sp + h * 0.06];
  }

  private project(u: number, v: number, h: number): [number, number, number] {
    const [rx, ry, depth] = this.raw(u, v, h);
    return [this.viewX + rx * this.viewScale, this.viewY + ry * this.viewScale, depth];
  }

  /**
   * Rotating the surface changes how much of the panel its silhouette needs, so
   * the scale is refitted to the projected bounds rather than fixed - otherwise
   * the valley sits small in the middle of a mostly empty panel.
   */
  private fitView(): void {
    const { nx, ny } = this.interpolator;
    let xmin = Infinity;
    let xmax = -Infinity;
    let ymin = Infinity;
    let ymax = -Infinity;
    for (let row = 0; row < ny; row++) {
      for (let col = 0; col < nx; col++) {
        const value = this.lattice[row * nx + col];
        if (Number.isNaN(value)) continue;
        const [rx, ry] = this.raw(col / (nx - 1), row / (ny - 1), this.unit(value));
        if (rx < xmin) xmin = rx;
        if (rx > xmax) xmax = rx;
        if (ry < ymin) ymin = ry;
        if (ry > ymax) ymax = ry;
      }
    }
    if (!Number.isFinite(xmin)) {
      this.viewScale = Math.min(this.width, this.height);
      this.viewX = this.width / 2;
      this.viewY = this.height / 2;
      return;
    }
    const pad = 12;
    this.viewScale = Math.min(
      (this.width - 2 * pad) / Math.max(1e-6, xmax - xmin),
      (this.height - 2 * pad) / Math.max(1e-6, ymax - ymin),
    );
    this.viewX = this.width / 2 - ((xmin + xmax) / 2) * this.viewScale;
    this.viewY = this.height / 2 - ((ymin + ymax) / 2) * this.viewScale;
  }

  /** Project a world coordinate, taking its height from the resampled field. */
  private projectWorld(x: number, y: number, lift: number): [number, number, number] | null {
    const { worldX, worldY, nx, ny } = this.interpolator;
    const xmax = worldX[0];
    const xmin = worldX[(ny - 1) * nx];
    const ymin = worldY[0];
    const ymax = worldY[nx - 1];
    const v = (xmax - x) / (xmax - xmin);
    const u = (y - ymin) / (ymax - ymin);
    if (u < 0 || u > 1 || v < 0 || v > 1) return null;
    const col = Math.min(nx - 1, Math.max(0, Math.round(u * (nx - 1))));
    const row = Math.min(ny - 1, Math.max(0, Math.round(v * (ny - 1))));
    const value = this.lattice[row * nx + col];
    const h = Number.isNaN(value) ? 0 : this.unit(value);
    return this.project(u, v, h + lift);
  }

  private render(): void {
    const { ctx, theme, width, height } = this;
    ctx.clearRect(0, 0, width, height);
    ctx.fillStyle = theme.surface;
    ctx.fillRect(0, 0, width, height);
    if (!this.values) return;

    this.fitView();

    const { nx, ny } = this.interpolator;
    const lattice = this.lattice;
    const quads = this.quads;
    quads.length = 0;

    // Light from the upper left, rotating with the model so relief stays put.
    const lx = Math.cos(this.yaw + 0.9);
    const ly = Math.sin(this.yaw + 0.9);

    const step = this.stride;
    for (let row = 0; row + step < ny; row += step) {
      for (let col = 0; col + step < nx; col += step) {
        const a = lattice[row * nx + col];
        const b = lattice[row * nx + col + step];
        const c = lattice[(row + step) * nx + col + step];
        const d = lattice[(row + step) * nx + col];
        if (Number.isNaN(a) || Number.isNaN(b) || Number.isNaN(c) || Number.isNaN(d)) continue;

        const ha = this.unit(a);
        const hb = this.unit(b);
        const hc = this.unit(c);
        const hd = this.unit(d);
        const u0 = col / (nx - 1);
        const u1 = (col + step) / (nx - 1);
        const v0 = row / (ny - 1);
        const v1 = (row + step) / (ny - 1);

        const p0 = this.project(u0, v0, ha);
        const p1 = this.project(u1, v0, hb);
        const p2 = this.project(u1, v1, hc);
        const p3 = this.project(u0, v1, hd);

        // Central-difference slope gives the normal, scaled by the same relief
        // exaggeration the projection applies.
        const du = ((hb + hc) - (ha + hd)) / 2 / (u1 - u0);
        const dv = ((hd + hc) - (ha + hb)) / 2 / (v1 - v0);
        const nzx = -du * Surface3D.RELIEF;
        const nzy = -dv * Surface3D.RELIEF;
        const norm = Math.sqrt(nzx * nzx + nzy * nzy + 1);
        const lambert = (nzx * lx + nzy * ly + 1.6) / (norm * 2.1);
        const light = Math.min(1.18, Math.max(0.55, 0.62 + lambert * 0.62));

        quads.push({
          depth: (p0[2] + p1[2] + p2[2] + p3[2]) / 4,
          screen: [[p0[0], p0[1]], [p1[0], p1[1]], [p2[0], p2[1]], [p3[0], p3[1]]],
          value: (ha + hb + hc + hd) / 4,
          light,
        });
      }
    }

    quads.sort((p, q) => p.depth - q.depth);
    for (const quad of quads) {
      const fill = this.ramp.shaded(quad.value, quad.light);
      ctx.fillStyle = fill;
      // Stroking with the fill colour closes the hairline seams canvas
      // antialiasing leaves between adjacent fills.
      ctx.strokeStyle = fill;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(quad.screen[0][0], quad.screen[0][1]);
      for (let i = 1; i < 4; i++) ctx.lineTo(quad.screen[i][0], quad.screen[i][1]);
      ctx.closePath();
      ctx.fill();
      ctx.stroke();
    }

    for (const overlay of this.overlays) this.drawOverlay(overlay);

    if (this.marker) {
      const p = this.projectWorld(this.marker[0], this.marker[1], 0.035);
      if (p) {
        ctx.strokeStyle = theme.surface;
        ctx.lineWidth = 4;
        for (const pass of [0, 1]) {
          if (pass === 1) {
            ctx.strokeStyle = theme.ink;
            ctx.lineWidth = 1.75;
          }
          ctx.beginPath();
          ctx.arc(p[0], p[1], 6, 0, 2 * Math.PI);
          ctx.moveTo(p[0] - 10, p[1]);
          ctx.lineTo(p[0] + 10, p[1]);
          ctx.moveTo(p[0], p[1] - 10);
          ctx.lineTo(p[0], p[1] + 10);
          ctx.stroke();
        }
      }
    }
  }

  private drawOverlay(overlay: SurfaceOverlay): void {
    const { ctx } = this;
    const projected = overlay.points
      .map((p) => this.projectWorld(p[0], p[1], 0.012))
      .filter((p): p is [number, number, number] => p !== null);
    if (projected.length < 2) return;
    ctx.save();
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    ctx.moveTo(projected[0][0], projected[0][1]);
    for (let i = 1; i < projected.length; i++) ctx.lineTo(projected[i][0], projected[i][1]);
    ctx.setLineDash([]);
    ctx.strokeStyle = this.theme.surface;
    ctx.lineWidth = 5;
    ctx.stroke();
    ctx.setLineDash(overlay.dash);
    ctx.strokeStyle = overlay.color;
    ctx.lineWidth = 2;
    ctx.stroke();
    ctx.restore();
  }
}
