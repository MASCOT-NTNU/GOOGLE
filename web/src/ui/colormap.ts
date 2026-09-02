/**
 * Colour ramps for the field panels.
 *
 * These are the ColorBrewer schemes, chosen per field rather than by a house
 * palette, because the panels are read the way the paper's figures are:
 *
 * - **BrBG** on the conditional mean and on the truth, diverging about the
 *   classification threshold. Both salinity panels share the ramp *and* the
 *   value range so they can be compared cell for cell.
 * - **RdBu reversed** on the standard deviation and on the cost valley - blue
 *   low, red high in both. Running the same ramp two ways on one screen would
 *   invert the reader's expectation halfway across it, so red means "more" of
 *   whatever the panel measures: more uncertainty, more cost.
 * - **YlGn** on EIBV and **YlGnBu** on IVR - neighbouring sequential schemes, so
 *   the two reward fields read as a pair without being mistaken for each other.
 *
 * All four are colourblind-safe ColorBrewer schemes.
 */

type Rgb = [number, number, number];

function hexToRgb(hex: string): Rgb {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

/** ColorBrewer 9-class schemes, light end first. */
const SCHEMES = {
  BrBG: ["#8c510a", "#bf812d", "#dfc27d", "#f6e8c3", "#f5f5f5",
    "#c7eae5", "#80cdc1", "#35978f", "#01665e"],
  RdBu: ["#b2182b", "#d6604d", "#f4a582", "#fddbc7", "#f7f7f7",
    "#d1e5f0", "#92c5de", "#4393c3", "#2166ac"],
  YlGn: ["#ffffe5", "#f7fcb9", "#d9f0a3", "#addd8e", "#78c679",
    "#41ab5d", "#238443", "#006837", "#004529"],
  YlGnBu: ["#ffffd9", "#edf8b1", "#c7e9b4", "#7fcdbb", "#41b6c4",
    "#1d91c0", "#225ea8", "#253494", "#081d58"],
} as const;

export type SchemeName = keyof typeof SCHEMES;

function interpolate(stops: Rgb[], t: number): Rgb {
  const clamped = Math.min(1, Math.max(0, t));
  const scaled = clamped * (stops.length - 1);
  const i = Math.min(stops.length - 2, Math.floor(scaled));
  const f = scaled - i;
  const a = stops[i];
  const b = stops[i + 1];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

const LUT_SIZE = 256;

/**
 * A baked colour ramp. `rgb` feeds the raster renderers, which write bytes into
 * an ImageData; `css` feeds canvas fills and the DOM.
 */
export class Ramp {
  private readonly r = new Uint8ClampedArray(LUT_SIZE);
  private readonly g = new Uint8ClampedArray(LUT_SIZE);
  private readonly b = new Uint8ClampedArray(LUT_SIZE);
  private readonly strings: string[] = new Array(LUT_SIZE);

  constructor(scheme: SchemeName, reversed = false) {
    const stops = SCHEMES[scheme].map(hexToRgb);
    if (reversed) stops.reverse();
    for (let i = 0; i < LUT_SIZE; i++) {
      const [r, g, b] = interpolate(stops, i / (LUT_SIZE - 1));
      this.r[i] = r;
      this.g[i] = g;
      this.b[i] = b;
      this.strings[i] = `rgb(${r},${g},${b})`;
    }
  }

  private index(t: number): number {
    return t <= 0 ? 0 : t >= 1 ? LUT_SIZE - 1 : (t * (LUT_SIZE - 1)) | 0;
  }

  /** Write the colour for `t` in [0, 1] into an RGBA byte buffer at `offset`. */
  writeRgba(t: number, out: Uint8ClampedArray, offset: number, alpha = 255): void {
    const i = this.index(t);
    out[offset] = this.r[i];
    out[offset + 1] = this.g[i];
    out[offset + 2] = this.b[i];
    out[offset + 3] = alpha;
  }

  css(t: number): string {
    return this.strings[this.index(t)];
  }

  /** Colour for `t`, darkened by a shading factor - used for 3D relief. */
  shaded(t: number, light: number): string {
    const i = this.index(t);
    return `rgb(${(this.r[i] * light) | 0},${(this.g[i] * light) | 0},${(this.b[i] * light) | 0})`;
  }
}

export const RAMPS = {
  BrBG: new Ramp("BrBG"),
  RdBuReversed: new Ramp("RdBu", true),
  YlGn: new Ramp("YlGn"),
  YlGnBu: new Ramp("YlGnBu"),
} as const;

/**
 * Map a value onto [0, 1] with `centre` landing on 0.5, so a diverging ramp's
 * neutral sits exactly on the classification threshold rather than wherever the
 * data happens to be symmetric.
 */
export function divergingUnit(value: number, centre: number, min: number, max: number): number {
  if (value <= centre) {
    const span = Math.max(1e-9, centre - min);
    return 0.5 * Math.min(1, Math.max(0, (value - min) / span));
  }
  const span = Math.max(1e-9, max - centre);
  return 0.5 + 0.5 * Math.min(1, Math.max(0, (value - centre) / span));
}
