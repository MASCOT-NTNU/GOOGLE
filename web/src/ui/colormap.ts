/**
 * Colour ramps for the field panels.
 *
 * Two jobs, two kinds of ramp:
 *
 * - **Magnitude** (standard deviation, EIBV, IVR, cost valley) uses a single
 *   blue hue. On the light surface it runs light -> dark so near-zero recedes
 *   into the page; on the dark surface it runs dark -> light for the same
 *   reason. Never a rainbow: hue carries no information here.
 * - **Salinity** (the conditional mean and the truth) uses a diverging
 *   blue <-> red ramp with a neutral midpoint pinned to the classification
 *   threshold, because that threshold is exactly the question the survey is
 *   asking - which side of 26.81 is this water on.
 */

type Rgb = [number, number, number];

function hexToRgb(hex: string): Rgb {
  const v = parseInt(hex.slice(1), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

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

/** Blue ramp, steps 100-700 of the sequential scale. */
const BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7",
  "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"].map(hexToRgb);

/** Red arm of the diverging ramp, mirroring the blue arm's lightness steps. */
const RED = ["#fbdcdb", "#f7c4c3", "#f3a9a8", "#ef8d8c", "#e96e6d", "#e34948",
  "#d03b3b", "#b93131", "#a02828", "#871f1f", "#6d1818"].map(hexToRgb);

const NEUTRAL_LIGHT = hexToRgb("#f0efec");
const NEUTRAL_DARK = hexToRgb("#383835");

export type Ramp = (t: number) => string;

const LUT_SIZE = 256;

function bake(fn: (t: number) => Rgb): Ramp {
  const lut: string[] = new Array(LUT_SIZE);
  for (let i = 0; i < LUT_SIZE; i++) {
    const [r, g, b] = fn(i / (LUT_SIZE - 1));
    lut[i] = `rgb(${r},${g},${b})`;
  }
  return (t: number) => lut[Math.min(LUT_SIZE - 1, Math.max(0, Math.round(t * (LUT_SIZE - 1))))];
}

/** Sequential magnitude ramp; near-zero always recedes toward the surface. */
export function sequentialRamp(dark: boolean): Ramp {
  const stops = dark ? [...BLUE].reverse() : BLUE;
  return bake((t) => interpolate(stops, t));
}

/**
 * Diverging ramp with the neutral pinned at t = 0.5. Feed it the salinity
 * rescaled so that the threshold lands on 0.5.
 */
export function divergingRamp(dark: boolean): Ramp {
  const neutral = dark ? NEUTRAL_DARK : NEUTRAL_LIGHT;
  const below = [...BLUE].reverse(); // dark blue (coldest/freshest) -> light
  const above = RED;
  return bake((t) => {
    if (t <= 0.5) return interpolate([...below, neutral], t * 2);
    return interpolate([neutral, ...above], (t - 0.5) * 2);
  });
}

/**
 * Map a salinity value onto [0, 1] with `threshold` at 0.5, so the diverging
 * ramp's neutral sits exactly on the classification boundary.
 */
export function salinityToUnit(value: number, threshold: number, min: number, max: number): number {
  if (value <= threshold) {
    const span = Math.max(1e-9, threshold - min);
    return 0.5 * Math.min(1, Math.max(0, (value - min) / span));
  }
  const span = Math.max(1e-9, max - threshold);
  return 0.5 + 0.5 * Math.min(1, Math.max(0, (value - threshold) / span));
}
