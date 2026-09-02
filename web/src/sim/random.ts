/**
 * Seeded pseudo-random numbers.
 *
 * The Python simulation seeds numpy so a replicate is reproducible; the browser
 * needs the same property, so every stochastic part of the port (the truth
 * field, the RRT* samples, the measurement noise) draws from this generator
 * rather than Math.random.
 */

/** Mulberry32: small, fast, good enough for a Monte-Carlo path planner. */
export class Random {
  private state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  /** Uniform on [0, 1). */
  uniform(): number {
    this.state = (this.state + 0x6d2b79f5) | 0;
    let t = Math.imul(this.state ^ (this.state >>> 15), 1 | this.state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }

  /** Uniform on [low, high). */
  range(low: number, high: number): number {
    return low + (high - low) * this.uniform();
  }

  /** Integer in [0, n). */
  int(n: number): number {
    return Math.floor(this.uniform() * n);
  }

  /** Standard normal, Box-Muller. */
  normal(): number {
    let u = 0;
    while (u === 0) u = this.uniform();
    const v = this.uniform();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  /** n standard normals. */
  normals(n: number): Float64Array {
    const out = new Float64Array(n);
    for (let i = 0; i < n; i++) out[i] = this.normal();
    return out;
  }
}
