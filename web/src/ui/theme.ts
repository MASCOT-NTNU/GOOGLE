/**
 * Theme tokens read from CSS custom properties, so the canvas layers paint with
 * the same palette the DOM uses and follow the light/dark switch.
 */
export interface Theme {
  dark: boolean;
  surface: string;
  plane: string;
  ink: string;
  inkSecondary: string;
  muted: string;
  gridline: string;
  axis: string;
  /** Categorical slot 1: the AUV and the track it has flown. */
  series1: string;
  /** Categorical slot 2: the trajectory RRT* chose for the next legs. */
  series2: string;
  /** Categorical slot 3: the RRT* tree. */
  series3: string;
  /** Water outside the operational area. */
  outside: string;
}

export function readTheme(root: HTMLElement = document.documentElement): Theme {
  const style = getComputedStyle(root);
  const get = (name: string) => style.getPropertyValue(name).trim();
  return {
    dark: get("--is-dark") === "1",
    surface: get("--surface-1"),
    plane: get("--plane"),
    ink: get("--text-primary"),
    inkSecondary: get("--text-secondary"),
    muted: get("--text-muted"),
    gridline: get("--gridline"),
    axis: get("--axis"),
    series1: get("--series-1"),
    series2: get("--series-2"),
    series3: get("--series-3"),
    outside: get("--outside"),
  };
}
