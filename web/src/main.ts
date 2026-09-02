/**
 * Page wiring: load the baked field, start the simulation worker, and keep the
 * canvases, metrics and controls in sync with the snapshots it sends back.
 *
 * No numerics live here - the main thread only paints and forwards intent.
 */
import { Chart } from "./ui/chart.js";
import { renderColorbar } from "./ui/legend.js";
import { LAYERS, MapView, type Layer, type MapOverlays } from "./ui/mapView.js";
import { readTheme, type Theme } from "./ui/theme.js";
import { DEFAULT_GRF_PARAMETERS } from "./sim/grf.js";
import type { FieldAsset, SimSettings, SimSnapshot, WorkerRequest, WorkerResponse } from "./sim/types.js";

const el = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
};

const THRESHOLD = DEFAULT_GRF_PARAMETERS.threshold;

const settings: SimSettings = {
  weightEibv: 1,
  weightIvr: 1,
  seed: 42,
  sigma: 1,
  nugget: 0.25,
  lateralRange: 700,
  rrtIterations: 1200,
};

let theme: Theme = readTheme();
let snapshot: SimSnapshot | null = null;
let activeLayer: Layer = LAYERS[0];
let running = false;
let busy = false;
let stepIntervalMs = 400;
let pendingStep: number | null = null;

const overlays: MapOverlays = { tree: true, plannedPath: true, trajectory: true };

const boot = el("boot");
const bootLabel = el("boot-label");
const status = el("status");
const tooltip = el("tooltip");
const readout = el("readout");

/* ---------------------------------------------------------------- worker */

const worker = new Worker(new URL("./sim/worker.ts", import.meta.url), { type: "module" });
const send = (message: WorkerRequest) => worker.postMessage(message);

/* ------------------------------------------------------------------ view */

let map: MapView;
let chartIbv: Chart;
let chartRmse: Chart;
let chartVar: Chart;

const fixed = (digits: number) => (v: number) =>
  Math.abs(v) >= 1000 ? v.toFixed(0) : v.toFixed(digits);

function buildLayerTabs(): void {
  const tabs = el("layer-tabs");
  tabs.innerHTML = "";
  for (const layer of LAYERS) {
    const button = document.createElement("button");
    button.type = "button";
    button.role = "tab";
    button.textContent = layer.label;
    button.setAttribute("aria-selected", String(layer.id === activeLayer.id));
    button.addEventListener("click", () => {
      activeLayer = layer;
      for (const other of tabs.querySelectorAll("button")) {
        other.setAttribute("aria-selected", String(other === button));
      }
      el("layer-caption").textContent = layer.description;
      render();
    });
    tabs.appendChild(button);
  }
  el("layer-caption").textContent = activeLayer.description;
}

function render(): void {
  if (!snapshot || !map) return;
  const values = snapshot[activeLayer.id] as Float64Array;
  const extent = map.extentOf(values);
  map.draw(snapshot, activeLayer, overlays, extent);
  renderColorbar(el("colorbar"), map, activeLayer, extent, THRESHOLD);

  const unit = activeLayer.unit ? ` ${activeLayer.unit}` : "";
  el("cb-min").textContent = `${extent.min.toFixed(2)}${unit}`;
  el("cb-max").textContent = `${extent.max.toFixed(2)}${unit}`;
  el("cb-mid").textContent = activeLayer.diverging
    ? `threshold ${THRESHOLD.toFixed(2)}`
    : `${((extent.min + extent.max) / 2).toFixed(2)}${unit}`;
}

function renderMetrics(): void {
  if (!snapshot) return;
  const metrics = snapshot.metrics;
  chartIbv.update({
    label: "IBV", caption: "Integrated Bernoulli variance",
    values: metrics.map((m) => m.ibv), format: fixed(1),
  });
  chartRmse.update({
    label: "RMSE", caption: "RMSE vs truth",
    values: metrics.map((m) => m.rmse), format: fixed(3),
  });
  chartVar.update({
    label: "Total variance", caption: "Total variance",
    values: metrics.map((m) => m.variance), format: fixed(0),
  });

  const last = metrics[metrics.length - 1];
  el("val-ibv").textContent = last ? last.ibv.toFixed(1) : "—";
  el("val-rmse").textContent = last ? last.rmse.toFixed(3) : "—";
  el("val-var").textContent = last ? last.variance.toFixed(0) : "—";

  const body = el("table-body");
  body.innerHTML = metrics
    .map((m) =>
      `<tr><td>${m.step}</td><td>${m.ibv.toFixed(1)}</td>` +
      `<td>${m.rmse.toFixed(3)}</td><td>${m.variance.toFixed(0)}</td></tr>`)
    .join("");
}

function setStatus(text: string): void {
  status.textContent = text;
}

/* ------------------------------------------------------------ transport */

function scheduleNext(): void {
  if (!running) return;
  pendingStep = window.setTimeout(() => {
    pendingStep = null;
    if (running && !busy) {
      busy = true;
      send({ type: "step" });
    }
  }, stepIntervalMs);
}

function setRunning(next: boolean): void {
  running = next;
  const button = el<HTMLButtonElement>("btn-run");
  button.textContent = next ? "Pause" : "Run";
  if (!next && pendingStep !== null) {
    clearTimeout(pendingStep);
    pendingStep = null;
  }
  if (next && !busy) {
    busy = true;
    send({ type: "step" });
  }
}

function restart(): void {
  setRunning(false);
  busy = true;
  boot.hidden = false;
  bootLabel.textContent = "Rebuilding the field…";
  send({ type: "reset", settings: { ...settings } });
}

/* ------------------------------------------------------------- controls */

function bindSlider(
  inputId: string, outputId: string,
  format: (v: number) => string,
  onChange: (v: number) => void,
  commit?: (v: number) => void,
): void {
  const input = el<HTMLInputElement>(inputId);
  const output = el(outputId);
  const sync = () => {
    output.textContent = format(Number(input.value));
  };
  input.addEventListener("input", () => {
    sync();
    onChange(Number(input.value));
  });
  if (commit) input.addEventListener("change", () => commit(Number(input.value)));
  sync();
}

function applyWeights(): void {
  el<HTMLInputElement>("in-eibv").value = settings.weightEibv.toFixed(2);
  el<HTMLInputElement>("in-ivr").value = settings.weightIvr.toFixed(2);
  el("out-eibv").textContent = settings.weightEibv.toFixed(2);
  el("out-ivr").textContent = settings.weightIvr.toFixed(2);
  syncPresetButtons();
  send({ type: "weights", weightEibv: settings.weightEibv, weightIvr: settings.weightIvr });
}

function syncPresetButtons(): void {
  const { weightEibv: e, weightIvr: i } = settings;
  const active = e === 2 && i === 0 ? "eibv" : e === 0 && i === 2 ? "ivr" : e === i ? "equal" : "";
  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-preset]")) {
    button.classList.toggle("is-active", button.dataset.preset === active);
  }
}

function bindControls(): void {
  el("btn-run").addEventListener("click", () => setRunning(!running));
  el("btn-step").addEventListener("click", () => {
    if (busy) return;
    setRunning(false);
    busy = true;
    send({ type: "step" });
  });
  el("btn-reset").addEventListener("click", restart);

  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-preset]")) {
    button.addEventListener("click", () => {
      const preset = button.dataset.preset;
      // The three weight sets the paper compares.
      settings.weightEibv = preset === "eibv" ? 2 : preset === "ivr" ? 0 : 1;
      settings.weightIvr = preset === "ivr" ? 2 : preset === "eibv" ? 0 : 1;
      applyWeights();
    });
  }

  bindSlider("in-eibv", "out-eibv", (v) => v.toFixed(2), (v) => {
    settings.weightEibv = v;
    applyWeights();
  });
  bindSlider("in-ivr", "out-ivr", (v) => v.toFixed(2), (v) => {
    settings.weightIvr = v;
    applyWeights();
  });
  bindSlider("in-iter", "out-iter", (v) => String(v), (v) => {
    settings.rrtIterations = v;
    send({ type: "iterations", value: v });
  });
  bindSlider("in-speed", "out-speed", (v) => `${v} ms`, (v) => {
    stepIntervalMs = v;
  });
  bindSlider("in-sigma", "out-sigma", (v) => v.toFixed(2), (v) => {
    settings.sigma = v;
  }, restart);
  bindSlider("in-nugget", "out-nugget", (v) => v.toFixed(2), (v) => {
    settings.nugget = v;
  }, restart);
  bindSlider("in-range", "out-range", (v) => `${v} m`, (v) => {
    settings.lateralRange = v;
  }, restart);

  const seed = el<HTMLInputElement>("in-seed");
  seed.addEventListener("change", () => {
    settings.seed = Number(seed.value) || 0;
    restart();
  });
  el("btn-shuffle").addEventListener("click", () => {
    settings.seed = Math.floor(Math.random() * 100000);
    seed.value = String(settings.seed);
    restart();
  });

  const bindOverlay = (id: string, key: keyof MapOverlays) => {
    const input = el<HTMLInputElement>(id);
    input.addEventListener("change", () => {
      overlays[key] = input.checked;
      render();
    });
  };
  bindOverlay("ov-tree", "tree");
  bindOverlay("ov-path", "plannedPath");
  bindOverlay("ov-traj", "trajectory");

  const tableButton = el<HTMLButtonElement>("btn-table");
  tableButton.addEventListener("click", () => {
    const view = el("table-view");
    view.hidden = !view.hidden;
    tableButton.setAttribute("aria-pressed", String(!view.hidden));
  });

  const themeButton = el<HTMLButtonElement>("theme-toggle");
  themeButton.addEventListener("click", () => {
    const next = theme.dark ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    refreshTheme();
  });
}

function refreshTheme(): void {
  theme = readTheme();
  document.querySelector("[data-theme-label]")!.textContent = theme.dark ? "Light" : "Dark";
  map?.setTheme(theme);
  chartIbv?.setTheme(theme);
  chartRmse?.setTheme(theme);
  chartVar?.setTheme(theme);
  render();
  renderMetrics();
}

/* ------------------------------------------------------------- map hover */

function bindMapHover(asset: FieldAsset): void {
  const canvas = el<HTMLCanvasElement>("map");
  canvas.addEventListener("pointermove", (event) => {
    if (!snapshot) return;
    const [x, y] = map.worldAt(event.clientX, event.clientY);
    let best = -1;
    let bestDist = Infinity;
    for (let i = 0; i < asset.grid.length; i++) {
      const dx = asset.grid[i][0] - x;
      const dy = asset.grid[i][1] - y;
      const d = dx * dx + dy * dy;
      if (d < bestDist) {
        bestDist = d;
        best = i;
      }
    }
    if (best < 0 || Math.sqrt(bestDist) > asset.neighbourDistance) {
      readout.hidden = true;
      return;
    }
    readout.hidden = false;
    readout.innerHTML =
      `<b>Cell ${best}</b><dl>` +
      `<dt>Mean</dt><dd>${snapshot.mu[best].toFixed(2)} psu</dd>` +
      `<dt>Truth</dt><dd>${snapshot.truth[best].toFixed(2)} psu</dd>` +
      `<dt>Std dev</dt><dd>${snapshot.sd[best].toFixed(3)}</dd>` +
      `<dt>EIBV</dt><dd>${snapshot.eibv[best].toFixed(3)}</dd>` +
      `<dt>IVR</dt><dd>${snapshot.ivr[best].toFixed(3)}</dd>` +
      `<dt>Cost</dt><dd>${snapshot.cost[best].toFixed(3)}</dd>` +
      `</dl>`;
  });
  canvas.addEventListener("pointerleave", () => {
    readout.hidden = true;
  });
}

/* ------------------------------------------------------------------ boot */

async function start(): Promise<void> {
  const response = await fetch(`${import.meta.env.BASE_URL}data/field.json`);
  if (!response.ok) throw new Error(`could not load the field asset (${response.status})`);
  const asset = (await response.json()) as FieldAsset;

  map = new MapView(el<HTMLCanvasElement>("map"), asset, theme, THRESHOLD);
  chartIbv = new Chart(el<HTMLCanvasElement>("chart-ibv"), tooltip, theme);
  chartRmse = new Chart(el<HTMLCanvasElement>("chart-rmse"), tooltip, theme);
  chartVar = new Chart(el<HTMLCanvasElement>("chart-var"), tooltip, theme);

  buildLayerTabs();
  bindControls();
  bindMapHover(asset);
  refreshTheme();

  const resize = () => {
    map.resize();
    chartIbv.resize();
    chartRmse.resize();
    chartVar.resize();
    render();
  };
  window.addEventListener("resize", resize);
  new ResizeObserver(resize).observe(el("map").parentElement!);

  worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
    const message = event.data;
    switch (message.type) {
      case "progress":
        bootLabel.textContent = `${message.label}…`;
        break;
      case "ready":
        setStatus(`${message.n} grid cells · field built in ${(message.buildMs / 1000).toFixed(1)} s`);
        break;
      case "snapshot": {
        snapshot = message.snapshot;
        busy = false;
        boot.hidden = true;
        resize();
        renderMetrics();
        const last = snapshot.metrics[snapshot.metrics.length - 1];
        setStatus(
          snapshot.step === 0
            ? "Ready. Press Run to start the mission."
            : `Step ${snapshot.step} · ${last ? `${(last.elapsed * 1000).toFixed(0)} ms/step` : ""}`,
        );
        scheduleNext();
        break;
      }
      case "error":
        setStatus(`Error: ${message.message}`);
        boot.hidden = true;
        busy = false;
        setRunning(false);
        break;
    }
  };

  send({ type: "init", asset, settings: { ...settings } });
}

start().catch((error: unknown) => {
  bootLabel.textContent = error instanceof Error ? error.message : String(error);
});
