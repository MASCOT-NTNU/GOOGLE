/**
 * Page wiring: load the baked field, start the simulation worker, and keep the
 * 3D valley, the planner map, the six field panels and the metric tiles in sync
 * with the snapshots it sends back.
 *
 * No numerics live here - the main thread only paints and forwards intent.
 */
import { RAMPS } from "./ui/colormap.js";
import { FieldPanel } from "./ui/fieldPanel.js";
import { FieldRaster } from "./ui/fieldRaster.js";
import { extentsOf, LAYERS, LAYER_BY_ID, type Extent, type Layer, type LayerId } from "./ui/layers.js";
import { renderColorbar } from "./ui/legend.js";
import { MapView, type MapOverlays } from "./ui/mapView.js";
import { Sparkline } from "./ui/chart.js";
import { Surface3D } from "./ui/surface3d.js";
import { readTheme, type Theme } from "./ui/theme.js";
import { DEFAULT_GRF_PARAMETERS } from "./sim/grf.js";
import type { FieldAsset, SimSettings, SimSnapshot, WorkerRequest, WorkerResponse } from "./sim/types.js";

const el = <T extends HTMLElement>(id: string): T => {
  const node = document.getElementById(id);
  if (!node) throw new Error(`missing element #${id}`);
  return node as T;
};

/** The schema `FieldAsset` is written against; `prepare_data.py` stamps it. */
const SCHEMA_VERSION = 1;
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
let extents = new Map<string, Extent>();
let mapLayer: Layer = LAYER_BY_ID.get("cost")!;
let running = false;
let busy = false;
let stepIntervalMs = 400;
let pendingStep: number | null = null;

const overlays: MapOverlays = { tree: true, plannedPath: true, trajectory: true };

const boot = el("boot");
const bootLabel = el("boot-label");
const status = el("status");
const readout = el("readout");

const worker = new Worker(new URL("./sim/worker.ts", import.meta.url), { type: "module" });
const send = (message: WorkerRequest) => worker.postMessage(message);

let map: MapView;
let valley: Surface3D;
const panels: FieldPanel[] = [];
const panelButtons = new Map<LayerId, HTMLElement>();
const panelScales = new Map<LayerId, HTMLElement>();
let sparkIbv: Sparkline;
let sparkRmse: Sparkline;
let sparkVar: Sparkline;

/* ------------------------------------------------------------------ view */

function extentFor(layer: Layer): Extent {
  return extents.get(layer.scaleKey) ?? { min: 0, max: 1 };
}

function formatValue(v: number): string {
  const abs = Math.abs(v);
  if (abs >= 100) return v.toFixed(0);
  if (abs >= 1) return v.toFixed(2);
  return v.toFixed(3);
}

function buildFieldPanels(raster: FieldRaster): void {
  const container = el("fields");
  container.innerHTML = "";
  for (const layer of LAYERS) {
    const card = document.createElement("button");
    card.type = "button";
    card.className = "field-panel";
    card.setAttribute("aria-pressed", String(layer.id === mapLayer.id));
    card.title = layer.caption;

    const heading = document.createElement("h3");
    heading.textContent = layer.label;
    const canvas = document.createElement("canvas");
    const scale = document.createElement("div");
    scale.className = "field-panel__scale";
    scale.innerHTML = "<span></span><span></span>";

    card.append(heading, canvas, scale);
    container.appendChild(card);

    // Clicking a panel puts that field under the planner overlays, which is the
    // only way to see the tree and the track against anything but the valley.
    card.addEventListener("click", () => {
      mapLayer = layer;
      for (const [id, node] of panelButtons) node.setAttribute("aria-pressed", String(id === layer.id));
      el("map-title").textContent = `Planner · ${layer.label.toLowerCase()}`;
      el("map-title").title = layer.caption;
      renderMap();
    });

    panels.push(new FieldPanel(canvas, raster, layer, theme, THRESHOLD));
    panelButtons.set(layer.id, card);
    panelScales.set(layer.id, scale);
  }
}

function renderMap(): void {
  if (!snapshot) return;
  const extent = extentFor(mapLayer);
  map.draw(snapshot, mapLayer, overlays, extent);
  renderColorbar(el("colorbar"), mapLayer, extent, THRESHOLD, theme.ink);
  const unit = mapLayer.unit ? ` ${mapLayer.unit}` : "";
  el("cb-min").textContent = `${formatValue(extent.min)}${unit}`;
  el("cb-max").textContent = `${formatValue(extent.max)}${unit}`;
  el("cb-mid").textContent = mapLayer.diverging
    ? `threshold ${THRESHOLD.toFixed(2)}`
    : formatValue((extent.min + extent.max) / 2) + unit;
}

function renderAll(): void {
  if (!snapshot) return;
  extents = extentsOf(snapshot);
  renderMap();
  for (const panel of panels) {
    const extent = extentFor(panel.layer);
    panel.draw(snapshot, extent);
    const scale = panelScales.get(panel.layer.id);
    if (scale) {
      scale.children[0].textContent = formatValue(extent.min);
      scale.children[1].textContent = formatValue(extent.max);
    }
  }
  const costLayer = LAYER_BY_ID.get("cost")!;
  valley.update(
    snapshot.cost,
    extentFor(costLayer),
    [
      ...(overlays.plannedPath ? [{ points: snapshot.plannedPath, color: theme.series2, dash: [7, 5] }] : []),
      ...(overlays.trajectory ? [{ points: snapshot.trajectory, color: theme.series1, dash: [] }] : []),
    ],
    snapshot.target,
  );
  renderMetrics();
}

function renderMetrics(): void {
  if (!snapshot) return;
  const metrics = snapshot.metrics;
  sparkIbv.update(metrics.map((m) => m.ibv));
  sparkRmse.update(metrics.map((m) => m.rmse));
  sparkVar.update(metrics.map((m) => m.variance));

  const last = metrics[metrics.length - 1];
  el("val-ibv").textContent = last ? last.ibv.toFixed(1) : "—";
  el("val-rmse").textContent = last ? last.rmse.toFixed(3) : "—";
  el("val-var").textContent = last ? last.variance.toFixed(0) : "—";

  el("table-body").innerHTML = metrics
    .map((m) =>
      `<tr><td>${m.step}</td><td>${m.ibv.toFixed(1)}</td>` +
      `<td>${m.rmse.toFixed(3)}</td><td>${m.variance.toFixed(0)}</td></tr>`)
    .join("");
}

function describePrior(asset: FieldAsset): void {
  const node = el("provenance");
  const prior = asset.prior;
  if (!prior) {
    node.textContent = "";
    return;
  }
  const tag = prior.placeholder ? '<span class="tag">placeholder</span>' : "";
  node.innerHTML = `<b>Prior</b>${tag} ${prior.label} · ` +
    `${asset.grid.length.toLocaleString()} cells at ${asset.neighbourDistance} m`;
  node.title = `Source: ${prior.source}`;
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
  el<HTMLButtonElement>("btn-run").textContent = next ? "Pause" : "Run";
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
  const sync = () => { output.textContent = format(Number(input.value)); };
  input.addEventListener("input", () => {
    sync();
    onChange(Number(input.value));
  });
  if (commit) input.addEventListener("change", () => commit(Number(input.value)));
  sync();
}

function applyWeights(): void {
  el<HTMLInputElement>("in-eibv").value = String(settings.weightEibv);
  el<HTMLInputElement>("in-ivr").value = String(settings.weightIvr);
  el("out-eibv").textContent = settings.weightEibv.toFixed(2);
  el("out-ivr").textContent = settings.weightIvr.toFixed(2);
  const { weightEibv: e, weightIvr: i } = settings;
  const active = e === 2 && i === 0 ? "eibv" : e === 0 && i === 2 ? "ivr" : e === i ? "equal" : "";
  for (const button of document.querySelectorAll<HTMLButtonElement>("[data-preset]")) {
    button.classList.toggle("is-active", button.dataset.preset === active);
  }
  send({ type: "weights", weightEibv: settings.weightEibv, weightIvr: settings.weightIvr });
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
      // The three weight sets the paper compares.
      const preset = button.dataset.preset;
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
  bindSlider("in-iter", "out-iter", String, (v) => {
    settings.rrtIterations = v;
    send({ type: "iterations", value: v });
  });
  bindSlider("in-speed", "out-speed", (v) => `${v} ms`, (v) => { stepIntervalMs = v; });
  bindSlider("in-sigma", "out-sigma", (v) => v.toFixed(2), (v) => { settings.sigma = v; }, restart);
  bindSlider("in-nugget", "out-nugget", (v) => v.toFixed(2), (v) => { settings.nugget = v; }, restart);
  bindSlider("in-range", "out-range", (v) => `${v} m`, (v) => { settings.lateralRange = v; }, restart);

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
      renderAll();
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

  el<HTMLButtonElement>("theme-toggle").addEventListener("click", () => {
    document.documentElement.dataset.theme = theme.dark ? "light" : "dark";
    refreshTheme();
  });
}

function refreshTheme(): void {
  theme = readTheme();
  document.querySelector("[data-theme-label]")!.textContent = theme.dark ? "Light" : "Dark";
  map?.setTheme(theme);
  valley?.setTheme(theme);
  for (const panel of panels) panel.setTheme(theme);
  sparkIbv?.setTheme(theme);
  sparkRmse?.setTheme(theme);
  sparkVar?.setTheme(theme);
  renderAll();
}

/* ------------------------------------------------------------------ boot */

function assetUrl(): string {
  const requested = new URLSearchParams(location.search).get("data");
  const base = `${import.meta.env.BASE_URL}data/field.json`;
  if (!requested) return base;
  const resolved = new URL(requested, `${location.origin}${import.meta.env.BASE_URL}`);
  if (resolved.origin !== location.origin) {
    console.warn(`ignoring cross-origin ?data=${requested}`);
    return base;
  }
  return resolved.pathname + resolved.search;
}

async function start(): Promise<void> {
  const url = assetUrl();
  const response = await fetch(url);
  if (!response.ok) throw new Error(`could not load ${url} (${response.status})`);
  const asset = (await response.json()) as FieldAsset;
  if (asset.schemaVersion !== undefined && asset.schemaVersion !== SCHEMA_VERSION) {
    throw new Error(
      `${url} is schema v${asset.schemaVersion}, this page expects v${SCHEMA_VERSION}. ` +
      "Re-run web/tools/prepare_data.py.",
    );
  }

  const gridX = new Float64Array(asset.grid.length);
  const gridY = new Float64Array(asset.grid.length);
  for (let i = 0; i < asset.grid.length; i++) {
    gridX[i] = asset.grid[i][0];
    gridY[i] = asset.grid[i][1];
  }

  // Two rasters: a dense one for the big map, a light one shared by the six
  // thumbnails, where the extra lattice nodes would not survive the downscale.
  const mapRaster = new FieldRaster(
    gridX, gridY, asset.polygonBorder, asset.polygonObstacle, asset.neighbourDistance, 4,
  );
  const panelRaster = new FieldRaster(
    gridX, gridY, asset.polygonBorder, asset.polygonObstacle, asset.neighbourDistance, 2.5,
  );

  map = new MapView(el<HTMLCanvasElement>("map"), asset, mapRaster, theme, THRESHOLD);
  valley = new Surface3D(
    el<HTMLCanvasElement>("valley3d"), gridX, gridY, mapRaster.bbox,
    asset.neighbourDistance, asset.polygonBorder, asset.polygonObstacle,
    theme, RAMPS.RdBuReversed,
  );
  sparkIbv = new Sparkline(el<HTMLCanvasElement>("spark-ibv"), theme);
  sparkRmse = new Sparkline(el<HTMLCanvasElement>("spark-rmse"), theme);
  sparkVar = new Sparkline(el<HTMLCanvasElement>("spark-var"), theme);

  buildFieldPanels(panelRaster);
  describePrior(asset);
  bindControls();
  bindMapHover();
  refreshTheme();

  const resize = () => {
    map.resize();
    valley.resize();
    for (const panel of panels) panel.resize();
    sparkIbv.resize();
    sparkRmse.resize();
    sparkVar.resize();
    renderAll();
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
        break;
      case "snapshot": {
        snapshot = message.snapshot;
        busy = false;
        boot.hidden = true;
        resize();
        const last = snapshot.metrics[snapshot.metrics.length - 1];
        status.textContent = snapshot.step === 0
          ? "Ready — press Run"
          : `Step ${snapshot.step} · ${last ? `${(last.elapsed * 1000).toFixed(0)} ms` : ""}`;
        scheduleNext();
        break;
      }
      case "error":
        status.textContent = `Error: ${message.message}`;
        boot.hidden = true;
        busy = false;
        setRunning(false);
        break;
    }
  };

  send({ type: "init", asset, settings: { ...settings } });
}

function bindMapHover(): void {
  const canvas = el<HTMLCanvasElement>("map");
  canvas.addEventListener("pointermove", (event) => {
    if (!snapshot) return;
    const [x, y] = map.worldAt(event.clientX, event.clientY);
    const cell = map.nearestCell(x, y);
    if (cell < 0) {
      readout.hidden = true;
      return;
    }
    readout.hidden = false;
    readout.innerHTML = "<dl>" + LAYERS.map((layer) => {
      const value = (snapshot![layer.id] as Float64Array)[cell];
      return `<dt>${layer.label}</dt><dd>${formatValue(value)}</dd>`;
    }).join("") + "</dl>";
  });
  canvas.addEventListener("pointerleave", () => { readout.hidden = true; });
}

start().catch((error: unknown) => {
  bootLabel.textContent = error instanceof Error ? error.message : String(error);
});
