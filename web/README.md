# RRTCV in the browser

A live, interactive port of the simulation study behind
[*RRT\*-Enhanced Long-Horizon Path Planning for AUV Adaptive Sampling using a Cost
Valley*](https://doi.org/10.1016/j.knosys.2025.113261) (Ge, Eidsvik & Olaisen,
2025). Everything runs client-side, so the whole thing is a static site that
GitHub Pages can serve.

The mission loop is the same sense–plan–act cycle as the Python:

1. **Sense.** The AUV flies to its next waypoint and collects a CTD trace along
   the leg.
2. **Update.** The trace is assimilated into a Gaussian random field over the
   operational area, giving a new conditional mean and covariance.
3. **Value.** From that covariance, two reward fields are computed for every
   candidate cell — EIBV (how much classification ambiguity would remain if we
   sampled there) and IVR (how much variance sampling there would remove) — and
   blended into the **cost valley**.
4. **Plan.** RRT\* grows a tree from the vehicle's look-ahead waypoint toward the
   valley's global minimum, scoring branches by travel distance *plus* the cost
   integrated along them, and the first step of the best trajectory becomes the
   next waypoint.

## Running it locally

```sh
cd web
npm install
npm run dev            # http://localhost:5173
```

Other scripts:

| Command | What it does |
|---|---|
| `npm run build` | Type-check and produce `web/dist/` |
| `npm run preview` | Serve the production build |
| `npm test` | Headless smoke test of the simulation core in node |
| `npm run check` | Type-check only |

`npm run build` bakes the GitHub Pages base path `/RRTCV/` into the asset URLs.
Override it for another host with `BASE_PATH=/ npm run build`.

## How it is put together

```
web/
├── public/data/field.json     baked field: polygons, grid, SINMOD prior mean
├── tools/prepare_data.py      regenerates that file from the repo's CSV/SINMOD data
├── src/sim/                   the simulation, dependency-free TypeScript
│   ├── field.ts               grid, border/obstacle predicates      <- src/common/Field.py
│   ├── grf.ts                 GRF kernel, assimilation, EI fields   <- src/common/GRF/GRF.py
│   ├── costValley.ts          weighted EIBV + IVR field             <- src/common/CostValley/
│   ├── rrtStarCV.ts           RRT* over the cost valley             <- src/planners/RRTSCV/
│   ├── ctd.ts                 simulated truth and CTD sampling      <- src/simulation/CTD.py
│   ├── agent.ts               sense–plan–act loop                   <- Agents/AgentRRTStar.py
│   ├── normal.ts              normal / bivariate-normal CDFs        <- prior/cdf.npz
│   ├── linalg.ts              Cholesky and SPD solves
│   └── worker.ts              runs all of the above off the main thread
└── src/ui/                    canvas rendering, charts, controls
```

The heavy numerics live in a Web Worker: a 1076 × 1076 covariance matrix, its
Cholesky factorisation (~0.3 s, once per parameter set), and an O(N²) reward-field
sweep per step. A mission step costs roughly 100 ms on a laptop, so the page stays
responsive while it runs.

### Regenerating the field asset

`public/data/field.json` (~33 kB) holds the operational-area polygons, the 1076-cell
hexagonal grid, and the SINMOD prior mean sampled onto it. Rebuild it from the
repository's own data with:

```sh
pip install numpy scipy shapely
python3 web/tools/prepare_data.py
```

## What differs from the Python

The port follows the reference implementation closely; these are the deliberate
departures, all forced by what a browser can be given:

- **Static prior, not AR(1).** The Python reads a time-varying SINMOD netCDF and
  propagates the field with an AR(1) process between assimilations
  (`assimilate_temporal_data`). The netCDF is not in this repository, so the port
  uses the time-averaged SINMOD field in `Publication/prior/sinmod.csv` as a fixed
  prior mean and takes the static update path (`assimilate_data`), which is the
  `Config.temporal_truth = False` variant the code already supports.
- **EIBV by the `fast_eibv` formula.** `GRF.get_ei_field` has three EIBV paths. The
  port implements the default one — `Phi2(mur, -mur; -v / sigma_prior^2)` with
  `mur = (threshold - mu) / sigma_post` — which is what the published simulations
  ran with. The bivariate CDF comes from a table built at start-up out of an exact
  Genz `bvnd` (agreeing with `scipy.stats.multivariate_normal.cdf` to 12 digits)
  rather than from the shipped 11 MB `cdf.npz`; interpolation error is under 2e-5.
  Note that `__get_eibv_analytical`, the non-default path, scales `mur` by
  `sigma_post` *and* by `sqrt(sigma_prior^2)`, which is not dimensionally
  consistent with the other two; the port does not reproduce that.
- **Budget mode is off**, as it is in `Config` for the simulation study, so there is
  no remaining-budget ellipse and no go-home planner.
- **RRT\* samples are drawn live** from a seeded generator with rejection against the
  operational area, instead of being read from the precomputed
  `RRT_Random_Locations.npy`.
- **Rewiring checks path legality.** The Python rewires a neighbour through the new
  node on cost alone; the port also requires the new edge to be traversable, which
  keeps tree branches from crossing the shoreline.
- **Only the RRT\* agent.** The myopic baseline in `src/planners/Myopic2D` is not
  ported.

`npm test` runs a 12-step mission in node and prints IBV, RMSE and total variance
per step, which is the quickest way to check the numerics after a change.

## Deployment

`.github/workflows/deploy-web.yml` builds `web/` and publishes it to GitHub Pages.
It needs **Settings → Pages → Source: GitHub Actions** enabled once on the
repository. It runs on pushes to `main` that touch `web/`, and can be started by
hand from the Actions tab (useful for previewing a branch — GitHub may first need
that branch allowed under **Settings → Environments → github-pages**).
