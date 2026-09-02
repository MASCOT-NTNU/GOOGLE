import { defineConfig } from "vite";

// The site is served from https://mascot-ntnu.github.io/RRTCV/ on GitHub Pages,
// so assets need that prefix in a production build. `BASE_PATH` lets the
// workflow (or a local `npm run build`) override it.
const base = process.env.BASE_PATH ?? "/RRTCV/";

export default defineConfig({
  base,
  build: {
    target: "es2020",
    outDir: "dist",
    assetsInlineLimit: 0,
  },
  worker: {
    format: "es",
  },
});
