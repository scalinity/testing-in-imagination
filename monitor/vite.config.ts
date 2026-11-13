import { defineConfig, type Plugin } from "vite";
import { svelte } from "@sveltejs/vite-plugin-svelte";
import { readFile } from "node:fs/promises";
import { resolve } from "node:path";

// Development only: serves snapshots written by `pnpm snapshot`, so the page can be opened in a
// plain browser on localhost. The packaged app never uses this; it talks to Rust over IPC.
const snapshots = (): Plugin => ({
  name: "tii-snapshots",
  apply: "serve",
  configureServer(server) {
    server.middlewares.use("/snapshot", async (req, res) => {
      const mode = (req.url ?? "").includes("demo") ? "demo" : "real";
      try {
        const body = await readFile(resolve(import.meta.dirname, ".snapshot", mode, "state.json"));
        res.setHeader("content-type", "application/json");
        res.end(body);
      } catch {
        res.statusCode = 404;
        res.end("{}");
      }
    });
  },
});

export default defineConfig({
  plugins: [svelte(), snapshots()],
  clearScreen: false,
  server: { port: 1420, strictPort: true, host: "localhost" },
  build: { target: "safari16", sourcemap: false },
});
