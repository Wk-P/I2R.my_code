import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

// New panel (2026-10-13): served by app/backend at /, API /api/v2 (app/backend/v2/). The old panel (app/frontend)
// is built with base /classic/ and stays reachable there.
export default defineConfig({
  plugins: [vue()],
  base: "/",
  build: { outDir: "../backend/static_v2", emptyOutDir: true, chunkSizeWarningLimit: 1500 },
  server: { proxy: { "/api": "http://localhost:8081" } },
});
