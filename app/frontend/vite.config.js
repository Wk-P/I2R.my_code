import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";

export default defineConfig({
  plugins: [vue()],
  base: "/classic/",   // 2026-10-13: the new panel (app/web) is served at /, this one at /classic/
  build: {
    outDir: "../backend/static",
    emptyOutDir: true,
  },
});
