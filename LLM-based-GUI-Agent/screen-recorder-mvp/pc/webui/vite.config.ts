import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  base: "/",
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
  server: {
    port: 5173,
    proxy: {
      "/api": "http://127.0.0.1:8776",
      "/app-assets": "http://127.0.0.1:8776",
      "/logo.svg": "http://127.0.0.1:8776",
      "/favicon.ico": "http://127.0.0.1:8776",
    },
  },
});
