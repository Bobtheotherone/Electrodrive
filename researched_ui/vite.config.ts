// FILE: researched_ui/vite.config.ts
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

/**
 * ResearchED Vite config.
 *
 * Design Doc citations:
 * - §3.1–§3.2: local web UI served by / coordinated with a Python backend over HTTP + WebSocket.
 * - FR-5: live run monitor implies real-time streams + plots (WebSocket support).
 * - FR-6: run controls require low-friction API calls to the backend during dev.
 *
 * Repo integration citations (Bobtheotherone/Electrodrive):
 * - The UI is kept in a standalone folder (researched_ui/) to remain optional/non-invasive (Non-goals §2.2).
 * - The Python backend is expected to expose REST endpoints under /api and WebSocket endpoints under /ws.
 */
const BACKEND_TARGET = "http://127.0.0.1:8000";

export default defineConfig(({ command }) => ({
  // Use absolute base in dev (more reliable), relative base in build (portable for Python static hosting).
  base: command === "build" ? "./" : "/",
  plugins: [react()],
  server: {
    proxy: {
      // Backend REST (Design Doc §3.1 HTTP).
      "/api": {
        target: BACKEND_TARGET,
        changeOrigin: true,
        secure: false
      },
      // Backend WebSocket (Design Doc §3.1 WS; FR-5 live streams, FR-6 controls/events).
      "/ws": {
        target: BACKEND_TARGET,
        ws: true,
        changeOrigin: true,
        secure: false
      }
    }
  },
  build: {
    // Self-contained output so the Python backend can serve it as a static directory (Design Doc §3.1).
    outDir: "dist",
    emptyOutDir: true
  }
}));
