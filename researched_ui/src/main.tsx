import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { RouterProvider } from "react-router-dom";
import { router } from "./router";

/**
 * ResearchED UI entry.
 *
 * Design Doc: §3.1 local web UI ↔ Python backend over HTTP + WebSocket.
 * Repo: Bobtheotherone/Electrodrive serves REST under /api/v1 and WS under /ws (electrodrive/researched/app.py).
 */

const el = document.getElementById("root");
if (!el) {
  throw new Error("ResearchED: missing #root element in index.html");
}

createRoot(el).render(
  <StrictMode>
    <RouterProvider router={router} />
  </StrictMode>,
);
