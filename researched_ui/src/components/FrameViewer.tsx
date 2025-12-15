import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// Source notes:
// - Design Doc: FR-5 (live visualization frame viewer), FR-3 (run-dir artifacts: viz/*.png).
// - Repo: electrodrive/viz/iter_viz.py (frames: prefers viz/viz_*.png, fallback viz/viz.png; overlay outputs exist; example glob logic) [~L118-130].
// - Repo: electrodrive/utils/logging.py (run_dir contract commonly uses viz/ folder; logs are written under run_dir) [context: events.jsonl path at ~L136].

export type FrameLike =
  | string
  | {
      path?: string;
      url?: string;
      index?: number;
      created_at?: string;
      mtime?: number;
      [key: string]: unknown;
    };

export interface NormalizedFrame {
  index: number;
  path?: string;
  url?: string;
  label: string;
}

export interface FrameViewerProps {
  runId?: string;

  /** If provided, component is presentational and will not poll. Can be URLs or relative paths. */
  frames?: ReadonlyArray<FrameLike>;

  /** Optional initial "running" state; used to decide whether to keep polling/cache-busting. */
  isRunning?: boolean;

  /** API base for polling frames when runId is provided (defaults to VITE_API_BASE or same-origin). */
  apiBase?: string;

  /** Poll interval (ms) when fetching frames list from backend. */
  pollIntervalMs?: number;

  /** Height in px of the image area. */
  height?: number;

  /** Allow arbitrary extra props to avoid breaking unknown call-sites. */
  [key: string]: unknown;
}

function getEnvApiBase(): string | undefined {
  try {
    const maybe = (import.meta as unknown as { env?: Record<string, string | undefined> }).env;
    const v = maybe?.VITE_API_BASE;
    if (typeof v === "string" && v.trim()) return v.trim();
  } catch {
    // ignore
  }
  return undefined;
}

function resolveApiBase(apiBase?: string): string {
  const base = (apiBase ?? getEnvApiBase() ?? "").trim();
  if (base) return base.replace(/\/+$/, "");
  if (typeof window !== "undefined" && window.location?.origin) return window.location.origin;
  return "";
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

async function fetchJson(url: string, signal?: AbortSignal): Promise<unknown> {
  const res = await fetch(url, { signal, headers: { Accept: "application/json" } });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
  if (res.status === 204) return null;

  const ct = res.headers.get("content-type") ?? "";
  if (!ct.includes("application/json")) {
    const text = await res.text();
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }
  return await res.json();
}

function parseFrameIndexFromPathOrLabel(s: string): number | undefined {
  const name = s.split("/").pop() ?? s;
  // viz_0001.png or viz_1.png
  const m = name.match(/^viz_(\d+)\.png$/i);
  if (m) {
    const n = Number(m[1]);
    if (Number.isFinite(n)) return n;
  }
  // singleton viz.png — treat as 0 for stable ordering
  if (/^viz\.png$/i.test(name)) return 0;
  return undefined;
}

function normalizeFrames(frames: ReadonlyArray<FrameLike> | undefined, runId?: string, apiBase?: string): NormalizedFrame[] {
  if (!frames) return [];
  const base = resolveApiBase(apiBase);

  const norm: NormalizedFrame[] = frames.map((f, i) => {
    if (typeof f === "string") {
      const s = f.trim();
      const looksUrl = s.startsWith("http://") || s.startsWith("https://") || s.startsWith("/");
      const label = s.split("/").slice(-1)[0] || `frame_${i}`;
      const parsedIdx = parseFrameIndexFromPathOrLabel(s);
      const idx = typeof parsedIdx === "number" ? parsedIdx : i;

      const url = looksUrl
        ? s
        : runId
          ? `${base}/api/runs/${encodeURIComponent(runId)}/files?path=${encodeURIComponent(s)}`
          : undefined;
      const path = looksUrl ? undefined : s;
      return { index: idx, url, path, label };
    }

    const path = typeof f.path === "string" ? f.path : undefined;
    const url0 = typeof f.url === "string" ? f.url : undefined;

    const parsedIdx = (path ? parseFrameIndexFromPathOrLabel(path) : undefined) ?? (url0 ? parseFrameIndexFromPathOrLabel(url0) : undefined);
    const idx =
      (typeof f.index === "number" && Number.isFinite(f.index) ? f.index : undefined) ??
      (typeof parsedIdx === "number" ? parsedIdx : i);

    const url =
      url0 ??
      (path && runId ? `${base}/api/runs/${encodeURIComponent(runId)}/files?path=${encodeURIComponent(path)}` : undefined);
    const label = (path ?? url0 ?? `frame_${idx}`).split("/").slice(-1)[0] || `frame_${idx}`;
    return { index: idx, path, url, label };
  });

  // Stable ordering: by index (if provided/derived) else by label.
  norm.sort((a, b) => a.index - b.index || a.label.localeCompare(b.label));
  return norm;
}

function overlayVariantPath(path: string): string {
  // Convention: *_overlay.png (best-effort; repo overlay generation varies).
  if (path.endsWith(".png")) return path.replace(/\.png$/i, "_overlay.png");
  return path + "_overlay.png";
}

export function FrameViewer(props: FrameViewerProps) {
  const { runId, frames: framesProp, isRunning = false, apiBase, pollIntervalMs = 2000, height = 420 } = props;

  const [frames, setFrames] = useState<NormalizedFrame[]>(() => normalizeFrames(framesProp, runId, apiBase));
  const [loading, setLoading] = useState<boolean>(!framesProp && !!runId);
  const [error, setError] = useState<string | null>(null);

  const [followLatest, setFollowLatest] = useState<boolean>(true);
  const [selectedIndex, setSelectedIndex] = useState<number>(0);
  const [play, setPlay] = useState<boolean>(false);
  const [preferOverlay, setPreferOverlay] = useState<boolean>(false);

  const cacheBustRef = useRef<number>(0);

  useEffect(() => {
    // Keep internal frames in sync when used as a presentational component.
    if (framesProp) {
      const n = normalizeFrames(framesProp, runId, apiBase);
      setFrames(n);
      setLoading(false);
      setError(null);
    }
  }, [framesProp, runId, apiBase]);

  const fetchFrames = useCallback(
    async (signal?: AbortSignal) => {
      if (!runId || framesProp) return;
      const base = resolveApiBase(apiBase);
      setLoading(true);
      setError(null);

      // Default expected endpoints (best-effort). If absent, backend can still serve single viz.png via /files.
      const urls = [
        `${base}/api/runs/${encodeURIComponent(runId)}/frames`,
        `${base}/api/runs/${encodeURIComponent(runId)}/frame/list`,
      ];

      let lastErr: unknown = null;
      for (const url of urls) {
        try {
          const data = await fetchJson(url, signal);
          const list: FrameLike[] = (() => {
            if (Array.isArray(data)) return data as FrameLike[];
            if (isPlainObject(data) && Array.isArray(data.frames)) return data.frames as FrameLike[];
            return [];
          })();

          const normalized = normalizeFrames(list, runId, apiBase);
          setFrames(normalized);
          setLoading(false);
          setError(null);
          return;
        } catch (e) {
          if (e instanceof DOMException && e.name === "AbortError") return;
          lastErr = e;
        }
      }

      // Fallback: try singleton viz.png before surfacing an error
      try {
        const fallbackPath = "viz/viz.png";
        const probe = `${base}/api/runs/${encodeURIComponent(runId)}/files?path=${encodeURIComponent(fallbackPath)}`;
        const resp = await fetch(probe, { method: "HEAD", signal });
        if (resp.ok) {
          setFrames(normalizeFrames([fallbackPath], runId, apiBase));
          setLoading(false);
          setError(null);
          return;
        }
      } catch (e) {
        if (e instanceof DOMException && e.name === "AbortError") return;
      }

      // If request was aborted, do nothing (avoid false error state)
      if (signal?.aborted) return;

      setLoading(false);
      setError(lastErr instanceof Error ? lastErr.message : "Failed to load frames");
    },
    [runId, framesProp, apiBase]
  );

  // Poll frames while running (or while followLatest is enabled).
  useEffect(() => {
    if (!runId || framesProp) return;
    const controller = new AbortController();
    void fetchFrames(controller.signal);

    const shouldPoll = isRunning || followLatest;
    if (!shouldPoll) return () => controller.abort();

    const t = window.setInterval(() => {
      cacheBustRef.current = Date.now();
      void fetchFrames(controller.signal);
    }, Math.max(750, pollIntervalMs));

    return () => {
      controller.abort();
      window.clearInterval(t);
    };
  }, [runId, framesProp, isRunning, followLatest, pollIntervalMs, fetchFrames]);

  // Keep selectedIndex aligned with "follow latest".
  useEffect(() => {
    if (!followLatest) return;
    if (frames.length === 0) return;
    setSelectedIndex(frames.length - 1);
  }, [followLatest, frames.length]);

  // Animation playback.
  useEffect(() => {
    if (!play) return;
    if (frames.length <= 1) return;

    const fps = 4;
    const t = window.setInterval(() => {
      setSelectedIndex((idx) => (idx + 1) % frames.length);
    }, Math.round(1000 / fps));

    return () => window.clearInterval(t);
  }, [play, frames.length]);

  const selected = useMemo(() => {
    if (frames.length === 0) return null;
    const idx = Math.min(Math.max(0, selectedIndex), frames.length - 1);
    return frames[idx];
  }, [frames, selectedIndex]);

  const overlayAvailable = useMemo(() => {
    if (!selected?.path) return false;
    const ov = overlayVariantPath(selected.path);
    return frames.some((f) => f.path === ov);
  }, [selected, frames]);

  const selectedUrl = useMemo(() => {
    if (!selected) return null;

    // If prefer overlay and overlay exists, redirect to overlay frame URL if present; else keep base.
    if (preferOverlay && selected.path) {
      const ovPath = overlayVariantPath(selected.path);
      const ov = frames.find((f) => f.path === ovPath);
      if (ov?.url) return ov.url;
      if (ovPath && runId) {
        const base = resolveApiBase(apiBase);
        return `${base}/api/runs/${encodeURIComponent(runId)}/files?path=${encodeURIComponent(ovPath)}`;
      }
    }
    return selected.url ?? null;
  }, [selected, preferOverlay, frames, runId, apiBase]);

  const srcWithBust = useMemo(() => {
    if (!selectedUrl) return null;
    // Avoid caching issues while live: append a cache-busting query param (FR-5 robustness).
    const bust = isRunning || followLatest ? cacheBustRef.current || Date.now() : 0;
    const hasQ = selectedUrl.includes("?");
    return `${selectedUrl}${hasQ ? "&" : "?"}v=${encodeURIComponent(String(bust))}`;
  }, [selectedUrl, isRunning, followLatest]);

  const onScrub = (idx: number) => {
    setSelectedIndex(idx);
    setFollowLatest(false);
    setPlay(false);
  };

  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "baseline" }}>
        <h3 style={{ margin: 0 }}>Visualization Frames</h3>
        <div style={{ fontSize: 12, opacity: 0.75 }}>
          {loading ? "Loading…" : `${frames.length} frame${frames.length === 1 ? "" : "s"}`}
          {runId ? <span> · runId {runId}</span> : null}
        </div>
      </div>

      <div
        style={{
          border: "1px solid rgba(0,0,0,0.12)",
          borderRadius: 8,
          background: "#fff",
          padding: 10,
          display: "flex",
          flexDirection: "column",
          gap: 10,
        }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <label style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12 }}>
            <input type="checkbox" checked={followLatest} onChange={(e) => setFollowLatest(e.target.checked)} />
            Follow latest
          </label>

          <button
            type="button"
            onClick={() => setPlay((p) => !p)}
            disabled={frames.length <= 1}
            style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid rgba(0,0,0,0.2)", background: "#fff" }}
          >
            {play ? "Pause" : "Play"}
          </button>

          {selected ? (
            <div style={{ fontSize: 12, opacity: 0.85 }}>
              Showing: <strong>{selected.label}</strong> ({selectedIndex + 1}/{frames.length})
            </div>
          ) : null}

          {overlayAvailable ? (
            <label style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, marginLeft: "auto" }}>
              <input type="checkbox" checked={preferOverlay} onChange={(e) => setPreferOverlay(e.target.checked)} />
              Use overlay if available
            </label>
          ) : null}
        </div>

        {error ? (
          <div style={{ fontSize: 12, color: "#8a0000", background: "#ffecec", border: "1px solid #ffbcbc", padding: 10, borderRadius: 6 }}>
            {error}
          </div>
        ) : null}

        {/* Image viewport */}
        <div
          style={{
            height,
            display: "grid",
            placeItems: "center",
            background: "rgba(0,0,0,0.03)",
            borderRadius: 6,
            overflow: "hidden",
          }}
        >
          {srcWithBust ? (
            <img
              src={srcWithBust}
              alt={selected ? `Frame ${selected.label}` : "Frame"}
              style={{ maxWidth: "100%", maxHeight: "100%", objectFit: "contain" }}
            />
          ) : (
            <div style={{ fontSize: 12, opacity: 0.8, padding: 12, textAlign: "center" }}>
              No frames available yet.
              <div style={{ marginTop: 6, opacity: 0.75 }}>
                Expected files like <code>viz/viz_*.png</code> (fallback <code>viz/viz.png</code>) per repo conventions.
              </div>
            </div>
          )}
        </div>

        {/* Scrubber */}
        {frames.length > 1 ? (
          <div style={{ display: "flex", gap: 10, alignItems: "center" }}>
            <input
              type="range"
              min={0}
              max={Math.max(0, frames.length - 1)}
              value={Math.min(selectedIndex, Math.max(0, frames.length - 1))}
              onChange={(e) => onScrub(Number(e.target.value))}
              style={{ flex: 1 }}
            />
            <div style={{ width: 74, textAlign: "right", fontSize: 12, opacity: 0.85 }}>
              {selectedIndex + 1}/{frames.length}
            </div>
          </div>
        ) : null}

        {/* Manual refresh for non-running mode */}
        {!framesProp && runId ? (
          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <button
              type="button"
              onClick={() => {
                cacheBustRef.current = Date.now();
                void fetchFrames();
              }}
              style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid rgba(0,0,0,0.2)", background: "#fff" }}
            >
              Refresh
            </button>
          </div>
        ) : null}
      </div>
    </section>
  );
}

export default FrameViewer;
