import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

/**
 * Live Run Monitor (Design Doc §8 day-one #3; FR-5 + FR-6 + FR-4).
 *
 * Design Doc requirements implemented here:
 * - FR-5: run state, live log stream (filter), live convergence plot, perf panel, live frame viewer.
 * - FR-6: pause/resume/terminate/write_every/snapshot token (snapshot must be unique string token).
 * - FR-4: robust log ingestion safety (event fallback: event/msg/message, iter/resid variants, ts parsing).
 * - FR-3: surface run dir contract paths (manifest, metrics, events/evidence logs, viz frames, artifacts, plots, report.html).
 *
 * Repo anchors inspected (to align with Electrodrive reality):
 * - electrodrive/researched/ws.py exposes:
 *     WS /ws/runs/{run_id}/events (streams {"type":"event", ...canonical fields...}, merging events.jsonl + evidence_log.jsonl + train_log.jsonl + metrics.jsonl)
 *     WS /ws/runs/{run_id}/frames (streams {"type":"frame", name,index,mtime,bytes_b64})
 *     WS /ws/runs/{run_id}/stdout and /stderr for raw streaming (FR-5 structured + raw).
 * - electrodrive/utils/logging.py: JsonlLogger writes {"ts","level","msg",...} to events.jsonl (so UI must not assume "event").
 * - electrodrive/viz/live_console.py: prefers events.jsonl but falls back to evidence_log.jsonl; parses event name from event/msg/message and resid variants.
 * - electrodrive/live/controls.py: ControlState fields + semantics; snapshot is str|null; seq/ack_seq fields exist; unknown keys preserved on merge.
 * - electrodrive/researched/app.py: REST mounted at /api/v1; WS mounted at /ws.
 */

type Workflow = "solve" | "images_discover" | "learn_train" | "fmm_suite" | (string & {});
type RunStatus = "running" | "success" | "error" | "killed" | (string & {});
type LogLevel = "debug" | "info" | "warning" | "error" | (string & {});
type JsonObject = Record<string, unknown>;

type ManifestV1 = {
  // Design Doc §5.1
  run_id: string;
  workflow: Workflow;
  started_at: string;
  ended_at?: string | null;
  status: RunStatus;

  git?: { sha?: string | null; branch?: string | null; dirty?: boolean | null; diff_summary?: string | null; [k: string]: unknown };
  env?: { python_version?: string; torch_version?: string; device?: string; dtype?: string; host?: string; [k: string]: unknown };
  inputs?: { spec_path?: string | null; config_path?: string | null; config?: unknown; command?: string[]; [k: string]: unknown };
  outputs?: { metrics_json?: string | null; events_jsonl?: string | null; evidence_log_jsonl?: string | null; viz_dir?: string | null; plots_dir?: string | null; report_html?: string | null; [k: string]: unknown };

  // Repo: may contain "researched" status block, etc. (electrodrive/researched/api.py _ensure_manifest_v1_shape)
  researched?: JsonObject;

  [k: string]: unknown;
};

type CanonicalLogRecord = {
  // Design Doc §5.2 canonical record; repo ws.py sends these fields in event messages.
  ts?: string;
  t?: number;
  level?: LogLevel | string;
  event: string;
  fields: JsonObject;
  iter?: number;
  resid?: number;
  resid_precond?: number;
  resid_true?: number;
  source?: string;
  event_source?: string;
  [k: string]: unknown;
};

type ArtifactSummary = {
  path: string;
  is_dir: boolean;
  size?: number;
  mtime?: number;
  url?: string;
  [k: string]: unknown;
};

type ControlPatch = {
  pause?: boolean;
  terminate?: boolean;
  write_every?: number | null;
  snapshot?: string | null;
  // allow passthrough extras
  [k: string]: unknown;
};

type RunState = {
  status?: RunStatus | string;
  exit_code?: number | null;
  message?: string;
  updated_at?: number;
};

type Frame = {
  name: string;
  index: number; // -1 if unknown
  mtime?: number;
  url?: string; // data URL
  bytes_b64?: string;
};

const DEFAULT_REST_PREFIX = "/api"; // prompt default
const DEFAULT_WS_PREFIX = "/ws"; // prompt default WS base; matches repo (electrodrive/researched/app.py prefix="/ws")

const REST_PREFIX_CANDIDATES = [
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "",
  "/api/v1", // repo
  DEFAULT_REST_PREFIX,
].map((s) => String(s || "").trim()).filter(Boolean);

const MAX_FRAME_BYTES_CACHE = 25; // keep bytes only for last N frames to avoid OOM

function uniq(xs: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const x of xs) {
    const v = x.trim();
    if (!v) continue;
    if (seen.has(v)) continue;
    seen.add(v);
    out.push(v);
  }
  return out;
}

function normalizeBase(base: string): string {
  const b = (base || "").trim();
  if (!b) return "";
  return b.replace(/\/+$/, "");
}

function toErrorMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  try { return JSON.stringify(e); } catch { return String(e); }
}

async function readJson(resp: Response): Promise<unknown> {
  const ct = (resp.headers.get("content-type") || "").toLowerCase();
  if (resp.status === 204) return null;
  if (ct.includes("application/json")) {
    try { return await resp.json(); } catch { return null; }
  }
  try { return (await resp.text()) || null; } catch { return null; }
}

async function fetchJsonWithFallback<T>(
  pathCandidates: string[],
  init: RequestInit & { signal?: AbortSignal } = {},
): Promise<T> {
  const bases = uniq(REST_PREFIX_CANDIDATES.map(normalizeBase).filter(Boolean));
  let lastErr: unknown = null;

  for (const base of bases) {
    for (const p of pathCandidates) {
      const path = p.startsWith("/") ? p : `/${p}`;
      const url = `${base}${path}`;
      try {
        const resp = await fetch(url, { ...init, credentials: "same-origin", headers: { Accept: "application/json", ...(init.headers || {}) } });
        const body = await readJson(resp);
        if (!resp.ok) {
          const msg = typeof body === "string" && body ? body : `HTTP ${resp.status} for ${url}`;
          const err = new Error(msg);
          (err as any).status = resp.status;
          throw err;
        }
        return body as T;
      } catch (e) {
        lastErr = e;
        const status = (e as any)?.status;
        if (status === 404 || status === 405) continue;
        continue;
      }
    }
  }
  throw lastErr ?? new Error("Request failed");
}

/** Convert current origin (http/https) to ws/wss. */
function wsOrigin(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}`;
}

/** Make an absolute ws(s) URL for a given path. */
function wsUrl(path: string): string {
  if (path.startsWith("ws://") || path.startsWith("wss://")) return path;
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${wsOrigin()}${p}`;
}

function safeNumber(v: unknown): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim()) {
    const x = Number(v);
    if (Number.isFinite(x)) return x;
  }
  return undefined;
}

function safeInt(v: unknown): number | undefined {
  const n = safeNumber(v);
  if (n === undefined) return undefined;
  return Number.isFinite(n) ? Math.trunc(n) : undefined;
}

function safeString(v: unknown): string | undefined {
  if (typeof v === "string") return v;
  if (v === null || v === undefined) return undefined;
  try { return String(v); } catch { return undefined; }
}

function parseIsoToEpochSeconds(ts: string): number | undefined {
  const s = ts.trim();
  if (!s) return undefined;
  const d = new Date(s);
  const ms = d.getTime();
  if (!Number.isFinite(ms)) return undefined;
  return ms / 1000;
}

/**
 * Frontend normalization safety (Design Doc FR-4; repo reality in electrodrive/utils/logging.py and viz/live_console.py):
 * - event name fallback: rec.event ?? rec.msg ?? rec.message
 * - iter variants: iter/iters/step/k
 * - resid variants: resid/resid_precond/resid_true and *_l2 variants
 * - timestamps: parse ts string or use numeric t; fallback to now
 */
function normalizeLogRecord(raw: unknown): CanonicalLogRecord | null {
  if (!raw || typeof raw !== "object") return null;
  const r = raw as Record<string, unknown>;

  // ws.py event messages are {"type":"event", ...fields...}
  const isEventWrapper = typeof r.type === "string" && r.type === "event";

  const candidate = isEventWrapper ? r : r;

  const eventName =
    safeString(candidate.event) ??
    safeString((candidate as any).msg) ??
    safeString((candidate as any).message) ??
    "";

  const ts = safeString(candidate.ts);
  const t = safeNumber(candidate.t) ?? (ts ? parseIsoToEpochSeconds(ts) : undefined) ?? Date.now() / 1000;

  const level = safeString(candidate.level) ?? safeString((candidate as any).lvl) ?? "info";

  const iter =
    safeInt(candidate.iter) ??
    safeInt((candidate as any).iters) ??
    safeInt((candidate as any).step) ??
    safeInt((candidate as any).k);

  const resid_precond =
    safeNumber((candidate as any).resid_precond) ??
    safeNumber((candidate as any).resid_precond_l2);

  const resid_true =
    safeNumber((candidate as any).resid_true) ??
    safeNumber((candidate as any).resid_true_l2);

  const resid =
    safeNumber((candidate as any).resid) ??
    resid_precond ??
    resid_true;

  const fieldsObj = (candidate.fields && typeof candidate.fields === "object" && !Array.isArray(candidate.fields))
    ? (candidate.fields as JsonObject)
    : (() => {
        // If backend did not provide a "fields" dict, create one by excluding known keys.
        const skip = new Set([
          "type", "ts", "t", "level", "event", "msg", "message",
          "iter", "iters", "step", "k",
          "resid", "resid_precond", "resid_true", "resid_precond_l2", "resid_true_l2",
          "fields", "source", "event_source",
        ]);
        const out: JsonObject = {};
        for (const [k, v] of Object.entries(candidate)) {
          if (skip.has(k)) continue;
          out[k] = v;
        }
        return out;
      })();

  return {
    ts,
    t,
    level,
    event: eventName,
    fields: fieldsObj,
    iter,
    resid,
    resid_precond,
    resid_true,
    source: safeString((candidate as any).source),
    event_source: safeString((candidate as any).event_source),
  };
}

function makeSnapshotToken(): string {
  // FR-6 + repo electrodrive/live/controls.py: snapshot is str|null token (one-shot marker), not boolean.
  const uuid =
    typeof crypto !== "undefined" && "randomUUID" in crypto && typeof (crypto as any).randomUUID === "function"
      ? (crypto as any).randomUUID()
      : `${Date.now()}-${Math.random().toString(16).slice(2)}`;
  return `${uuid}@${new Date().toISOString()}`;
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

function formatSeconds(s: number): string {
  if (!Number.isFinite(s) || s < 0) return "—";
  if (s < 60) return `${s.toFixed(1)}s`;
  const m = Math.floor(s / 60);
  const rem = s - m * 60;
  if (m < 60) return `${m}m ${Math.floor(rem)}s`;
  const h = Math.floor(m / 60);
  const mm = m - h * 60;
  return `${h}h ${mm}m`;
}

function ShortKV(props: { k: string; v: string }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "140px 1fr", gap: 8, fontSize: 12 }}>
      <div style={{ color: "#6b7280" }}>{props.k}</div>
      <div style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" }}>{props.v}</div>
    </div>
  );
}

function SmallButton(props: { children: string; onClick?: () => void; disabled?: boolean; kind?: "primary" | "danger" | "default"; title?: string }) {
  const kind = props.kind ?? "default";
  const bg = kind === "primary" ? "#111827" : kind === "danger" ? "#b91c1c" : "#ffffff";
  const fg = kind === "primary" || kind === "danger" ? "#ffffff" : "#111827";
  const border = kind === "default" ? "1px solid #e5e7eb" : "1px solid transparent";
  return (
    <button
      type="button"
      title={props.title}
      disabled={props.disabled}
      onClick={props.onClick}
      style={{
        padding: "6px 10px",
        borderRadius: 10,
        border,
        background: bg,
        color: fg,
        cursor: props.disabled ? "not-allowed" : "pointer",
        opacity: props.disabled ? 0.6 : 1,
      }}
    >
      {props.children}
    </button>
  );
}

/** Minimal SVG line chart (no external libs) — required by prompt. */
function LineChart(props: { points: { x: number; y: number }[]; width?: number; height?: number; yLog?: boolean; title?: string }) {
  const width = props.width ?? 760;
  const height = props.height ?? 220;
  const pad = { l: 44, r: 14, t: 14, b: 28 };

  const pts = props.points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (pts.length < 2) {
    return (
      <div style={{ width, height, display: "grid", placeItems: "center", color: "#6b7280", border: "1px solid #e5e7eb", borderRadius: 12 }}>
        No data yet
      </div>
    );
  }

  const xMin = Math.min(...pts.map((p) => p.x));
  const xMax = Math.max(...pts.map((p) => p.x));

  const yVals = pts.map((p) => p.y);
  const allPositive = yVals.every((y) => y > 0);
  const yLog = props.yLog ?? allPositive;

  const yTx = (y: number) => (yLog ? Math.log10(Math.max(y, 1e-300)) : y);

  const yT = pts.map((p) => yTx(p.y));
  let yMin = Math.min(...yT);
  let yMax = Math.max(...yT);
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }

  const W = width;
  const H = height;
  const innerW = W - pad.l - pad.r;
  const innerH = H - pad.t - pad.b;

  const sx = (x: number) => pad.l + ((x - xMin) / (xMax - xMin || 1)) * innerW;
  const sy = (y: number) => pad.t + (1 - (yTx(y) - yMin) / (yMax - yMin || 1)) * innerH;

  const d = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x).toFixed(2)} ${sy(p.y).toFixed(2)}`).join(" ");

  // ticks (simple)
  const xTicks = 5;
  const yTicks = 4;

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} style={{ border: "1px solid #e5e7eb", borderRadius: 12, background: "#fff" }} aria-label={props.title ?? "Line chart"}>
      {props.title ? <text x={pad.l} y={12} fontSize="11" fill="#6b7280">{props.title}</text> : null}

      {/* axes */}
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={H - pad.b} stroke="#e5e7eb" />
      <line x1={pad.l} y1={H - pad.b} x2={W - pad.r} y2={H - pad.b} stroke="#e5e7eb" />

      {/* grid + ticks */}
      {Array.from({ length: xTicks + 1 }).map((_, i) => {
        const t = i / xTicks;
        const x = pad.l + t * innerW;
        const xv = xMin + t * (xMax - xMin);
        return (
          <g key={`xt-${i}`}>
            <line x1={x} y1={pad.t} x2={x} y2={H - pad.b} stroke="#f3f4f6" />
            <text x={x} y={H - 10} fontSize="10" fill="#6b7280" textAnchor="middle">
              {Number.isFinite(xv) ? Math.round(xv).toString() : ""}
            </text>
          </g>
        );
      })}
      {Array.from({ length: yTicks + 1 }).map((_, i) => {
        const t = i / yTicks;
        const y = pad.t + t * innerH;
        const yv = yMax - t * (yMax - yMin);
        const label = yLog ? `1e${yv.toFixed(1)}` : yv.toFixed(2);
        return (
          <g key={`yt-${i}`}>
            <line x1={pad.l} y1={y} x2={W - pad.r} y2={y} stroke="#f3f4f6" />
            <text x={pad.l - 8} y={y + 3} fontSize="10" fill="#6b7280" textAnchor="end">
              {label}
            </text>
          </g>
        );
      })}

      {/* data */}
      <path d={d} fill="none" stroke="#111827" strokeWidth="1.5" />
    </svg>
  );
}

export default function RunMonitor() {
  const { runId } = useParams();
  const id = String(runId || "").trim();

  const [manifest, setManifest] = useState<ManifestV1 | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactSummary[] | null>(null);

  const [runState, setRunState] = useState<RunState>({ status: "running" });
  const [controlsBusy, setControlsBusy] = useState(false);

  const [events, setEvents] = useState<CanonicalLogRecord[]>([]);
  const [rawLines, setRawLines] = useState<{ stream: "stdout" | "stderr"; t: number; text: string }[]>([]);
  const [sourcesSeen, setSourcesSeen] = useState<Set<string>>(() => new Set());

  const [levelFilter, setLevelFilter] = useState<string>("all");
  const [substrFilter, setSubstrFilter] = useState<string>("");

  const [points, setPoints] = useState<{ x: number; y: number }[]>([]);

  const [frames, setFrames] = useState<Frame[]>([]);
  const [frameIndex, setFrameIndex] = useState<number>(0);

  const [wsStatus, setWsStatus] = useState<string>("disconnected");
  const [err, setErr] = useState<string | null>(null);

  const eventsWsRef = useRef<WebSocket | null>(null);
  const framesWsRef = useRef<WebSocket | null>(null);
  const stdoutWsRef = useRef<WebSocket | null>(null);
  const stderrWsRef = useRef<WebSocket | null>(null);

  const reconnectRef = useRef<{ events: number; frames: number; stdout: number; stderr: number }>({ events: 0, frames: 0, stdout: 0, stderr: 0 });

  // Generation counter to prevent zombie reconnects / stale socket updates.
  const wsGenRef = useRef(0);

  const startedEpoch = useMemo(() => {
    const s = manifest?.started_at || "";
    const t = s ? parseIsoToEpochSeconds(s) : undefined;
    return t;
  }, [manifest?.started_at]);

  const elapsed = useMemo(() => {
    if (!startedEpoch) return undefined;
    return Date.now() / 1000 - startedEpoch;
  }, [startedEpoch, events.length]); // update on new events

  const filteredEvents = useMemo(() => {
    const lvl = levelFilter.toLowerCase();
    const sub = substrFilter.trim().toLowerCase();
    return events.filter((e) => {
      if (lvl !== "all" && String(e.level || "").toLowerCase() !== lvl) return false;
      if (sub) {
        const hay = `${e.event} ${JSON.stringify(e.fields || {})}`.toLowerCase();
        if (!hay.includes(sub)) return false;
      }
      return true;
    });
  }, [events, levelFilter, substrFilter]);

  const latestPerf = useMemo(() => {
    // Parse best-effort perf signals from recent events:
    // - JsonlLogger smart_health may include gpu_mem_alloc_mb/gpu_mem_reserved_mb (electrodrive/utils/logging.py)
    // - live_console reads metrics.json gpu_mem_peak_mb.* (electrodrive/viz/live_console.py)
    let gpuAlloc: number | undefined;
    let gpuReserved: number | undefined;
    let lastIterDt: number | undefined;

    // estimate iter dt from last few points by time delta in event stream (t field)
    const withTime = events.filter((e) => typeof e.t === "number").slice(0, 50).reverse();
    for (let i = withTime.length - 1; i >= 1; i--) {
      const a = withTime[i - 1];
      const b = withTime[i];
      if (typeof a.iter === "number" && typeof b.iter === "number" && b.iter > a.iter) {
        const dt = (b.t! - a.t!) / (b.iter - a.iter);
        if (Number.isFinite(dt) && dt > 0) lastIterDt = dt;
      }
    }

    for (const e of events.slice(0, 200)) {
      const f = e.fields || {};
      const a = safeNumber(f.gpu_mem_alloc_mb) ?? safeNumber((e as any).gpu_mem_alloc_mb);
      const r = safeNumber(f.gpu_mem_reserved_mb) ?? safeNumber((e as any).gpu_mem_reserved_mb);
      if (a !== undefined) gpuAlloc = a;
      if (r !== undefined) gpuReserved = r;
      if (gpuAlloc !== undefined && gpuReserved !== undefined) break;
    }

    return { gpuAlloc, gpuReserved, lastIterDt };
  }, [events]);

  const latestFrame = useMemo(() => {
    if (frames.length === 0) return null;
    const idx = clamp(frameIndex, 0, frames.length - 1);
    return frames[idx];
  }, [frames, frameIndex]);
  const latestFrameIndex = latestFrame?.index;

  const refreshManifest = async (signal?: AbortSignal) => {
    if (!id) return;
    setErr(null);

    try {
      const rid = encodeURIComponent(id);

      // Default prompt: GET /api/runs/{runId}/manifest
      // Repo: often returned via GET /api/v1/runs/{runId} (electrodrive/researched/app.py).
      const raw = await fetchJsonWithFallback<unknown>(
        [`/runs/${rid}/manifest`, `/runs/${rid}`],
        { method: "GET", signal },
      );

      const man = (raw && typeof raw === "object" && (raw as any).manifest) ? (raw as any).manifest : raw;
      if (man && typeof man === "object") {
        setManifest(man as ManifestV1);

        // Use manifest status as a fallback run state (FR-5 "run state: running/exited").
        const st = safeString((man as any).status);
        if (st) setRunState((prev) => ({ ...prev, status: st }));

        const exitCode = safeInt((man as any).exit_code ?? (man as any)?.researched?.exit_code);
        if (exitCode !== undefined) setRunState((prev) => ({ ...prev, exit_code: exitCode }));
      }
    } catch (e) {
      setErr(toErrorMessage(e));
    }
  };

  const refreshArtifacts = async (signal?: AbortSignal) => {
    if (!id) return;
    try {
      const rid = encodeURIComponent(id);
      // Default prompt: GET /api/runs/{runId}/artifacts
      const raw = await fetchJsonWithFallback<unknown>([`/runs/${rid}/artifacts`], { method: "GET", signal });
      const items = (raw && typeof raw === "object" && Array.isArray((raw as any).artifacts)) ? (raw as any).artifacts : raw;
      setArtifacts(Array.isArray(items) ? (items as ArtifactSummary[]) : []);
    } catch {
      setArtifacts([]);
    }
  };

  // Initial fetch (manifest + artifacts) and periodic poll for run state (FR-5: run state + exit code).
  useEffect(() => {
    if (!id) return;
    const ac = new AbortController();
    void refreshManifest(ac.signal);
    void refreshArtifacts(ac.signal);

    // Poll manifest every ~2s to reflect status transitions even if WS doesn't provide run_state.
    const h = window.setInterval(() => {
      void refreshManifest(ac.signal);
    }, 2000);

    return () => {
      ac.abort();
      window.clearInterval(h);
    };
  }, [id]);

  const closeWs = () => {
    for (const ref of [eventsWsRef, framesWsRef, stdoutWsRef, stderrWsRef]) {
      try { ref.current?.close(); } catch { /* ignore */ }
      ref.current = null;
    }
  };

  const connectEventsWs = (
    kind: "events" | "stdout" | "stderr",
    attempt: number,
    gen: number,
    candIndex: number = 0,
  ) => {
    if (!id) return;
    if (wsGenRef.current !== gen) return; // cancelled / stale

    const rid = encodeURIComponent(id);

    // IMPORTANT: repo-specific endpoint FIRST; multiplex fallback second.
    const pathCandidates =
      kind === "events"
        ? [
            `${DEFAULT_WS_PREFIX}/runs/${rid}/events`,
            `${DEFAULT_WS_PREFIX}/runs/${rid}`, // optional legacy/multiplex fallback
          ]
        : [
            `${DEFAULT_WS_PREFIX}/runs/${rid}/${kind}`,
          ];

    if (candIndex >= pathCandidates.length) {
      // All candidates failed for this attempt; do a backoff reconnect.
      if (attempt < 3 && wsGenRef.current === gen) {
        const delay = 250 * Math.pow(2, attempt);
        window.setTimeout(() => connectEventsWs(kind, attempt + 1, gen, 0), delay);
      }
      return;
    }

    const setRef = (sock: WebSocket | null) => {
      if (kind === "events") eventsWsRef.current = sock;
      if (kind === "stdout") stdoutWsRef.current = sock;
      if (kind === "stderr") stderrWsRef.current = sock;
    };

    // Close any existing socket of this kind before reopening.
    try { (kind === "events" ? eventsWsRef.current : kind === "stdout" ? stdoutWsRef.current : stderrWsRef.current)?.close(); } catch {}
    setRef(null);

    const u = wsUrl(pathCandidates[candIndex]);
    const ws = new WebSocket(u);
    setRef(ws);

    let opened = false;

    ws.onopen = () => {
      if (wsGenRef.current !== gen) return;
      opened = true;
      setWsStatus((s) => (s === "disconnected" ? `${kind}:open` : `${s}, ${kind}:open`));
    };

    ws.onerror = () => {
      if (wsGenRef.current !== gen) return;
      setWsStatus((s) => (s.includes(`${kind}:error`) ? s : `${s}, ${kind}:error`));
    };

    ws.onclose = () => {
      if (wsGenRef.current !== gen) return; // do NOT reconnect if stale/unmounted
      setWsStatus((s) => (s.includes(`${kind}:closed`) ? s : `${s}, ${kind}:closed`));
      setRef(null);

      // If it never opened, assume wrong endpoint → try next candidate immediately.
      if (!opened) {
        connectEventsWs(kind, attempt, gen, candIndex + 1);
        return;
      }

      // If it DID open and then closed, reconnect same candidate with backoff.
      if (attempt < 3) {
        const delay = 250 * Math.pow(2, attempt);
        window.setTimeout(() => connectEventsWs(kind, attempt + 1, gen, candIndex), delay);
      }
    };

    ws.onmessage = (ev) => {
      if (wsGenRef.current !== gen) return;

      const text = String(ev.data ?? "");

      // stdout/stderr endpoints may send plain text; handle that safely.
      if ((kind === "stdout" || kind === "stderr") && (text[0] !== "{" && text[0] !== "[")) {
        setRawLines((prev) => [{ stream: kind, t: Date.now() / 1000, text }, ...prev].slice(0, 400));
        return;
      }

      try {
        const msg = JSON.parse(text);
        if (!msg || typeof msg !== "object") return;

        const m = msg as any;

        // Repo events endpoint: {"type":"event", ...canonical fields...} (electrodrive/researched/ws.py)
        if (m.type === "event" || (typeof m.event === "string" && m.fields && typeof m.fields === "object")) {
          const rec = normalizeLogRecord(m);
          if (!rec) return;

          setEvents((prev) => [rec, ...prev].slice(0, 800));

          // Track sources to reflect FR-4 merge policy (Design Doc §1.4; ws.py merges events.jsonl + evidence_log.jsonl etc).
          const src = safeString(m.source);
          if (src) {
            setSourcesSeen((prev) => {
              if (prev.has(src)) return prev;
              const n = new Set(prev);
              n.add(src);
              return n;
            });
          }

          // Convergence points (FR-5): resid vs iter
          if (typeof rec.iter === "number" && typeof rec.resid === "number" && Number.isFinite(rec.iter) && Number.isFinite(rec.resid)) {
            setPoints((prev) => [...prev, { x: rec.iter!, y: rec.resid! }].slice(-2500));
          }
          return;
        }

        // Raw stream messages (repo): {"type":"raw","stream":"stdout|stderr","t":...,"text":...}
        if (m.type === "raw" && (m.stream === "stdout" || m.stream === "stderr") && typeof m.text === "string") {
          setRawLines((prev) => {
            const next = [{ stream: m.stream, t: safeNumber(m.t) ?? Date.now() / 1000, text: m.text }, ...prev];
            return next.slice(0, 400);
          });
          return;
        }

        // Optional run state messages (prompt default): {type:"run_state", ...}
        if (m.type === "run_state") {
          setRunState((prev) => ({
            ...prev,
            status: safeString(m.status) ?? prev.status,
            exit_code: safeInt(m.exit_code) ?? prev.exit_code,
            message: safeString(m.message) ?? prev.message,
            updated_at: Date.now(),
          }));
        }

        // Optional metrics message (prompt default)
        if (m.type === "metrics" && m.metrics && typeof m.metrics === "object") {
          // Keep as generic events for visibility; could be charted later.
          const rec = normalizeLogRecord({ event: "metrics", level: "info", t: Date.now() / 1000, fields: m.metrics });
          if (rec) setEvents((prev) => [rec, ...prev].slice(0, 800));
        }
      } catch {
        // If JSON parse fails for stdout/stderr, show as raw line.
        if (kind === "stdout" || kind === "stderr") {
          setRawLines((prev) => [{ stream: kind, t: Date.now() / 1000, text }, ...prev].slice(0, 400));
        }
      }
    };
  };

  const connectFramesWs = (attempt: number, gen: number, candIndex: number = 0) => {
    if (!id) return;
    if (wsGenRef.current !== gen) return;

    const rid = encodeURIComponent(id);

    // IMPORTANT: repo-specific endpoint FIRST; multiplex fallback second.
    const candidates = [
      `${DEFAULT_WS_PREFIX}/runs/${rid}/frames`,
      `${DEFAULT_WS_PREFIX}/runs/${rid}`, // optional legacy/multiplex fallback
    ];

    if (candIndex >= candidates.length) {
      if (attempt < 3 && wsGenRef.current === gen) {
        const delay = 250 * Math.pow(2, attempt);
        window.setTimeout(() => connectFramesWs(attempt + 1, gen, 0), delay);
      }
      return;
    }

    try { framesWsRef.current?.close(); } catch {}
    framesWsRef.current = null;

    const ws = new WebSocket(wsUrl(candidates[candIndex]));
    framesWsRef.current = ws;

    let opened = false;

    ws.onopen = () => {
      if (wsGenRef.current !== gen) return;
      opened = true;
      setWsStatus((s) => (s === "disconnected" ? "frames:open" : `${s}, frames:open`));
    };

    ws.onerror = () => {
      if (wsGenRef.current !== gen) return;
      setWsStatus((s) => (s.includes("frames:error") ? s : `${s}, frames:error`));
    };

    ws.onclose = () => {
      if (wsGenRef.current !== gen) return;
      setWsStatus((s) => (s.includes("frames:closed") ? s : `${s}, frames:closed`));
      framesWsRef.current = null;

      if (!opened) {
        connectFramesWs(attempt, gen, candIndex + 1);
        return;
      }

      if (attempt < 3) {
        const delay = 250 * Math.pow(2, attempt);
        window.setTimeout(() => connectFramesWs(attempt + 1, gen, candIndex), delay);
      }
    };

    ws.onmessage = (ev) => {
      if (wsGenRef.current !== gen) return;

      try {
        const msg = JSON.parse(String(ev.data ?? "null"));
        if (!msg || typeof msg !== "object") return;
        const m = msg as any;

        // Repo frames endpoint: {"type":"frame","name","index","mtime","bytes_b64"} (electrodrive/researched/ws.py)
        if (m.type === "frame" && typeof m.name === "string") {
          const name = m.name;
          const idx = safeInt(m.index) ?? -1;
          const mtime = safeNumber(m.mtime);
          const b64 = typeof m.bytes_b64 === "string" ? m.bytes_b64 : undefined;
          const url = b64 ? `data:image/png;base64,${b64}` : undefined;

          setFrames((prev) => {
            const existingIdx = prev.findIndex((f) => f.name === name);
            const next = [...prev];
            const frame: Frame = { name, index: idx, mtime, url, bytes_b64: b64 };
            if (existingIdx >= 0) next[existingIdx] = frame;
            else next.push(frame);

            next.sort((a, b) => {
              const ai = a.index >= 0 ? a.index : Number.MAX_SAFE_INTEGER;
              const bi = b.index >= 0 ? b.index : Number.MAX_SAFE_INTEGER;
              if (ai !== bi) return ai - bi;
              const am = a.mtime ?? 0;
              const bm = b.mtime ?? 0;
              return am - bm;
            });

            const trimmed = next.slice(0, 2000);

            // Drop heavy bytes/url for older frames to prevent memory blowups.
            const keepFrom = Math.max(0, trimmed.length - MAX_FRAME_BYTES_CACHE);
            for (let i = 0; i < keepFrom; i++) {
              if (trimmed[i].url || trimmed[i].bytes_b64) {
                trimmed[i] = { ...trimmed[i], url: undefined, bytes_b64: undefined };
              }
            }

            return trimmed;
          });

          // Auto-follow latest frame without relying on stale closures.
          setFrameIndex(Number.MAX_SAFE_INTEGER);
          return;
        }

        // Prompt multiplex frame notification: {type:"viz_frame", path,...}
        if (m.type === "viz_frame" && typeof m.path === "string") {
          const name = String(m.path);
          const url = typeof m.url === "string" ? m.url : undefined;
          setFrames((prev) => {
            const next = [...prev, { name, index: prev.length, url }];
            return next.slice(-2000);
          });
          setFrameIndex(Number.MAX_SAFE_INTEGER);
        }
      } catch {
        // ignore
      }
    };
  };

  useEffect(() => {
    if (!id) return;

    // Bump generation so all old sockets stop reconnecting / mutating state.
    wsGenRef.current += 1;
    const gen = wsGenRef.current;

    setWsStatus("connecting");
    setErr(null);

    closeWs();
    setEvents([]);
    setRawLines([]);
    setPoints([]);
    setFrames([]);
    setFrameIndex(0);
    setSourcesSeen(new Set());

    reconnectRef.current = { events: 0, frames: 0, stdout: 0, stderr: 0 };

    // Connect WS streams (repo supports /ws/runs/{id}/events and /frames; multiplex is fallback).
    connectEventsWs("events", 0, gen);
    connectFramesWs(0, gen);

    // Optional raw logs (FR-5 structured + raw): connect stdout/stderr.
    connectEventsWs("stdout", 0, gen);
    connectEventsWs("stderr", 0, gen);

    return () => {
      // Invalidate generation BEFORE closing so onclose doesn't reconnect.
      wsGenRef.current += 1;
      closeWs();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  const postControls = async (patch: ControlPatch) => {
    if (!id) return;
    setErr(null);
    setControlsBusy(true);
    try {
      const rid = encodeURIComponent(id);

      // Default prompt: POST /api/runs/{runId}/controls
      // Repo likely: POST /api/v1/runs/{runId}/control or /control/update; keep tolerant.
      await fetchJsonWithFallback<unknown>(
        [`/runs/${rid}/controls`, `/runs/${rid}/control`, `/runs/${rid}/control/update`],
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(patch),
        },
      );
    } catch (e) {
      setErr(toErrorMessage(e));
    } finally {
      setControlsBusy(false);
    }
  };

  const onPause = () => void postControls({ pause: true });
  const onResume = () => void postControls({ pause: false });
  const onTerminate = () => void postControls({ terminate: true });
  const onSnapshot = () => void postControls({ snapshot: makeSnapshotToken() });

  const [writeEveryStr, setWriteEveryStr] = useState<string>("");

  const applyWriteEvery = () => {
    const s = writeEveryStr.trim();
    if (!s) {
      void postControls({ write_every: null });
      return;
    }
    const n = Number(s);
    if (!Number.isFinite(n) || n < 1) {
      setErr("write_every must be an integer >= 1 (or empty to clear).");
      return;
    }
    void postControls({ write_every: Math.trunc(n) });
  };

  const outputsSummary = useMemo(() => {
    const out = manifest?.outputs || {};
    const events = safeString(out.events_jsonl) ?? "events.jsonl";
    const evidence = safeString(out.evidence_log_jsonl) ?? "evidence_log.jsonl";
    const metrics = safeString(out.metrics_json) ?? "metrics.json";
    const viz = safeString(out.viz_dir) ?? "viz/";
    const plots = safeString(out.plots_dir) ?? "plots/";
    const report = safeString(out.report_html) ?? "report.html";
    return { events, evidence, metrics, viz, plots, report };
  }, [manifest]);

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 18 }}>Live Monitor</h1>
          <div style={{ marginTop: 4, fontSize: 12, color: "#6b7280" }}>
            Design Doc §8 day-one • FR-5 live monitor • FR-6 controls • FR-4 normalization safety
          </div>
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <Link to={`/runs/${encodeURIComponent(id)}`}>Dashboards</Link>
          <Link to="/runs">Run Library</Link>
          <SmallButton onClick={() => void refreshManifest()} disabled={!id}>Refresh manifest</SmallButton>
          <div style={{ fontSize: 12, color: "#6b7280" }}>
            WS: <b>{wsStatus}</b>
          </div>
        </div>
      </div>

      {err ? (
        <section style={{ border: "1px solid #fecaca", background: "#fff", borderRadius: 12, padding: 12 }}>
          <div style={{ color: "#b91c1c", fontSize: 13 }}>{err}</div>
        </section>
      ) : null}

      <section style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
        <div style={{ display: "grid", gap: 10 }}>
          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 6 }}>
            <div style={{ fontWeight: 700 }}>Run</div>
            <ShortKV k="run_id" v={id || "—"} />
            <ShortKV k="workflow" v={String(manifest?.workflow ?? "—")} />
            <ShortKV k="status" v={String(runState.status ?? manifest?.status ?? "—")} />
            <ShortKV k="exit_code" v={runState.exit_code !== undefined && runState.exit_code !== null ? String(runState.exit_code) : "—"} />
            <ShortKV k="started_at" v={String(manifest?.started_at ?? "—")} />
            <ShortKV k="ended_at" v={String(manifest?.ended_at ?? "—")} />
            <ShortKV k="elapsed" v={elapsed !== undefined ? formatSeconds(elapsed) : "—"} />
            <ShortKV k="sources_seen" v={Array.from(sourcesSeen).sort().join(", ") || "—"} />
          </div>

          <div style={{ fontSize: 12, color: "#6b7280" }}>
            FR-3 contract refs: <code>{outputsSummary.metrics}</code>, <code>{outputsSummary.events}</code> and/or <code>{outputsSummary.evidence}</code>, <code>{outputsSummary.viz}</code>, <code>{outputsSummary.plots}</code>, <code>{outputsSummary.report}</code>.
          </div>
        </div>
      </section>

      <section style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
          <div>
            <div style={{ fontWeight: 700 }}>Controls (FR-6)</div>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              Repo protocol is file-based control.json (electrodrive/live/controls.py): pause/terminate/write_every/snapshot token + seq/ack_seq.
            </div>
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <SmallButton onClick={onPause} disabled={controlsBusy} title="Set pause=true">Pause</SmallButton>
            <SmallButton onClick={onResume} disabled={controlsBusy} title="Set pause=false">Resume</SmallButton>
            <SmallButton kind="danger" onClick={onTerminate} disabled={controlsBusy} title="Set terminate=true (cooperative)">Terminate</SmallButton>
            <SmallButton onClick={onSnapshot} disabled={controlsBusy} title="Set snapshot=<unique token> (NOT boolean)">Snapshot (token)</SmallButton>
          </div>
        </div>

        <div style={{ display: "flex", gap: 10, alignItems: "end", flexWrap: "wrap", marginTop: 10 }}>
          <label style={{ display: "grid", gap: 6 }}>
            <div style={{ fontSize: 12, color: "#6b7280" }}>write_every (&gt;=1; empty clears)</div>
            <input
              value={writeEveryStr}
              onChange={(e) => setWriteEveryStr(e.target.value)}
              placeholder="e.g. 5"
              style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid #e5e7eb", width: 160 }}
            />
          </label>
          <SmallButton onClick={applyWriteEvery} disabled={controlsBusy}>Apply</SmallButton>
        </div>
      </section>

      <section style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
        <div style={{ display: "grid", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontWeight: 700 }}>Convergence (FR-5)</div>
              <div style={{ fontSize: 12, color: "#6b7280" }}>
                Residual vs iteration from normalized events (FR-4; event/msg/message fallback like electrodrive/viz/live_console.py).
              </div>
            </div>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              points: <b>{points.length}</b>
            </div>
          </div>

          <LineChart
            points={points}
            yLog={true}
            title="resid vs iter (log10 y when positive)"
          />
        </div>
      </section>

      <section style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
          <div>
            <div style={{ fontWeight: 700 }}>Performance (FR-5)</div>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              Best-effort from event fields (gpu_mem_alloc_mb/gpu_mem_reserved_mb via JsonlLogger smart_health in electrodrive/utils/logging.py).
            </div>
          </div>
        </div>

        <div style={{ marginTop: 10, display: "grid", gap: 6 }}>
          <ShortKV k="elapsed" v={elapsed !== undefined ? formatSeconds(elapsed) : "—"} />
          <ShortKV k="iter_dt_est" v={latestPerf.lastIterDt !== undefined ? `${latestPerf.lastIterDt.toFixed(3)} s/iter` : "—"} />
          <ShortKV k="gpu_mem_alloc_mb" v={latestPerf.gpuAlloc !== undefined ? latestPerf.gpuAlloc.toFixed(1) : "—"} />
          <ShortKV k="gpu_mem_reserved_mb" v={latestPerf.gpuReserved !== undefined ? latestPerf.gpuReserved.toFixed(1) : "—"} />
        </div>
      </section>

      <section style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
        <div style={{ display: "grid", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontWeight: 700 }}>Frames (FR-5)</div>
              <div style={{ fontSize: 12, color: "#6b7280" }}>
                Streams <code>viz/*.png</code> as bytes via WS (repo: electrodrive/researched/ws.py /runs/{"{run_id}"}/frames).
              </div>
            </div>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              frames: <b>{frames.length}</b>
            </div>
          </div>

          {frames.length > 0 ? (
            <div style={{ display: "grid", gap: 10 }}>
              <label style={{ display: "grid", gap: 6 }}>
                <div style={{ fontSize: 12, color: "#6b7280" }}>Timeline</div>
                <input
                  type="range"
                  min={0}
                  max={Math.max(0, frames.length - 1)}
                  value={clamp(frameIndex, 0, Math.max(0, frames.length - 1))}
                  onChange={(e) => setFrameIndex(Number(e.target.value))}
                />
              </label>

              <div style={{ fontSize: 12, color: "#6b7280" }}>
                Showing: <b>{latestFrame?.name ?? "—"}</b>{" "}
                {latestFrameIndex !== undefined && latestFrameIndex !== null && latestFrameIndex >= 0 ? `(index ${latestFrameIndex})` : ""}{" "}
                {latestFrame?.mtime ? `mtime=${latestFrame.mtime.toFixed(2)}` : ""}
              </div>

              {latestFrame?.url ? (
                <img
                  alt="Latest visualization frame"
                  src={latestFrame.url}
                  style={{ maxWidth: "100%", height: "auto", borderRadius: 12, border: "1px solid #e5e7eb", background: "#f9fafb" }}
                />
              ) : (
                <div style={{ color: "#6b7280" }}>Frame received but no URL/bytes available.</div>
              )}
            </div>
          ) : (
            <div style={{ color: "#6b7280" }}>No frames yet.</div>
          )}
        </div>
      </section>

      <section style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
        <div style={{ display: "grid", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontWeight: 700 }}>Logs (FR-5; FR-4 normalization safety)</div>
              <div style={{ fontSize: 12, color: "#6b7280" }}>
                Structured events merge multiple JSONL files (repo: electrodrive/researched/ws.py merges events.jsonl + evidence_log.jsonl + train_log.jsonl + metrics.jsonl).
              </div>
            </div>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              total events: <b>{events.length}</b>
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "end" }}>
            <label style={{ display: "grid", gap: 6 }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>Level</div>
              <select
                value={levelFilter}
                onChange={(e) => setLevelFilter(e.target.value)}
                style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid #e5e7eb", minWidth: 160 }}
              >
                <option value="all">all</option>
                <option value="debug">debug</option>
                <option value="info">info</option>
                <option value="warning">warning</option>
                <option value="error">error</option>
              </select>
            </label>

            <label style={{ display: "grid", gap: 6, flex: "1 1 240px" }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>Filter substring (event or fields)</div>
              <input
                value={substrFilter}
                onChange={(e) => setSubstrFilter(e.target.value)}
                placeholder="e.g. gmres, smart_health, tile_size"
                style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid #e5e7eb" }}
              />
            </label>
          </div>

          <div
            role="log"
            aria-live="polite"
            style={{
              maxHeight: 340,
              overflow: "auto",
              border: "1px solid #e5e7eb",
              borderRadius: 12,
              background: "#f9fafb",
              padding: 10,
              fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
              fontSize: 12,
              lineHeight: 1.35,
              whiteSpace: "pre-wrap",
            }}
          >
            {filteredEvents.length === 0 ? (
              <span style={{ color: "#6b7280" }}>Waiting for events…</span>
            ) : (
              filteredEvents.slice(0, 200).map((e, i) => {
                const t = typeof e.t === "number" ? e.t.toFixed(3) : "";
                const lvl = String(e.level || "");
                const src = e.source ? ` [${e.source}]` : "";
                const iter = typeof e.iter === "number" ? ` iter=${e.iter}` : "";
                const resid = typeof e.resid === "number" ? ` resid=${e.resid}` : "";
                const line = `${t} ${lvl} ${e.event}${iter}${resid}${src}`;
                return <div key={`${t}-${i}`}>{line}</div>;
              })
            )}
          </div>

          <div style={{ fontSize: 12, color: "#6b7280" }}>
            Repo note: JsonlLogger records use <code>msg</code> for label (electrodrive/utils/logging.py) and live_console derives event name from <code>event/msg/message</code> and resid variants (electrodrive/viz/live_console.py).
          </div>
        </div>
      </section>

      <section style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
        <div style={{ display: "grid", gap: 10 }}>
          <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
            <div>
              <div style={{ fontWeight: 700 }}>Raw stdout/stderr (FR-5)</div>
              <div style={{ fontSize: 12, color: "#6b7280" }}>
                Repo provides WS <code>/ws/runs/{"{run_id}"}/stdout</code> and <code>/ws/runs/{"{run_id}"}/stderr</code> (electrodrive/researched/ws.py).
              </div>
            </div>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              lines: <b>{rawLines.length}</b>
            </div>
          </div>

          <div
            role="log"
            aria-live="polite"
            style={{
              maxHeight: 240,
              overflow: "auto",
              border: "1px solid #e5e7eb",
              borderRadius: 12,
              background: "#f9fafb",
              padding: 10,
              fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
              fontSize: 12,
              lineHeight: 1.35,
              whiteSpace: "pre-wrap",
            }}
          >
            {rawLines.length === 0 ? (
              <span style={{ color: "#6b7280" }}>No raw lines yet.</span>
            ) : (
              rawLines.slice(0, 200).map((l, i) => (
                <div key={`${l.stream}-${l.t}-${i}`}>
                  <span style={{ color: l.stream === "stderr" ? "#b91c1c" : "#6b7280" }}>
                    [{l.stream}]
                  </span>{" "}
                  {l.text}
                </div>
              ))
            )}
          </div>
        </div>
      </section>

      <section style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 10, flexWrap: "wrap" }}>
          <div>
            <div style={{ fontWeight: 700 }}>Artifacts (FR-3)</div>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              Run dir contract includes <code>artifacts/</code>, <code>plots/</code>, and <code>report.html</code>.
            </div>
          </div>
          <SmallButton onClick={() => void refreshArtifacts()} disabled={!id}>Refresh artifacts</SmallButton>
        </div>

        <div style={{ marginTop: 10, maxHeight: 220, overflow: "auto", border: "1px solid #e5e7eb", borderRadius: 12 }}>
          <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
            <thead>
              <tr style={{ textAlign: "left" }}>
                <th style={{ padding: "8px 10px", borderBottom: "1px solid #e5e7eb" }}>Path</th>
                <th style={{ padding: "8px 10px", borderBottom: "1px solid #e5e7eb" }}>Type</th>
                <th style={{ padding: "8px 10px", borderBottom: "1px solid #e5e7eb" }}>Size</th>
              </tr>
            </thead>
            <tbody>
              {!artifacts ? (
                <tr><td colSpan={3} style={{ padding: 10, color: "#6b7280" }}>Loading…</td></tr>
              ) : artifacts.length === 0 ? (
                <tr><td colSpan={3} style={{ padding: 10, color: "#6b7280" }}>No artifacts returned.</td></tr>
              ) : (
                artifacts.slice(0, 200).map((a) => (
                  <tr key={a.path}>
                    <td style={{ padding: "7px 10px", borderBottom: "1px solid #f3f4f6" }}><code>{a.path}</code></td>
                    <td style={{ padding: "7px 10px", borderBottom: "1px solid #f3f4f6" }}>{a.is_dir ? "dir" : "file"}</td>
                    <td style={{ padding: "7px 10px", borderBottom: "1px solid #f3f4f6" }}>{typeof a.size === "number" ? a.size : "—"}</td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>
      </section>
    </div>
  );
}
