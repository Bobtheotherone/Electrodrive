import { useEffect, useMemo, useRef, useState, type KeyboardEvent } from "react";
import { Link, useSearchParams } from "react-router-dom";

/**
 * ResearchED Experimental Upgrades page.
 *
 * Design Doc (must ship day one): §8 includes a dedicated “Experimental Upgrades” section.
 * Design Doc requirements implemented here:
 * - FR-9.1..FR-9.6: Dedicated Experimental Upgrades dashboards (six tabs/panels).
 * - FR-3: Run-dir artifact contract: rely only on run_dir artifacts; tolerate missing.
 * - FR-4: Defensive log normalization when sampling logs (event/msg/message; iter/resid variants; events.jsonl + evidence_log.jsonl).
 *
 * Repo anchors inspected (to align UI behavior with Electrodrive reality):
 * - JsonlLogger writes events.jsonl with fields ts/level/msg (not necessarily “event”).
 * - Legacy live console tails evidence_log.jsonl and uses event/msg/message + iter/resid variants.
 * - images_discover writes discovered_system.json and discovery_manifest.json.
 * - solve runs write manifest.json; formats may vary so tolerate missing fields.
 * - discovered_system.json contains metadata/system_metadata/images entries with optional group_info.
 * - PlotService can generate basis and gate dashboards (plots/basis_scatter.png, plots/family_mass.png, plots/gate_dashboard.json/png, plots/log_coverage.png).
 */

/* ------------------------------- Types (local) ------------------------------ */

type Workflow = "solve" | "images_discover" | "learn_train" | "fmm_suite" | (string & {});
type RunStatus = "running" | "success" | "error" | "killed" | (string & {});
type JsonObject = Record<string, unknown>;

type ManifestV1 = {
  // Design Doc §5.1 (unified v1), but solver runs may not match (repo: electrodrive/cli.py) so keep optional.
  run_id?: string;
  workflow?: Workflow;
  started_at?: string;
  ended_at?: string | null;
  status?: RunStatus | string;

  git?: { sha?: string | null; branch?: string | null; dirty?: boolean | null; diff_summary?: string | null; [k: string]: unknown };
  env?: { python_version?: string; torch_version?: string; device?: string; dtype?: string; host?: string; [k: string]: unknown };
  inputs?: { spec_path?: string | null; config_path?: string | null; config?: unknown; command?: string[]; [k: string]: unknown };
  outputs?: { metrics_json?: string | null; events_jsonl?: string | null; evidence_log_jsonl?: string | null; viz_dir?: string | null; plots_dir?: string | null; report_html?: string | null; [k: string]: unknown };
  gate?: { gate1_status?: string | null; gate2_status?: string | null; gate3_status?: string | null; structure_score?: number | null; novelty_score?: number | null; [k: string]: unknown };
  spec_digest?: JsonObject;

  // repo extras allowed
  [k: string]: unknown;
};

type ArtifactSummary = {
  path: string; // relative path in run_dir
  is_dir: boolean;
  size?: number;
  mtime?: number;
  url?: string; // optional: direct URL to fetch artifact (backend-dependent)
  [k: string]: unknown;
};

// Canonical log record (Design Doc §5.2); but we must be defensive when sampling raw JSONL (FR-4).
type CanonicalLogRecord = {
  ts?: string;
  t?: number;
  level?: string;
  event?: string;
  msg?: string;
  message?: string;
  fields?: JsonObject;
  iter?: number;
  iters?: number;
  step?: number;
  k?: number;
  resid?: number;
  resid_precond?: number;
  resid_true?: number;
  resid_precond_l2?: number;
  resid_true_l2?: number;
  [k: string]: unknown;
};

type DiscoveredSystem = {
  metadata?: JsonObject;
  system_metadata?: JsonObject;
  images?: Array<{
    type?: string;
    params?: Record<string, unknown>;
    group_info?: Record<string, unknown>;
    weight?: number;
    [k: string]: unknown;
  }>;
  [k: string]: unknown;
};

type GateDashboard = {
// plot_service.py writes plots/gate_dashboard.json (FR-9.5) and includes log_coverage (FR-9.6).
  run_id?: string;
  workflow?: string;
  gate2_status?: string;
  structure_score?: number | null;
  gate3_status?: string;
  novelty_score?: number | null;
  gate2_summary?: JsonObject | null;
  log_coverage?: {
    // From plot_service coverage snapshot:
    total_lines_seen?: number;
    total_records_parsed?: number;
    total_records_emitted?: number;
    total_json_errors?: number;
    total_non_dict_records?: number;
    dropped_by_dedup?: number;
    ingested_files?: string[];
    per_file?: Record<string, Record<string, number>>;
    event_source_counts?: Record<string, number>;
    residual_field_detection_counts?: Record<string, number>;
    last_event_t?: number | null;
    [k: string]: unknown;
  };
  warnings?: string[];
  [k: string]: unknown;
};

type CoverageComputed = {
  files_present: { events_jsonl: boolean; evidence_log_jsonl: boolean; other: string[] };
  event_name_source_counts: { event: number; msg: number; message: number; parsed_message_json: number; missing: number };
  residual_fields_detected: string[];
  total_records: number;
  parsed_records: number;
  warnings: string[];
};

/* -------------------------------- Endpoints -------------------------------- */

// Prompt default endpoints (kept as constants for easy change):
const DEFAULT_REST_PREFIX = "/api";
const DEFAULT_WS_PREFIX = "/ws";

// Repo reality: ResearchED backend mounts REST under /api/v1 and WS under /ws (electrodrive/researched/app.py; also used by our other pages).
const REST_PREFIX_CANDIDATES = [
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "",
  "/api/v1",
  DEFAULT_REST_PREFIX,
].map((s) => String(s || "").trim()).filter(Boolean);

/* --------------------------------- Utilities -------------------------------- */

const LS_RUN_IDS = "researched.upgrades.selectedRunIds.v1";
const LS_ACTIVE_TAB = "researched.upgrades.activeTab.v1";

function uniq<T>(xs: T[]): T[] {
  const out: T[] = [];
  const seen = new Set<string>();
  for (const x of xs) {
    const k = typeof x === "string" ? x : JSON.stringify(x);
    if (seen.has(k)) continue;
    seen.add(k);
    out.push(x);
  }
  return out;
}

function normalizeBase(base: string): string {
  const b = (base || "").trim();
  if (!b) return "";
  return b.replace(/\/+$/, "");
}

function normRelPath(p: string): string {
  return String(p || "")
    .trim()
    .replace(/\\/g, "/")        // windows → posix
    .replace(/^(\.\/)+/, "")    // strip leading ./
    .replace(/^\/+/, "");       // strip leading /
}

function toErrorMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  try {
    return JSON.stringify(e);
  } catch {
    return String(e);
  }
}

async function readJson(resp: Response): Promise<unknown> {
  const ct = (resp.headers.get("content-type") || "").toLowerCase();
  if (resp.status === 204) return null;
  if (ct.includes("application/json")) {
    try {
      return await resp.json();
    } catch {
      return null;
    }
  }
  try {
    const t = await resp.text();
    return t || null;
  } catch {
    return null;
  }
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
        const resp = await fetch(url, {
          ...init,
          credentials: "same-origin",
          headers: { Accept: "application/json", ...(init.headers || {}) },
        });
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
        // network errors -> continue trying other candidates
        continue;
      }
    }
  }

  throw lastErr ?? new Error("Request failed");
}

async function fetchTextWithFallback(
  pathCandidates: string[],
  init: RequestInit & { signal?: AbortSignal } = {},
): Promise<{ text: string; contentType: string }> {
  const bases = uniq(REST_PREFIX_CANDIDATES.map(normalizeBase).filter(Boolean));
  let lastErr: unknown = null;

  for (const base of bases) {
    for (const p of pathCandidates) {
      const path = p.startsWith("/") ? p : `/${p}`;
      const url = `${base}${path}`;
      try {
        const resp = await fetch(url, { ...init, credentials: "same-origin" });
        const ct = (resp.headers.get("content-type") || "").toLowerCase();
        const text = await resp.text();
        if (!resp.ok) {
          const err = new Error(text || `HTTP ${resp.status} for ${url}`);
          (err as any).status = resp.status;
          throw err;
        }
        return { text, contentType: ct };
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

function safeParseJson(text: string): { ok: true; value: unknown } | { ok: false; error: string } {
  const s = (text || "").trim();
  if (!s) return { ok: true, value: null };
  try {
    return { ok: true, value: JSON.parse(s) };
  } catch (e) {
    return { ok: false, error: toErrorMessage(e) };
  }
}

function safeNumber(v: unknown): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim()) {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

function safeString(v: unknown): string | undefined {
  if (typeof v === "string") return v;
  if (v === null || v === undefined) return undefined;
  try {
    return String(v);
  } catch {
    return undefined;
  }
}

function parseRunIdsInput(s: string): string[] {
  return s
    .split(/[,\s]+/)
    .map((x) => x.trim())
    .filter(Boolean);
}

function loadRunIdsFromStorage(): string[] {
  try {
    const raw = localStorage.getItem(LS_RUN_IDS);
    if (!raw) return [];
    const parsed = safeParseJson(raw);
    if (!parsed.ok || !Array.isArray(parsed.value)) return [];
    return parsed.value.map((x) => String(x)).filter(Boolean);
  } catch {
    return [];
  }
}

function saveRunIdsToStorage(ids: string[]) {
  try {
    localStorage.setItem(LS_RUN_IDS, JSON.stringify(uniq(ids)));
  } catch {
    // ignore
  }
}

function loadActiveTab(): string | null {
  try {
    return localStorage.getItem(LS_ACTIVE_TAB);
  } catch {
    return null;
  }
}

function saveActiveTab(tab: string) {
  try {
    localStorage.setItem(LS_ACTIVE_TAB, tab);
  } catch {
    // ignore
  }
}

function wsOrigin(): string {
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}`;
}

function wsUrl(path: string): string {
  if (path.startsWith("ws://") || path.startsWith("wss://")) return path;
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${wsOrigin()}${p}`;
}

function hashToHue(s: string): number {
  let h = 0;
  for (let i = 0; i < s.length; i++) h = (h * 31 + s.charCodeAt(i)) >>> 0;
  return h % 360;
}

function colorForKey(key: string, alpha = 0.9): string {
  const hue = hashToHue(key || "unknown");
  return `hsla(${hue}, 65%, 45%, ${alpha})`;
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.max(lo, Math.min(hi, n));
}

/* ------------------------------- Chart components ------------------------------- */

function AxisLabel(props: { x: number; y: number; text: string; anchor?: "start" | "middle" | "end" }) {
  return (
    <text x={props.x} y={props.y} fontSize="10" fill="#6b7280" textAnchor={props.anchor ?? "start"}>
      {props.text}
    </text>
  );
}

function ScatterPlot(props: {
  title?: string;
  points: Array<{ x: number; y: number; colorKey: string; shapeKey?: number | string; label?: string }>;
  width?: number;
  height?: number;
  xLabel?: string;
  yLabel?: string;
}) {
  const width = props.width ?? 760;
  const height = props.height ?? 260;
  const pad = { l: 44, r: 10, t: 18, b: 34 };

  const pts = props.points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (pts.length === 0) {
    return (
      <div style={{ height, display: "grid", placeItems: "center", border: "1px solid #e5e7eb", borderRadius: 12, color: "#6b7280" }}>
        No data
      </div>
    );
  }

  const xMin = Math.min(...pts.map((p) => p.x));
  const xMax = Math.max(...pts.map((p) => p.x));
  const yMin = Math.min(...pts.map((p) => p.y));
  const yMax = Math.max(...pts.map((p) => p.y));

  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;

  const sx = (x: number) => pad.l + ((x - xMin) / (xMax - xMin || 1)) * innerW;
  const sy = (y: number) => pad.t + (1 - (y - yMin) / (yMax - yMin || 1)) * innerH;

  const shapeFor = (k: unknown) => {
    const v = typeof k === "number" ? k : typeof k === "string" ? hashToHue(k) : 0;
    return v % 3; // 0 circle, 1 square, 2 triangle
  };

  const legendKeys = uniq(pts.map((p) => p.colorKey)).slice(0, 10);

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ border: "1px solid #e5e7eb", borderRadius: 12, background: "#fff" }}>
      {props.title ? <text x={pad.l} y={12} fontSize="11" fill="#6b7280">{props.title}</text> : null}

      {/* axes */}
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={height - pad.b} stroke="#e5e7eb" />
      <line x1={pad.l} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} stroke="#e5e7eb" />

      {/* labels */}
      <AxisLabel x={pad.l} y={height - 10} text={props.xLabel ?? "x"} anchor="start" />
      <AxisLabel x={pad.l - 10} y={pad.t + 8} text={props.yLabel ?? "y"} anchor="end" />

      {/* points */}
      {pts.slice(0, 6000).map((p, i) => {
        const cx = sx(p.x);
        const cy = sy(p.y);
        const col = colorForKey(p.colorKey, 0.75);
        const shape = shapeFor(p.shapeKey);
        const r = 3;
        const key = `${p.colorKey}-${i}`;

        if (shape === 1) {
          return (
            <rect key={key} x={cx - r} y={cy - r} width={2 * r} height={2 * r} fill={col} stroke="rgba(17,24,39,0.2)">
              <title>{p.label ?? `${p.colorKey} x=${p.x.toFixed(3)} y=${p.y.toFixed(3)}`}</title>
            </rect>
          );
        }
        if (shape === 2) {
          const d = `M ${cx} ${cy - r} L ${cx + r} ${cy + r} L ${cx - r} ${cy + r} Z`;
          return (
            <path key={key} d={d} fill={col} stroke="rgba(17,24,39,0.2)">
              <title>{p.label ?? `${p.colorKey} x=${p.x.toFixed(3)} y=${p.y.toFixed(3)}`}</title>
            </path>
          );
        }
        return (
          <circle key={key} cx={cx} cy={cy} r={r} fill={col} stroke="rgba(17,24,39,0.2)">
            <title>{p.label ?? `${p.colorKey} x=${p.x.toFixed(3)} y=${p.y.toFixed(3)}`}</title>
          </circle>
        );
      })}

      {/* legend (top-right) */}
      <g>
        {legendKeys.map((k, i) => {
          const x = width - pad.r - 160;
          const y = pad.t + 12 + i * 14;
          return (
            <g key={k}>
              <rect x={x} y={y - 9} width={10} height={10} fill={colorForKey(k)} />
              <text x={x + 14} y={y} fontSize="10" fill="#374151">
                {k}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}

function BarChart(props: {
  title?: string;
  data: Array<{ key: string; value: number; labelRight?: string }>;
  width?: number;
  height?: number;
  xLabel?: string;
  yLabel?: string;
}) {
  const width = props.width ?? 760;
  const height = props.height ?? 260;
  const pad = { l: 44, r: 14, t: 18, b: 54 };

  const rows = props.data.filter((d) => Number.isFinite(d.value)).slice(0, 30);
  if (rows.length === 0) {
    return (
      <div style={{ height, display: "grid", placeItems: "center", border: "1px solid #e5e7eb", borderRadius: 12, color: "#6b7280" }}>
        No data
      </div>
    );
  }

  const maxV = Math.max(...rows.map((d) => d.value), 1e-12);
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;
  const barW = innerW / rows.length;

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ border: "1px solid #e5e7eb", borderRadius: 12, background: "#fff" }}>
      {props.title ? <text x={pad.l} y={12} fontSize="11" fill="#6b7280">{props.title}</text> : null}

      {/* axes */}
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={height - pad.b} stroke="#e5e7eb" />
      <line x1={pad.l} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} stroke="#e5e7eb" />

      <AxisLabel x={pad.l} y={height - 10} text={props.xLabel ?? "category"} anchor="start" />
      <AxisLabel x={pad.l - 10} y={pad.t + 8} text={props.yLabel ?? "value"} anchor="end" />

      {rows.map((d, i) => {
        const h = (d.value / maxV) * innerH;
        const x = pad.l + i * barW + 2;
        const y = pad.t + (innerH - h);
        const w = Math.max(2, barW - 4);
        const fill = colorForKey(d.key, 0.85);
        return (
          <g key={d.key}>
            <rect x={x} y={y} width={w} height={h} fill={fill}>
              <title>{`${d.key}: ${d.value.toPrecision(4)}${d.labelRight ? ` (${d.labelRight})` : ""}`}</title>
            </rect>
            <text x={x + w / 2} y={height - pad.b + 12} fontSize="9" fill="#374151" textAnchor="middle">
              {d.key.length > 12 ? `${d.key.slice(0, 11)}…` : d.key}
            </text>
            {d.labelRight ? (
              <text x={x + w / 2} y={y - 3} fontSize="9" fill="#6b7280" textAnchor="middle">
                {d.labelRight}
              </text>
            ) : null}
          </g>
        );
      })}
    </svg>
  );
}

function MultiLineChart(props: {
  title?: string;
  series: Array<{ name: string; points: Array<{ x: number; y: number }> }>;
  width?: number;
  height?: number;
  xLabel?: string;
  yLabel?: string;
}) {
  const width = props.width ?? 760;
  const height = props.height ?? 260;
  const pad = { l: 44, r: 14, t: 18, b: 34 };

  const allPts = props.series.flatMap((s) => s.points).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (allPts.length < 2) {
    return (
      <div style={{ height, display: "grid", placeItems: "center", border: "1px solid #e5e7eb", borderRadius: 12, color: "#6b7280" }}>
        No data
      </div>
    );
  }

  const xMin = Math.min(...allPts.map((p) => p.x));
  const xMax = Math.max(...allPts.map((p) => p.x));
  const yMin = Math.min(...allPts.map((p) => p.y));
  const yMax = Math.max(...allPts.map((p) => p.y));
  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;

  const sx = (x: number) => pad.l + ((x - xMin) / (xMax - xMin || 1)) * innerW;
  const sy = (y: number) => pad.t + (1 - (y - yMin) / (yMax - yMin || 1)) * innerH;

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ border: "1px solid #e5e7eb", borderRadius: 12, background: "#fff" }}>
      {props.title ? <text x={pad.l} y={12} fontSize="11" fill="#6b7280">{props.title}</text> : null}

      {/* axes */}
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={height - pad.b} stroke="#e5e7eb" />
      <line x1={pad.l} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} stroke="#e5e7eb" />
      <AxisLabel x={pad.l} y={height - 10} text={props.xLabel ?? "x"} anchor="start" />
      <AxisLabel x={pad.l - 10} y={pad.t + 8} text={props.yLabel ?? "y"} anchor="end" />

      {/* series */}
      {props.series.map((s) => {
        const pts = s.points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
        if (pts.length < 2) return null;
        const d = pts
          .sort((a, b) => a.x - b.x)
          .map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x).toFixed(2)} ${sy(p.y).toFixed(2)}`)
          .join(" ");
        const col = colorForKey(s.name, 0.9);
        return (
          <path key={s.name} d={d} fill="none" stroke={col} strokeWidth="1.6">
            <title>{s.name}</title>
          </path>
        );
      })}

      {/* legend */}
      <g>
        {props.series.slice(0, 8).map((s, i) => {
          const x = width - pad.r - 180;
          const y = pad.t + 12 + i * 14;
          return (
            <g key={s.name}>
              <line x1={x} y1={y - 4} x2={x + 14} y2={y - 4} stroke={colorForKey(s.name)} strokeWidth="2" />
              <text x={x + 18} y={y} fontSize="10" fill="#374151">
                {s.name}
              </text>
            </g>
          );
        })}
      </g>
    </svg>
  );
}

/* --------------------------- Log normalization (FR-4) -------------------------- */

function normalizeLogFields(rec: CanonicalLogRecord): {
  eventName: string;
  iter?: number;
  resid?: number;
  resid_precond?: number;
  resid_true?: number;
} {
  // FR-4 defensive normalization:
  // eventName = rec.event ?? rec.msg ?? rec.message ?? ""
  // iter = rec.iter ?? rec.iters ?? rec.step ?? rec.k
  // resid = rec.resid ?? rec.resid_precond ?? rec.resid_true
  const eventName = String(rec.event ?? rec.msg ?? rec.message ?? "");
  const iter = safeNumber(rec.iter) ?? safeNumber(rec.iters) ?? safeNumber(rec.step) ?? safeNumber(rec.k);
  const rPre = safeNumber(rec.resid_precond) ?? safeNumber(rec.resid_precond_l2);
  const rTrue = safeNumber(rec.resid_true) ?? safeNumber(rec.resid_true_l2);
  const r = safeNumber(rec.resid) ?? rPre ?? rTrue;
  return {
    eventName,
    iter: iter !== undefined ? Math.trunc(iter) : undefined,
    resid: r,
    resid_precond: rPre,
    resid_true: rTrue,
  };
}

function sampleJsonlCoverage(text: string, maxLines: number): CoverageComputed {
  // FR-9.6 coverage panel best-effort when backend does not provide it:
  // Count which field is used for event name and which residual variants appear.
  const counts = { event: 0, msg: 0, message: 0, parsed_message_json: 0, missing: 0 };
  const residualDetected = new Set<string>();
  const warnings: string[] = [];
  let total = 0;
  let parsed = 0;

  const lines = text.split(/\r?\n/);
  for (let i = 0; i < lines.length && parsed < maxLines; i++) {
    const ln = (lines[i] || "").trim();
    if (!ln) continue;
    total++;
    let obj: unknown;
    try {
      obj = JSON.parse(ln);
    } catch {
      continue;
    }
    if (!obj || typeof obj !== "object" || Array.isArray(obj)) continue;
    parsed++;

    const rec = obj as CanonicalLogRecord;

    const hasEvent = Object.prototype.hasOwnProperty.call(rec, "event");
    const hasMsg = Object.prototype.hasOwnProperty.call(rec, "msg");
    const hasMessage = Object.prototype.hasOwnProperty.call(rec, "message");
    const raw = rec.event ?? rec.msg ?? rec.message;

    if (hasEvent) counts.event++;
    else if (hasMsg) counts.msg++;
    else if (hasMessage) counts.message++;
    else counts.missing++;

    // detect embedded JSON in message string (learn/train pattern in design doc FR-4)
    if ((hasMsg || hasMessage) && typeof raw === "string") {
      const s = raw.trim();
      if (s.startsWith("{") && s.endsWith("}")) {
        try {
          const parsedMsg = JSON.parse(s);
          if (parsedMsg && typeof parsedMsg === "object" && !Array.isArray(parsedMsg) && (parsedMsg as any).event) {
            counts.parsed_message_json++;
          }
        } catch {
          // ignore
        }
      }
    }

    for (const k of ["resid", "resid_precond", "resid_precond_l2", "resid_true", "resid_true_l2"]) {
      if (Object.prototype.hasOwnProperty.call(rec, k)) residualDetected.add(k);
    }
  }

  if (counts.msg > 0 && counts.event === 0) {
  // Repo truth: JsonlLogger uses msg (electrodrive/utils/logging.py)
    warnings.push("Logs use 'msg' as label (not 'event'); older parsers that require 'event' will miss events (FR-9.6).");
  }
  if (!residualDetected.has("resid") && (residualDetected.has("resid_precond") || residualDetected.has("resid_true"))) {
    warnings.push("Residual telemetry detected only in resid_precond/resid_true variants; tools expecting 'resid' may miss it (FR-9.6).");
  }

  return {
    files_present: { events_jsonl: false, evidence_log_jsonl: false, other: [] },
    event_name_source_counts: counts,
    residual_fields_detected: Array.from(residualDetected),
    total_records: total,
    parsed_records: parsed,
    warnings,
  };
}

// Prevent TypeScript noUnusedLocals failures while iterating.
// Remove when these helpers are used by this page.
void DEFAULT_WS_PREFIX;
void wsOrigin;
void wsUrl;
void clamp;
void normalizeLogFields;

/* ------------------------------ UI building blocks ----------------------------- */

function Card(props: { title: string; subtitle?: string; children: JSX.Element | JSX.Element[]; right?: JSX.Element }) {
  return (
    <section style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div>
          <div style={{ fontWeight: 700 }}>{props.title}</div>
          {props.subtitle ? <div style={{ marginTop: 4, fontSize: 12, color: "#6b7280" }}>{props.subtitle}</div> : null}
        </div>
        {props.right ?? null}
      </div>
      <div style={{ marginTop: 10 }}>{props.children}</div>
    </section>
  );
}

function SmallButton(props: { children: string; onClick?: () => void; disabled?: boolean; kind?: "default" | "primary" | "danger"; title?: string }) {
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

type TabId = "basis" | "conditioning" | "collocation" | "reference" | "gates" | "audit";
const TABS: Array<{ id: TabId; label: string; hint: string }> = [
  { id: "basis", label: "FR-9.1 Basis", hint: "discovered_system.json → scatter + family mass" },
  { id: "conditioning", label: "FR-9.2 Conditioning", hint: "col_norm telemetry; min/max fallback" },
  { id: "collocation", label: "FR-9.3 Collocation", hint: "z hist, interface distance, subtract_physical audit" },
  { id: "reference", label: "FR-9.4 Reference", hint: "reference_eval.json / plot" },
  { id: "gates", label: "FR-9.5 Gates", hint: "gate2/gate3 + trends + fingerprint hooks" },
  { id: "audit", label: "FR-9.6 Log Audit", hint: "event_source + residual variants + ingested files" },
];

function Tabs(props: { active: TabId; onChange: (t: TabId) => void }) {
  const refs = useRef<Array<HTMLButtonElement | null>>([]);

  const idx = TABS.findIndex((t) => t.id === props.active);
  const onKeyDown = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight" && e.key !== "Home" && e.key !== "End") return;
    e.preventDefault();
    let next = idx;
    if (e.key === "ArrowLeft") next = (idx - 1 + TABS.length) % TABS.length;
    if (e.key === "ArrowRight") next = (idx + 1) % TABS.length;
    if (e.key === "Home") next = 0;
    if (e.key === "End") next = TABS.length - 1;
    props.onChange(TABS[next].id);
    refs.current[next]?.focus();
  };

  return (
    <div role="tablist" aria-label="Experimental upgrades tabs" style={{ display: "flex", gap: 8, flexWrap: "wrap" }} onKeyDown={onKeyDown}>
      {TABS.map((t, i) => {
        const active = t.id === props.active;
        return (
          <button
            key={t.id}
            ref={(el) => (refs.current[i] = el)}
            role="tab"
            aria-selected={active}
            aria-controls={`panel-${t.id}`}
            id={`tab-${t.id}`}
            type="button"
            onClick={() => props.onChange(t.id)}
            style={{
              padding: "6px 10px",
              borderRadius: 10,
              border: "1px solid #e5e7eb",
              background: active ? "#e5e7eb" : "#fff",
              color: "#111827",
              cursor: "pointer",
              fontSize: 13,
            }}
            title={t.hint}
          >
            {t.label}
          </button>
        );
      })}
    </div>
  );
}

/* ------------------------------ Run data loading ------------------------------ */

type FileLoad<T> =
  | { status: "idle" }
  | { status: "loading" }
  | { status: "error"; error: string; rawSnippet?: string }
  | { status: "ok"; value: T; rawText?: string };

type RunStore = {
  runId: string;
  loadingBasics: boolean;
  basicsError?: string;

  manifest?: ManifestV1;
  artifacts?: ArtifactSummary[];

  // optional backend-provided precomputed upgrades payload (if endpoint exists)
  upgradesPayload?: JsonObject;

  // cached file loads by relative path
  files: Record<string, FileLoad<unknown>>;
};

function findArtifact(artifacts: ArtifactSummary[] | undefined, relPath: string): ArtifactSummary | undefined {
  const target = normRelPath(relPath);
  if (!target) return undefined;

  const list = artifacts || [];

  // 1) exact normalized match
  let hit = list.find((a) => normRelPath(a.path) === target);
  if (hit) return hit;

  // 2) suffix match (handles "run_dir/plots/x.json" vs "plots/x.json")
  hit = list.find((a) => {
    const ap = normRelPath(a.path);
    return ap.endsWith(`/${target}`) || target.endsWith(`/${ap}`);
  });
  if (hit) return hit;

  // 3) basename match as last resort
  const base = target.split("/").pop();
  if (!base) return undefined;
  return list.find((a) => normRelPath(a.path).split("/").pop() === base);
}

function findFirstArtifactByPredicate(artifacts: ArtifactSummary[] | undefined, pred: (a: ArtifactSummary) => boolean): ArtifactSummary | undefined {
  return (artifacts || []).find(pred);
}

async function loadArtifacts(runId: string, signal?: AbortSignal): Promise<ArtifactSummary[]> {
  const rid = encodeURIComponent(runId);
  // Default contract: GET /api/runs/{runId}/artifacts
  // Repo: /api/v1/runs/{runId}/artifacts (ResearchED backend).
  const raw = await fetchJsonWithFallback<unknown>([`/runs/${rid}/artifacts`], { method: "GET", signal });
  const items = (raw && typeof raw === "object" && Array.isArray((raw as any).artifacts)) ? (raw as any).artifacts : raw;
  return Array.isArray(items) ? (items as ArtifactSummary[]) : [];
}

async function loadManifest(runId: string, signal?: AbortSignal): Promise<ManifestV1 | null> {
  const rid = encodeURIComponent(runId);
  // Default contract: GET /api/runs/{runId}/manifest
  // Repo: GET /api/v1/runs/{runId} returns {manifest,...} (as used in prior pages); tolerate either shape.
  const raw = await fetchJsonWithFallback<unknown>([`/runs/${rid}/manifest`, `/runs/${rid}`], { method: "GET", signal });
  const man = (raw && typeof raw === "object" && (raw as any).manifest) ? (raw as any).manifest : raw;
  return man && typeof man === "object" ? (man as ManifestV1) : null;
}

async function loadUpgradesPayloadOptional(runId: string, signal?: AbortSignal): Promise<JsonObject | null> {
  const rid = encodeURIComponent(runId);
  // Optional endpoint per prompt: GET /api/runs/{runId}/upgrades
  // If missing, caller treats as null (best-effort).
  try {
    const raw = await fetchJsonWithFallback<unknown>([`/runs/${rid}/upgrades`], { method: "GET", signal });
    if (raw && typeof raw === "object") return raw as JsonObject;
    return null;
  } catch {
    return null;
  }
}

async function fetchFileText(runId: string, relPath: string, artifacts?: ArtifactSummary[], signal?: AbortSignal): Promise<string> {
  const rid = encodeURIComponent(runId);
  const path = normRelPath(relPath);

  // 1) Prefer artifact-provided URL (if backend provides direct links).
  const art = findArtifact(artifacts, path);
  if (art?.url && typeof art.url === "string") {
    const resp = await fetch(art.url, { method: "GET", signal, credentials: "same-origin" });
    if (!resp.ok) throw new Error(`Failed to fetch artifact url (${resp.status})`);
    return await resp.text();
  }

  // 2) Otherwise use default “files?path=” endpoint (prompt).
  // Be defensive: if artifacts contain a path that differs by prefix, try both.
  const queryPaths = art
    ? uniq([normRelPath(art.path), path].filter(Boolean))
    : [path];

  const candidates = queryPaths.flatMap((qp) => {
    const qs = new URLSearchParams({ path: qp }).toString();
    return [
      `/runs/${rid}/files?${qs}`,
      `/runs/${rid}/file?${qs}`,
      `/runs/${rid}/raw?${qs}`,
    ];
  });

  const res = await fetchTextWithFallback(candidates, { method: "GET", signal });
  return res.text;
}

async function readResponseTextCapped(resp: Response, maxBytes: number): Promise<string> {
  const body = resp.body;
  if (!body) {
    const t = await resp.text();
    return t.length > maxBytes ? t.slice(0, maxBytes) : t;
  }

  const reader = body.getReader();
  const chunks: Uint8Array[] = [];
  let n = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!value) continue;

    if (n + value.byteLength > maxBytes) {
      const keep = value.slice(0, Math.max(0, maxBytes - n));
      if (keep.byteLength) chunks.push(keep);
      try { await reader.cancel(); } catch { /* ignore */ }
      break;
    }

    chunks.push(value);
    n += value.byteLength;

    if (n >= maxBytes) {
      try { await reader.cancel(); } catch { /* ignore */ }
      break;
    }
  }

  const total = chunks.reduce((acc, c) => acc + c.byteLength, 0);
  const buf = new Uint8Array(total);
  let off = 0;
  for (const c of chunks) { buf.set(c, off); off += c.byteLength; }

  return new TextDecoder().decode(buf);
}

async function fetchFileTextCapped(
  runId: string,
  relPath: string,
  artifacts: ArtifactSummary[] | undefined,
  maxBytes: number,
  signal?: AbortSignal,
): Promise<string> {
  const rid = encodeURIComponent(runId);
  const path = normRelPath(relPath);

  const art = findArtifact(artifacts, path);

  const isDirectUrl = (u: string) =>
    u.startsWith("http://") || u.startsWith("https://") || u.startsWith("/");

  // If backend provides a direct URL, use it (do not prefix with REST base).
  if (art?.url && typeof art.url === "string") {
    const resp = await fetch(art.url, {
      method: "GET",
      signal,
      credentials: "same-origin",
      headers: { Range: `bytes=-${maxBytes}` },
    });
    if (!resp.ok && resp.status !== 206) throw new Error(`HTTP ${resp.status} for ${art.url}`);
    return await readResponseTextCapped(resp, maxBytes);
  }

  // Otherwise hit the raw/files endpoints. If artifacts contain a slightly different path, try both.
  const queryPaths = art
    ? uniq([normRelPath(art.path), path].filter(Boolean))
    : [path];

  const pathCandidates = queryPaths.flatMap((qp) => {
    const qs = new URLSearchParams({ path: qp }).toString();
    return [
      `/runs/${rid}/files?${qs}`,
      `/runs/${rid}/file?${qs}`,
      `/runs/${rid}/raw?${qs}`,
    ];
  });

  const bases = uniq(REST_PREFIX_CANDIDATES.map(normalizeBase).filter(Boolean));
  let lastErr: unknown = null;

  for (const base of bases) {
    for (const p of pathCandidates) {
      const url = isDirectUrl(p) ? p : `${base}${p.startsWith("/") ? p : `/${p}`}`;
      try {
        const resp = await fetch(url, {
          method: "GET",
          signal,
          credentials: "same-origin",
          headers: { Range: `bytes=-${maxBytes}` },
        });

        if (!resp.ok && resp.status !== 206) {
          const txt = await resp.text().catch(() => "");
          const err = new Error(txt || `HTTP ${resp.status} for ${url}`);
          (err as any).status = resp.status;
          throw err;
        }

        return await readResponseTextCapped(resp, maxBytes);
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

async function loadJsonFile<T>(
  runId: string,
  relPath: string,
  artifacts?: ArtifactSummary[],
  signal?: AbortSignal,
): Promise<{ ok: true; value: T; rawText: string } | { ok: false; error: string; rawSnippet?: string }> {
  try {
    const text = await fetchFileText(runId, relPath, artifacts, signal);
    const parsed = safeParseJson(text);
    if (!parsed.ok) {
      return { ok: false, error: parsed.error, rawSnippet: text.slice(0, 2000) };
    }
    return { ok: true, value: parsed.value as T, rawText: text };
  } catch (e) {
    return { ok: false, error: toErrorMessage(e) };
  }
}

/* ------------------------------ Panel computations ------------------------------ */

function extractZ(params: unknown): number | undefined {
  if (!params || typeof params !== "object" || Array.isArray(params)) return undefined;
  const p = params as any;

  // Direct scalar keys (FR-9.1: allow z_norm)
  const zn = safeNumber(p.z_norm);
  if (zn !== undefined) return zn;

  const z = safeNumber(p.z);
  if (z !== undefined) return z;

  const z0 = safeNumber(p.z0);
  if (z0 !== undefined) return z0;

  // Vector-like keys
  const vecZ = (v: any) => (Array.isArray(v) && v.length >= 3 ? safeNumber(v[2]) : undefined);

  return (
    vecZ(p.position) ??
    vecZ(p.center) ??
    vecZ(p.point) ??
    vecZ(p.axis_point) ??
    vecZ(p.axisPoint) ??
    undefined
  );
}

function computeBasisPlotsFromDiscoveredSystem(sys: DiscoveredSystem): {
  scatter: Array<{ x: number; y: number; colorKey: string; shapeKey?: number | string; label?: string }>;
  familyMass: Array<{ family: string; mass: number; count: number }>;
  warnings: string[];
} {
// FR-9.1 acceptance: from discovered_system.json alone (Design Doc FR-9.1). Repo format: images[].weight + group_info + params (io.py).
  const images = Array.isArray(sys.images) ? sys.images : [];
  const warnings: string[] = [];

  const scatter: Array<{ x: number; y: number; colorKey: string; shapeKey?: number | string; label?: string }> = [];
  const massByFamily = new Map<string, { mass: number; count: number }>();

  for (let i = 0; i < images.length; i++) {
    const img = images[i] || {};
    const w = safeNumber((img as any).weight) ?? 0;
    const absW = Math.abs(w);

    const gi = (img as any).group_info;
    const groupInfo = gi && typeof gi === "object" && !Array.isArray(gi) ? (gi as Record<string, unknown>) : {};

    const family = safeString(groupInfo.family_name) ?? safeString(groupInfo.family) ?? safeString((img as any).type) ?? "unknown";
    const conductorId = groupInfo.conductor_id ?? undefined;

    // z position: FR-9.1 allows z or z_norm; be defensive across basis element shapes (params may vary by type).
    const z = extractZ((img as any).params);

    if (z === undefined) {
      // Some basis types may not have a single z coordinate; drop from scatter but keep for mass.
      warnings.push("Some images missing a z coordinate (no params.z/z_norm/position/center/etc); scatter omits those points.");
    }

    const entry = massByFamily.get(family) ?? { mass: 0, count: 0 };
    entry.mass += absW;
    entry.count += 1;
    massByFamily.set(family, entry);

    if (z !== undefined) {
      scatter.push({
        x: z,
        y: absW,
        colorKey: family,
        shapeKey: conductorId as any,
        label: `${family} z=${z.toFixed(4)} |w|=${absW.toExponential(3)} (type=${String((img as any).type || "")})`,
      });
    }
  }

  const familyMass = Array.from(massByFamily.entries())
    .map(([family, v]) => ({ family, mass: v.mass, count: v.count }))
    .sort((a, b) => b.mass - a.mass);

  return { scatter, familyMass, warnings: uniq(warnings).slice(0, 6) };
}

function extractGateScores(manifest?: ManifestV1 | null, gateDashboard?: GateDashboard | null, discoveryManifest?: JsonObject | null): {
  gate2_status?: string;
  structure_score?: number;
  gate3_status?: string;
  novelty_score?: number;
} {
  // FR-9.5: pull from manifest.gate OR gate_dashboard.json if present.
  const gFromDash = gateDashboard ?? null;
  if (gFromDash) {
    return {
      gate2_status: safeString(gFromDash.gate2_status),
      structure_score: safeNumber(gFromDash.structure_score ?? undefined),
      gate3_status: safeString(gFromDash.gate3_status),
      novelty_score: safeNumber(gFromDash.novelty_score ?? undefined),
    };
  }
  const g = manifest?.gate;
  if (g && typeof g === "object") {
    return {
      gate2_status: safeString((g as any).gate2_status),
      structure_score: safeNumber((g as any).structure_score),
      gate3_status: safeString((g as any).gate3_status),
      novelty_score: safeNumber((g as any).novelty_score),
    };
  }
// images_discover discovery_manifest.json has gate*_status fields but may not include scores (repo: images_discover.py).
  if (discoveryManifest && typeof discoveryManifest === "object") {
    const dm = discoveryManifest as any;
    return {
      gate2_status: safeString(dm.gate2_status),
      structure_score: safeNumber(dm.structure_score),
      gate3_status: safeString(dm.gate3_status),
      novelty_score: safeNumber(dm.novelty_score),
    };
  }
  return {};
}

/* -------------------------------- Main component ------------------------------- */

export default function Upgrades() {
  const [sp] = useSearchParams();

  // Selection control: query string ?runId= and localStorage persistence.
  // Requirement: read ?runId=; allow multiple runIds; persist selection.
  const initialFromQuery = useMemo(() => {
    const ids = sp.getAll("runId").map((x) => x.trim()).filter(Boolean);
    // accept common aliases
    const alt = sp.getAll("run_id").concat(sp.getAll("r")).map((x) => x.trim()).filter(Boolean);
    return uniq([...ids, ...alt]);
  }, [sp]);

  const [selectedRunIds, setSelectedRunIds] = useState<string[]>(() => {
    const stored = loadRunIdsFromStorage();
    return uniq([...initialFromQuery, ...stored]);
  });

  useEffect(() => {
    // If user navigates to /upgrades?runId=... we merge that into selection.
    if (initialFromQuery.length === 0) return;
    setSelectedRunIds((prev) => uniq([...initialFromQuery, ...prev]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialFromQuery.join("|")]);

  useEffect(() => {
    saveRunIdsToStorage(selectedRunIds);
  }, [selectedRunIds]);

  const [activeTab, setActiveTab] = useState<TabId>(() => {
    const s = loadActiveTab();
    if (s && (TABS as any).some((t: any) => t.id === s)) return s as TabId;
    return "basis";
  });

  useEffect(() => {
    saveActiveTab(activeTab);
  }, [activeTab]);

  const [runIdInput, setRunIdInput] = useState("");

  // Run data store
  const [store, setStore] = useState<Record<string, RunStore>>({});

  // Guard to prevent duplicate in-flight file fetches under state churn.
  const inFlight = useRef(new Set<string>());

  // Load basics (manifest + artifacts + optional upgrades payload) for each selected run.
  useEffect(() => {
    if (selectedRunIds.length === 0) return;

    const ac = new AbortController();

    const ensureRun = (runId: string) => {
      setStore((prev) => {
        if (prev[runId]) return prev;
        return {
          ...prev,
          [runId]: { runId, loadingBasics: true, files: {} },
        };
      });

      (async () => {
        try {
          setStore((prev) => ({ ...prev, [runId]: { ...(prev[runId] || { runId, files: {} }), loadingBasics: true, basicsError: undefined } }));
          const [manifest, artifacts, upgradesPayload] = await Promise.all([
            loadManifest(runId, ac.signal),
            loadArtifacts(runId, ac.signal),
            loadUpgradesPayloadOptional(runId, ac.signal),
          ]);
          setStore((prev) => ({
            ...prev,
            [runId]: {
              ...(prev[runId] || { runId, files: {} }),
              loadingBasics: false,
              basicsError: undefined,
              manifest: manifest ?? undefined,
              artifacts,
              upgradesPayload: upgradesPayload ?? undefined,
              files: (prev[runId]?.files || {}),
            },
          }));
        } catch (e) {
          setStore((prev) => ({
            ...prev,
            [runId]: {
              ...(prev[runId] || { runId, files: {} }),
              loadingBasics: false,
              basicsError: toErrorMessage(e),
            },
          }));
        }
      })();
    };

    for (const id of selectedRunIds) ensureRun(id);

    return () => ac.abort();
  }, [selectedRunIds]);

  const addRuns = () => {
    const ids = parseRunIdsInput(runIdInput);
    if (ids.length === 0) return;
    setSelectedRunIds((prev) => uniq([...prev, ...ids]));
    setRunIdInput("");
  };

  const removeRun = (runId: string) => {
    setSelectedRunIds((prev) => prev.filter((x) => x !== runId));
    setStore((prev) => {
      const next = { ...prev };
      delete next[runId];
      return next;
    });
  };

  // Lazy-load files needed per tab.
  useEffect(() => {
    if (selectedRunIds.length === 0) return;
    const ac = new AbortController();

    const needJson = async <T,>(runId: string, relPath: string) => {
      const rs = store[runId];
      if (!rs) return;
      const key = normRelPath(relPath);
      const existing = rs.files[key];
      if (existing && existing.status !== "idle") return; // already loading/loaded/error

      const inflightKey = `${runId}:${key}`;
      if (inFlight.current.has(inflightKey)) return;
      inFlight.current.add(inflightKey);

      setStore((prev) => ({
        ...prev,
        [runId]: {
          ...(prev[runId] || { runId, loadingBasics: false, files: {} }),
          files: { ...(prev[runId]?.files || {}), [key]: { status: "loading" } },
        },
      }));

      try {
        const res = await loadJsonFile<T>(runId, key, rs.artifacts, ac.signal);
        if (res.ok) {
          setStore((prev) => ({
            ...prev,
            [runId]: {
              ...(prev[runId] || { runId, loadingBasics: false, files: {} }),
              files: { ...(prev[runId]?.files || {}), [key]: { status: "ok", value: res.value as unknown, rawText: res.rawText } },
            },
          }));
        } else {
          setStore((prev) => ({
            ...prev,
            [runId]: {
              ...(prev[runId] || { runId, loadingBasics: false, files: {} }),
              files: {
                ...(prev[runId]?.files || {}),
                [key]: { status: "error", error: res.error, rawSnippet: res.rawSnippet },
              },
            },
          }));
        }
      } finally {
        inFlight.current.delete(inflightKey);
      }
    };

    // Minimal set of files by tab (FR-9 panels).
    for (const runId of selectedRunIds) {
      const rs = store[runId];
      if (!rs || rs.loadingBasics) continue;

      const artifacts = rs.artifacts || [];
      const has = (p: string) => !!findArtifact(artifacts, p);

      if (activeTab === "basis") {
        // FR-9.1 source of truth: discovered_system.json
        if (has("discovered_system.json")) void needJson<DiscoveredSystem>(runId, "discovered_system.json");
        if (has("discovery_manifest.json")) void needJson<JsonObject>(runId, "discovery_manifest.json");
      }

      if (activeTab === "conditioning") {
        // Prefer any conditioning/col_norm JSON summaries if present, else allow log sampling on demand.
        const cond = findFirstArtifactByPredicate(artifacts, (a) => a.path.toLowerCase().includes("conditioning") && a.path.toLowerCase().endsWith(".json"));
        const colNorm = findFirstArtifactByPredicate(artifacts, (a) => a.path.toLowerCase().includes("col_norm") && a.path.toLowerCase().endsWith(".json"));
        if (cond) void needJson<JsonObject>(runId, normRelPath(cond.path));
        else if (colNorm) void needJson<JsonObject>(runId, normRelPath(colNorm.path));
        else if (has("discovered_system.json")) void needJson<DiscoveredSystem>(runId, "discovered_system.json"); // contextual
      }

      if (activeTab === "collocation") {
        const coll = findFirstArtifactByPredicate(artifacts, (a) => a.path.toLowerCase().includes("collocation") && a.path.toLowerCase().endsWith(".json"));
        const oracle = findFirstArtifactByPredicate(artifacts, (a) => a.path.toLowerCase().includes("oracle") && a.path.toLowerCase().endsWith(".json"));
        if (coll) void needJson<JsonObject>(runId, normRelPath(coll.path));
        if (oracle) void needJson<JsonObject>(runId, normRelPath(oracle.path));
        if (has("discovery_manifest.json")) void needJson<JsonObject>(runId, "discovery_manifest.json");
      }

      if (activeTab === "reference") {
        const refJson = has("reference_eval.json")
          ? "reference_eval.json"
          : (findFirstArtifactByPredicate(artifacts, (a) => a.path.toLowerCase().includes("reference_eval") && a.path.toLowerCase().endsWith(".json"))?.path ?? "");
        if (refJson) void needJson<JsonObject>(runId, normRelPath(refJson));
      }

      if (activeTab === "gates" || activeTab === "audit") {
        // Prefer PlotService outputs under plots/ (repo: electrodrive/researched/plot_service.py).
        if (has("plots/gate_dashboard.json")) void needJson<GateDashboard>(runId, "plots/gate_dashboard.json");
        if (has("discovery_manifest.json")) void needJson<JsonObject>(runId, "discovery_manifest.json");
      }
    }

    return () => ac.abort();
  }, [activeTab, selectedRunIds, store]);

  /* ------------------------------ Derived views ------------------------------ */

  const selectedRuns = useMemo(() => {
    return selectedRunIds.map((id) => store[id]).filter(Boolean) as RunStore[];
  }, [selectedRunIds, store]);

  const trendRuns = useMemo(() => {
    const runs = [...selectedRuns];
    const t = (r: RunStore) => {
      const s = safeString(r.manifest?.started_at) ?? "";
      const ms = Date.parse(s);
      return Number.isFinite(ms) ? ms : Number.NaN;
    };
    runs.sort((a, b) => {
      const ta = t(a);
      const tb = t(b);
      const aOk = Number.isFinite(ta);
      const bOk = Number.isFinite(tb);
      if (aOk && bOk) return ta - tb;
      if (aOk) return -1;
      if (bOk) return 1;
      return a.runId.localeCompare(b.runId);
    });
    return runs;
  }, [selectedRuns]);

  const panelBasis = useMemo(() => {
    // For each run: compute basis plots from discovered_system.json (FR-9.1).
    const byRun: Record<string, { status: "loading" | "missing" | "error" | "ok"; error?: string; scatter?: any; familyMass?: any; warnings?: string[] }> = {};
    for (const r of selectedRuns) {
      const art = r.artifacts || [];
      if (!findArtifact(art, "discovered_system.json")) {
        byRun[r.runId] = { status: "missing", error: "discovered_system.json not found (FR-9.1 requires this for basis plots)." };
        continue;
      }
      const file = r.files["discovered_system.json"];
      if (!file || file.status === "idle" || file.status === "loading") {
        byRun[r.runId] = { status: "loading" };
        continue;
      }
      if (file.status === "error") {
        byRun[r.runId] = { status: "error", error: file.error };
        continue;
      }
      const sys = file.value as DiscoveredSystem;
      const computed = computeBasisPlotsFromDiscoveredSystem(sys);
      byRun[r.runId] = { status: "ok", scatter: computed.scatter, familyMass: computed.familyMass, warnings: computed.warnings };
    }
    return byRun;
  }, [selectedRuns]);

  const panelGates = useMemo(() => {
    const perRun: Record<string, { gate2_status?: string; structure_score?: number; gate3_status?: string; novelty_score?: number; source: string }> = {};
    for (const r of selectedRuns) {
      const gateDash = (r.files["plots/gate_dashboard.json"] && r.files["plots/gate_dashboard.json"].status === "ok")
        ? (r.files["plots/gate_dashboard.json"] as any).value as GateDashboard
        : null;

      const discMan = (r.files["discovery_manifest.json"] && r.files["discovery_manifest.json"].status === "ok")
        ? (r.files["discovery_manifest.json"] as any).value as JsonObject
        : null;

      const scores = extractGateScores(r.manifest, gateDash, discMan);
      const source = gateDash ? "plots/gate_dashboard.json" : r.manifest?.gate ? "manifest.gate" : discMan ? "discovery_manifest.json" : "n/a";
      perRun[r.runId] = { ...scores, source };
    }

    // Trend lines across selected runs (FR-9.5) — use started_at ordering where available.
    const trendGate2 = trendRuns
      .map((r, i) => ({ x: i + 1, y: perRun[r.runId]?.structure_score }))
      .filter((p) => typeof p.y === "number" && Number.isFinite(p.y as number)) as Array<{ x: number; y: number }>;

    const trendNovelty = trendRuns
      .map((r, i) => ({ x: i + 1, y: perRun[r.runId]?.novelty_score }))
      .filter((p) => typeof p.y === "number" && Number.isFinite(p.y as number)) as Array<{ x: number; y: number }>;

    return { perRun, trendGate2, trendNovelty };
  }, [selectedRuns, trendRuns]);

  const panelAudit = useMemo(() => {
    // Prefer PlotService coverage if available (plots/gate_dashboard.json includes log_coverage).
    const perRun: Record<string, { best?: CoverageComputed; from: string; raw?: GateDashboard["log_coverage"] | null }> = {};

    for (const r of selectedRuns) {
      const artifacts = r.artifacts || [];
      const filesPresent = {
        events_jsonl: !!findArtifact(artifacts, "events.jsonl") || !!findArtifact(artifacts, (r.manifest?.outputs?.events_jsonl || "")),
        evidence_log_jsonl: !!findArtifact(artifacts, "evidence_log.jsonl") || !!findArtifact(artifacts, (r.manifest?.outputs?.evidence_log_jsonl || "")),
        other: [] as string[],
      };

      const gateDashFile = r.files["plots/gate_dashboard.json"];
      if (gateDashFile && gateDashFile.status === "ok") {
        const gd = gateDashFile.value as GateDashboard;
        const cov = gd.log_coverage;
        const eventCounts = cov?.event_source_counts || {};
        const residCounts = cov?.residual_field_detection_counts || {};
        const ingested = cov?.ingested_files || [];
        const warnings: string[] = [];

        // Fix-it checklist (FR-9.6)
        if ((eventCounts as any).msg && !(eventCounts as any).event) warnings.push("Event name source is mostly 'msg' (repo JsonlLogger); tools expecting 'event' may miss events.");
        if ((residCounts as any).resid_precond && !(residCounts as any).resid) warnings.push("Residuals detected via resid_precond variants without resid; legacy tooling may miss them.");
        if (filesPresent.evidence_log_jsonl && !filesPresent.events_jsonl) warnings.push("Only evidence_log.jsonl present; tools expecting events.jsonl may miss events unless bridged (§1.4).");
        if (filesPresent.events_jsonl && filesPresent.evidence_log_jsonl) warnings.push("Both events.jsonl and evidence_log.jsonl present; ingestion should merge/deduplicate (§1.4).");

        perRun[r.runId] = {
          from: "plots/gate_dashboard.json (PlotService coverage)",
          raw: cov ?? null,
          best: {
            files_present: { ...filesPresent, other: ingested.filter((x) => x !== "events.jsonl" && x !== "evidence_log.jsonl") },
            event_name_source_counts: {
              event: Number((eventCounts as any).event || 0),
              msg: Number((eventCounts as any).msg || 0),
              message: Number((eventCounts as any).message || 0),
              parsed_message_json: Number((eventCounts as any).parsed_from_message_json || 0),
              missing: Number((eventCounts as any).unknown || 0),
            },
            residual_fields_detected: Object.keys(residCounts).filter((k) => Number((residCounts as any)[k] || 0) > 0),
            total_records: Number(cov?.total_lines_seen || 0),
            parsed_records: Number(cov?.total_records_parsed || 0),
            warnings,
          },
        };
      } else {
        // Best-effort from artifacts only (allowed by prompt when /upgrades not available).
        const base: CoverageComputed = {
          files_present: { ...filesPresent, other: [] },
          event_name_source_counts: { event: 0, msg: 0, message: 0, parsed_message_json: 0, missing: 0 },
          residual_fields_detected: [],
          total_records: 0,
          parsed_records: 0,
          warnings: [],
        };

        if (filesPresent.events_jsonl && filesPresent.evidence_log_jsonl) base.warnings.push("Both events.jsonl and evidence_log.jsonl exist; ingestion must merge/deduplicate (§1.4).");
        if (!filesPresent.events_jsonl && filesPresent.evidence_log_jsonl) base.warnings.push("Only evidence_log.jsonl exists; older tools may miss events unless bridged (§1.4).");
        if (filesPresent.events_jsonl && !filesPresent.evidence_log_jsonl) base.warnings.push("Only events.jsonl exists; legacy evidence_log consumers may miss events unless bridged (§1.4).");

        perRun[r.runId] = { from: "artifacts presence only (no coverage JSON found)", best: base, raw: null };
      }
    }

    return perRun;
  }, [selectedRuns]);

  /* ------------------------------ Panel actions ------------------------------ */

  const computeSampleCoverage = async (runId: string) => {
    // Optional enhancement: if no PlotService coverage is present, sample a manageable number of JSONL lines.
    // Must remain defensive (FR-4) and never assume logs are small.
    const r = store[runId];
    if (!r) return;

    const artifacts = r.artifacts || [];
    const ev = findArtifact(artifacts, "events.jsonl");
    const evd = findArtifact(artifacts, "evidence_log.jsonl");

    const maxSizeBytes = 2_000_000; // capped fetch to avoid freezing the browser
    const candidates: Array<{ name: string; art?: ArtifactSummary }> = [
      { name: "events.jsonl", art: ev },
      { name: "evidence_log.jsonl", art: evd },
    ];

    let combined: CoverageComputed | null = null;

    for (const c of candidates) {
      if (!c.art) continue;

      try {
        const text = await fetchFileTextCapped(runId, c.name, artifacts, maxSizeBytes);
        const cov = sampleJsonlCoverage(text, 5000);
        cov.files_present.events_jsonl = !!ev;
        cov.files_present.evidence_log_jsonl = !!evd;
        cov.files_present.other = [];
        if (!combined) combined = cov;
        else {
          // merge counts and sets
          combined.total_records += cov.total_records;
          combined.parsed_records += cov.parsed_records;
          combined.event_name_source_counts.event += cov.event_name_source_counts.event;
          combined.event_name_source_counts.msg += cov.event_name_source_counts.msg;
          combined.event_name_source_counts.message += cov.event_name_source_counts.message;
          combined.event_name_source_counts.parsed_message_json += cov.event_name_source_counts.parsed_message_json;
          combined.event_name_source_counts.missing += cov.event_name_source_counts.missing;
          combined.residual_fields_detected = uniq([...combined.residual_fields_detected, ...cov.residual_fields_detected]);
          combined.warnings = uniq([...combined.warnings, ...cov.warnings]);
        }
      } catch {
        // ignore
      }
    }

    if (!combined) {
      alert("Could not sample logs (files missing or fetch failed).");
      return;
    }

    // Store as a synthetic “coverage” object under upgradesPayload so the UI can render it without backend support.
    setStore((prev) => ({
      ...prev,
      [runId]: {
        ...(prev[runId] || r),
        upgradesPayload: { ...(prev[runId]?.upgradesPayload || {}), _client_sampled_coverage: combined },
      },
    }));
  };

  /* ------------------------------ Rendering helpers ------------------------------ */

  const renderArtifactLinkOrPath = (r: RunStore, relPath: string) => {
    const art = findArtifact(r.artifacts, relPath);
    if (art?.url) {
      return (
        <a href={art.url} target="_blank" rel="noreferrer">
          {relPath}
        </a>
      );
    }
    return <code>{relPath}</code>;
  };

  const renderPngIfAvailable = (r: RunStore, relPath: string) => {
    const art = findArtifact(r.artifacts, relPath);
    if (art?.url) {
      return (
        <div style={{ display: "grid", gap: 8 }}>
          <div style={{ fontSize: 12, color: "#6b7280" }}>{renderArtifactLinkOrPath(r, relPath)}</div>
          <img
            alt={relPath}
            src={art.url}
            style={{ maxWidth: "100%", height: "auto", borderRadius: 12, border: "1px solid #e5e7eb", background: "#f9fafb" }}
          />
        </div>
      );
    }
    return (
      <div style={{ fontSize: 12, color: "#6b7280" }}>
        PNG present but no direct URL provided by backend: <code>{relPath}</code>
      </div>
    );
  };

  /* -------------------------------- Page header -------------------------------- */

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 18 }}>Experimental Upgrades</h1>
          <div style={{ marginTop: 4, fontSize: 12, color: "#6b7280" }}>
            Design Doc §8 day-one • FR-9.1–FR-9.6 • Artifact-only + tolerant loading (FR-3) • Defensive normalization (FR-4)
          </div>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <Link to="/runs">Run Library</Link>
          <Link to="/compare">Compare</Link>
        </div>
      </div>

      <Card
        title="Select run(s)"
        subtitle="Reads ?runId= from the URL, supports multiple run IDs (trends), and persists selection in localStorage."
      >
        <div style={{ display: "grid", gap: 10 }}>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "end" }}>
            <label style={{ display: "grid", gap: 6, flex: "1 1 320px" }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>Add run IDs (comma or space separated)</div>
              <input
                value={runIdInput}
                onChange={(e) => setRunIdInput(e.target.value)}
                placeholder="e.g. 2025-12-12T10... 8f3c... runA, runB"
                style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid #e5e7eb" }}
              />
            </label>
            <SmallButton kind="primary" onClick={addRuns} disabled={!runIdInput.trim()}>
              Add
            </SmallButton>
            <SmallButton
              onClick={() => {
                setSelectedRunIds([]);
                setStore({});
              }}
              disabled={selectedRunIds.length === 0}
            >
              Clear
            </SmallButton>
          </div>

          {selectedRunIds.length === 0 ? (
            <div style={{ color: "#6b7280", fontSize: 13 }}>
              No runs selected. Add a run ID above or open <code>/upgrades?runId=&lt;ID&gt;</code>.
            </div>
          ) : (
            <div style={{ display: "grid", gap: 8 }}>
              {selectedRunIds.map((id) => {
                const r = store[id];
                const status = r?.loadingBasics ? "loading…" : r?.basicsError ? "error" : safeString(r?.manifest?.status) ?? "ok";
                return (
                  <div
                    key={id}
                    style={{
                      display: "flex",
                      justifyContent: "space-between",
                      gap: 12,
                      padding: "8px 10px",
                      border: "1px solid #e5e7eb",
                      borderRadius: 12,
                      background: "#fff",
                      alignItems: "center",
                      flexWrap: "wrap",
                    }}
                  >
                    <div style={{ display: "grid", gap: 2 }}>
                      <div style={{ fontSize: 13 }}>
                        <b>{id}</b>{" "}
                        <span style={{ fontSize: 12, color: "#6b7280" }}>
                          ({status})
                        </span>
                      </div>
                      <div style={{ fontSize: 12, color: "#6b7280" }}>
                        workflow: <b>{safeString(r?.manifest?.workflow) ?? "—"}</b> • started_at:{" "}
                        <b>{safeString(r?.manifest?.started_at) ?? "—"}</b>
                      </div>
                      {r?.basicsError ? <div style={{ fontSize: 12, color: "#b91c1c" }}>{r.basicsError}</div> : null}
                    </div>

                    <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
                      <Link to={`/runs/${encodeURIComponent(id)}`}>Dashboards</Link>
                      <Link to={`/runs/${encodeURIComponent(id)}/monitor`}>Monitor</Link>
                      <SmallButton kind="danger" onClick={() => removeRun(id)}>Remove</SmallButton>
                    </div>
                  </div>
                );
              })}
            </div>
          )}

          <div style={{ marginTop: 6 }}>
            <Tabs
              active={activeTab}
              onChange={(t) => setActiveTab(t)}
            />
          </div>
        </div>
      </Card>

      {/* ------------------------------- Panel: FR-9.1 ------------------------------- */}
      {activeTab === "basis" ? (
        <div role="tabpanel" id="panel-basis" aria-labelledby="tab-basis" style={{ display: "grid", gap: 12 }}>
          <Card
            title="FR-9.1 Basis expressivity"
            subtitle="Source of truth: discovered_system.json only (acceptance). Scatter: z vs |w| colored by family_name; Bar: per-family L1 mass + count."
          >
            <div style={{ display: "grid", gap: 12 }}>
              {selectedRuns.length === 0 ? (
                <div style={{ color: "#6b7280" }}>Select a run above.</div>
              ) : (
                selectedRuns.map((r) => {
                  const result = panelBasis[r.runId];
                  const artifacts = r.artifacts || [];
                  const hasScatterPng = !!findArtifact(artifacts, "plots/basis_scatter.png");
                  const hasMassPng = !!findArtifact(artifacts, "plots/family_mass.png");

                  return (
                    <div key={r.runId} style={{ display: "grid", gap: 10, borderTop: "1px solid #f3f4f6", paddingTop: 10 }}>
                      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                        <div style={{ fontWeight: 700 }}>{r.runId}</div>
                        <div style={{ fontSize: 12, color: "#6b7280" }}>
                          workflow: <b>{safeString(r.manifest?.workflow) ?? "—"}</b> • status: <b>{safeString(r.manifest?.status) ?? "—"}</b>
                        </div>
                      </div>

                      {result?.status === "loading" ? <div style={{ color: "#6b7280" }}>Loading discovered_system.json…</div> : null}
                      {result?.status === "missing" ? <div style={{ color: "#b91c1c", fontSize: 13 }}>{result.error}</div> : null}
                      {result?.status === "error" ? (
                        <div style={{ display: "grid", gap: 6 }}>
                          <div style={{ color: "#b91c1c", fontSize: 13 }}>Failed to parse discovered_system.json: {result.error}</div>
                          <div style={{ fontSize: 12, color: "#6b7280" }}>
                            File written by images_discover (repo: electrodrive/tools/images_discover.py).
                          </div>
                        </div>
                      ) : null}

                      {result?.status === "ok" ? (
                        <>
                          {result.warnings && result.warnings.length ? (
                            <div style={{ fontSize: 12, color: "#b45309" }}>
                              Warnings: {result.warnings.join(" • ")}
                            </div>
                          ) : null}

                          <ScatterPlot
                            title="z vs |w| (color=family_name; shape≈conductor_id) — FR-9.1"
                            points={result.scatter}
                            xLabel="z (or z_norm)"
                            yLabel="|w|"
                          />

                          <BarChart
                            title="Family mass (L1 sum |w|) — FR-9.1 (count shown above bar)"
                            data={result.familyMass.slice(0, 24).map((f: any) => ({
                              key: String(f.family),
                              value: Number(f.mass),
                              labelRight: `n=${f.count}`,
                            }))}
                            xLabel="family"
                            yLabel="L1 mass"
                          />

                          <div style={{ display: "grid", gap: 6 }}>
                            <div style={{ fontWeight: 700 }}>Top families</div>
                            <div style={{ overflowX: "auto" }}>
                              <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                                <thead>
                                  <tr style={{ textAlign: "left" }}>
                                    <th style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb" }}>family</th>
                                    <th style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb" }}>L1 mass</th>
                                    <th style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb" }}>count</th>
                                  </tr>
                                </thead>
                                <tbody>
                                  {result.familyMass.slice(0, 12).map((f: any) => (
                                    <tr key={String(f.family)}>
                                      <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>
                                        <span style={{ display: "inline-flex", alignItems: "center", gap: 8 }}>
                                          <span style={{ width: 10, height: 10, background: colorForKey(String(f.family)), display: "inline-block", borderRadius: 2 }} />
                                          <code>{String(f.family)}</code>
                                        </span>
                                      </td>
                                      <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>{Number(f.mass).toExponential(3)}</td>
                                      <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>{Number(f.count)}</td>
                                    </tr>
                                  ))}
                                </tbody>
                              </table>
                            </div>
                          </div>

                          {(hasScatterPng || hasMassPng) ? (
                            <div style={{ display: "grid", gap: 10 }}>
                              <div style={{ fontWeight: 700 }}>Precomputed plots (optional)</div>
                              <div style={{ fontSize: 12, color: "#6b7280" }}>
                                If PlotService ran, it may have produced these PNGs (repo: electrodrive/researched/plot_service.py).
                              </div>
                              {hasScatterPng ? renderPngIfAvailable(r, "plots/basis_scatter.png") : null}
                              {hasMassPng ? renderPngIfAvailable(r, "plots/family_mass.png") : null}
                            </div>
                          ) : null}
                        </>
                      ) : null}
                    </div>
                  );
                })
              )}

              <div style={{ fontSize: 12, color: "#6b7280" }}>
                Repo note: discovered_system.json images contain serialized basis elements + weights (electrodrive/images/io.py) and optional group_info (electrodrive/images/basis.py).
              </div>
            </div>
          </Card>
        </div>
      ) : null}

      {/* ------------------------------ Panel: FR-9.2 ------------------------------ */}
      {activeTab === "conditioning" ? (
        <div role="tabpanel" id="panel-conditioning" aria-labelledby="tab-conditioning" style={{ display: "grid", gap: 12 }}>
          <Card
            title="FR-9.2 Conditioning + solver behavior"
            subtitle="If telemetry exists, render it; if only min/max exists, render with “limited conditioning telemetry” warning. Otherwise show instrumentation-needed notice."
          >
            <div style={{ display: "grid", gap: 12 }}>
              {selectedRuns.length === 0 ? <div style={{ color: "#6b7280" }}>Select a run above.</div> : null}

              {selectedRuns.map((r) => {
                const artifacts = r.artifacts || [];
                const condJson = findFirstArtifactByPredicate(artifacts, (a) => a.path.toLowerCase().includes("conditioning") && a.path.toLowerCase().endsWith(".json"))
                  ?? findFirstArtifactByPredicate(artifacts, (a) => a.path.toLowerCase().includes("col_norm") && a.path.toLowerCase().endsWith(".json"));
                const condKey = condJson ? normRelPath(condJson.path) : "";
                const condFile = condJson ? r.files[condKey] : undefined;

                // Optional plot outputs
                const condPng = findFirstArtifactByPredicate(artifacts, (a) => a.path.toLowerCase().includes("conditioning") && a.path.toLowerCase().endsWith(".png"));

                // Heuristic: min/max may exist in logs as basis_operator_stats events (Design Doc FR-9.2),
                // but without a file endpoint we cannot guarantee access. We show a button to sample logs safely.
                const hasEvents = !!findArtifact(artifacts, "events.jsonl");
                const hasEvidence = !!findArtifact(artifacts, "evidence_log.jsonl");

                const renderTelemetry = () => {
                  if (!condJson) return null;
                  if (!condFile || condFile.status === "loading" || condFile.status === "idle") return <div style={{ color: "#6b7280" }}>Loading {condKey}…</div>;
                  if (condFile.status === "error") {
                    return (
                      <div style={{ display: "grid", gap: 6 }}>
                        <div style={{ color: "#b91c1c", fontSize: 13 }}>Failed to parse {condKey}: {condFile.error}</div>
                        {condFile.rawSnippet ? (
                          <pre style={{ margin: 0, padding: 10, background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, overflowX: "auto", fontSize: 12 }}>
                            {condFile.rawSnippet}
                          </pre>
                        ) : null}
                      </div>
                    );
                  }
                  const obj = condFile.value as any;

                  // Try common shapes:
                  // - histogram: {bin_edges:[...], counts:[...]}
                  // - stats series: {rounds:[...], col_norm_max:[...], col_norm_p95:[...], ...}
                  const hist = obj?.hist ?? obj?.histogram ?? null;
                  const edges = Array.isArray(hist?.bin_edges) ? hist.bin_edges.map(Number) : null;
                  const counts = Array.isArray(hist?.counts) ? hist.counts.map(Number) : null;

                  const seriesKeys = ["col_norm_max", "col_norm_median", "col_norm_p95", "col_norm_min"];
                  const rounds = Array.isArray(obj?.rounds) ? obj.rounds.map(Number) : null;

                  const series = seriesKeys
                    .map((k) => {
                      const arr = obj?.[k];
                      if (!Array.isArray(arr) || !rounds) return null;
                      const pts = rounds.map((x: number, i: number) => ({ x, y: Number(arr[i]) })).filter((p: any) => Number.isFinite(p.x) && Number.isFinite(p.y));
                      return pts.length ? { name: k, points: pts } : null;
                    })
                    .filter(Boolean) as Array<{ name: string; points: Array<{ x: number; y: number }> }>;

                  return (
                    <div style={{ display: "grid", gap: 10 }}>
                      {edges && counts && edges.length >= 2 && counts.length >= 1 ? (
                        <BarChart
                          title="col_norm histogram (if provided) — FR-9.2"
                          data={counts.slice(0, 24).map((c: number, i: number) => ({
                            key: `${i}`,
                            value: c,
                            labelRight: edges[i] !== undefined ? `${Number(edges[i]).toPrecision(2)}…` : "",
                          }))}
                          xLabel="bin"
                          yLabel="count"
                        />
                      ) : null}

                      {series.length ? (
                        <MultiLineChart
                          title="conditioning trends across rounds (if provided) — FR-9.2"
                          series={series}
                          xLabel="round"
                          yLabel="col_norm"
                        />
                      ) : null}

                      {!edges && !series.length ? (
                        <div style={{ fontSize: 12, color: "#b45309" }}>
                          Limited conditioning telemetry: JSON exists but no recognized histogram/trend structure. Per FR-9.2 acceptance, the UI should still render and flag limitations.
                        </div>
                      ) : null}

                      {condPng ? (
                        <div style={{ display: "grid", gap: 6 }}>
                          <div style={{ fontWeight: 700 }}>Precomputed image</div>
                          {renderPngIfAvailable(r, normRelPath(condPng.path))}
                        </div>
                      ) : null}
                    </div>
                  );
                };

                return (
                  <div key={r.runId} style={{ display: "grid", gap: 10, borderTop: "1px solid #f3f4f6", paddingTop: 10 }}>
                    <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                      <div style={{ fontWeight: 700 }}>{r.runId}</div>
                      <div style={{ fontSize: 12, color: "#6b7280" }}>
                        workflow: <b>{safeString(r.manifest?.workflow) ?? "—"}</b>
                      </div>
                    </div>

                    {condJson ? (
                      renderTelemetry()
                    ) : (
                      <div style={{ display: "grid", gap: 8 }}>
                        <div style={{ color: "#6b7280", fontSize: 13 }}>
                          No conditioning telemetry artifact found. Per FR-9.2 instrumentation requirements, richer telemetry must be logged/saved to unlock histograms and trends.
                        </div>
                        <div style={{ fontSize: 12, color: "#6b7280" }}>
                          Logs may still contain <code>basis_operator_stats</code> min/max (Design Doc FR-9.2); if your backend exposes file reads, add a small JSON summary artifact.
                        </div>
                        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                          <div style={{ fontSize: 12, color: "#6b7280" }}>
                            logs present: <b>{hasEvents ? "events.jsonl" : ""}{hasEvents && hasEvidence ? " + " : ""}{hasEvidence ? "evidence_log.jsonl" : ""}{!hasEvents && !hasEvidence ? "none" : ""}</b>
                          </div>
                          <SmallButton onClick={() => void computeSampleCoverage(r.runId)} title="Best-effort log sampling (capped to avoid huge reads).">
                            Sample logs (FR-4)
                          </SmallButton>
                        </div>
                      </div>
                    )}
                  </div>
                );
              })}

              <div style={{ fontSize: 12, color: "#6b7280" }}>
                Repo note: JsonlLogger schema uses <code>msg</code> (not <code>event</code>) in events.jsonl (electrodrive/utils/logging.py).
              </div>
            </div>
          </Card>
        </div>
      ) : null}

      {/* ------------------------------ Panel: FR-9.3 ------------------------------ */}
      {activeTab === "collocation" ? (
        <div role="tabpanel" id="panel-collocation" aria-labelledby="tab-collocation" style={{ display: "grid", gap: 12 }}>
          <Card
            title="FR-9.3 Collocation sampling + oracle audit"
            subtitle="Render collocation summaries if present; otherwise show instrumentation-needed notice. Surface subtract_physical comparisons when available."
          >
            <div style={{ display: "grid", gap: 12 }}>
              {selectedRuns.length === 0 ? <div style={{ color: "#6b7280" }}>Select a run above.</div> : null}

              {selectedRuns.map((r) => {
                const artifacts = r.artifacts || [];
                const coll = findFirstArtifactByPredicate(artifacts, (a) => a.path.toLowerCase().includes("collocation") && a.path.toLowerCase().endsWith(".json"));
                const oracle = findFirstArtifactByPredicate(artifacts, (a) => a.path.toLowerCase().includes("oracle") && a.path.toLowerCase().endsWith(".json"));
                const subtractFlag = (() => {
                  // images_discover logs subtract_physical in discovery_manifest.json metadata (repo).
                  const dm = r.files["discovery_manifest.json"];
                  if (dm && dm.status === "ok") return Boolean((dm.value as any)?.subtract_physical);
                  return undefined;
                })();

                const collKey = coll ? normRelPath(coll.path) : "";
                const oracleKey = oracle ? normRelPath(oracle.path) : "";

                const collFile = coll ? r.files[collKey] : undefined;
                const oracleFile = oracle ? r.files[oracleKey] : undefined;

                const renderJsonPreview = (obj: unknown) => (
                  <pre style={{ margin: 0, padding: 10, background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, overflowX: "auto", fontSize: 12 }}>
                    {JSON.stringify(obj, null, 2).slice(0, 6000)}
                  </pre>
                );

                return (
                  <div key={r.runId} style={{ display: "grid", gap: 10, borderTop: "1px solid #f3f4f6", paddingTop: 10 }}>
                    <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                      <div style={{ fontWeight: 700 }}>{r.runId}</div>
                      <div style={{ fontSize: 12, color: "#6b7280" }}>
                        workflow: <b>{safeString(r.manifest?.workflow) ?? "—"}</b> • subtract_physical: <b>{subtractFlag === undefined ? "—" : subtractFlag ? "true" : "false"}</b>
                      </div>
                    </div>

                    {!coll ? (
                      <div style={{ color: "#6b7280", fontSize: 13 }}>
                        No collocation summary artifact found. Per FR-9.3 instrumentation requirements, save a lightweight collocation summary (counts + z hist + interface distance hist) to unlock this dashboard.
                      </div>
                    ) : (
                      <div style={{ display: "grid", gap: 8 }}>
                        <div style={{ fontSize: 12, color: "#6b7280" }}>
                          Collocation summary: {renderArtifactLinkOrPath(r, collKey)}
                        </div>
                        {!collFile || collFile.status === "loading" || collFile.status === "idle" ? (
                          <div style={{ color: "#6b7280" }}>Loading {collKey}…</div>
                        ) : collFile.status === "error" ? (
                          <div style={{ color: "#b91c1c", fontSize: 13 }}>Failed to parse {collKey}: {collFile.error}</div>
                        ) : (
                          renderJsonPreview(collFile.value)
                        )}
                      </div>
                    )}

                    {!oracle ? (
                      <div style={{ color: "#6b7280", fontSize: 13 }}>
                        No oracle audit artifact found. Per FR-9.3, add optional subtract_physical comparisons to enable residual distribution summaries.
                      </div>
                    ) : (
                      <div style={{ display: "grid", gap: 8 }}>
                        <div style={{ fontSize: 12, color: "#6b7280" }}>
                          Oracle audit: {renderArtifactLinkOrPath(r, oracleKey)}
                        </div>
                        {!oracleFile || oracleFile.status === "loading" || oracleFile.status === "idle" ? (
                          <div style={{ color: "#6b7280" }}>Loading {oracleKey}…</div>
                        ) : oracleFile.status === "error" ? (
                          <div style={{ color: "#b91c1c", fontSize: 13 }}>Failed to parse {oracleKey}: {oracleFile.error}</div>
                        ) : (
                          renderJsonPreview(oracleFile.value)
                        )}
                      </div>
                    )}

                    <div style={{ fontSize: 12, color: "#6b7280" }}>
                      Repo note: images_discover exposes <code>--subtract-physical</code> and passes it into discovery as <code>subtract_physical_potential</code> (electrodrive/tools/images_discover.py).
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>
        </div>
      ) : null}

      {/* ------------------------------ Panel: FR-9.4 ------------------------------ */}
      {activeTab === "reference" ? (
        <div role="tabpanel" id="panel-reference" aria-labelledby="tab-reference" style={{ display: "grid", gap: 12 }}>
          <Card
            title="FR-9.4 Analytic reference stability"
            subtitle="If reference_eval.json and/or reference_eval_plot.png exist, render/link them; otherwise show instrumentation-needed notice."
          >
            <div style={{ display: "grid", gap: 12 }}>
              {selectedRuns.length === 0 ? <div style={{ color: "#6b7280" }}>Select a run above.</div> : null}

              {selectedRuns.map((r) => {
                const artifacts = r.artifacts || [];
                const refJson = findArtifact(artifacts, "reference_eval.json")
                  ?? findFirstArtifactByPredicate(artifacts, (a) => a.path.toLowerCase().includes("reference_eval") && a.path.toLowerCase().endsWith(".json"));
                const refPng = findArtifact(artifacts, "reference_eval_plot.png")
                  ?? findFirstArtifactByPredicate(artifacts, (a) => a.path.toLowerCase().includes("reference_eval") && a.path.toLowerCase().endsWith(".png"));

                const refKey = refJson ? normRelPath((refJson as any).path ?? "reference_eval.json") : "";
                const refFile = refJson ? r.files[refKey] : undefined;

                return (
                  <div key={r.runId} style={{ display: "grid", gap: 10, borderTop: "1px solid #f3f4f6", paddingTop: 10 }}>
                    <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                      <div style={{ fontWeight: 700 }}>{r.runId}</div>
                      <div style={{ fontSize: 12, color: "#6b7280" }}>
                        workflow: <b>{safeString(r.manifest?.workflow) ?? "—"}</b>
                      </div>
                    </div>

                    {!refJson && !refPng ? (
                      <div style={{ color: "#6b7280", fontSize: 13 }}>
                        No reference_eval artifacts found. Per FR-9.4 instrumentation requirements, emit <code>reference_eval.json</code> and optionally <code>reference_eval_plot.png</code>.
                      </div>
                    ) : null}

                    {refPng ? (
                      <div style={{ display: "grid", gap: 6 }}>
                        <div style={{ fontSize: 12, color: "#6b7280" }}>Plot: {renderArtifactLinkOrPath(r, normRelPath((refPng as any).path ?? "reference_eval_plot.png"))}</div>
                        {renderPngIfAvailable(r, normRelPath((refPng as any).path ?? "reference_eval_plot.png"))}
                      </div>
                    ) : null}

                    {refJson ? (
                      <div style={{ display: "grid", gap: 6 }}>
                        <div style={{ fontSize: 12, color: "#6b7280" }}>JSON: {renderArtifactLinkOrPath(r, refKey)}</div>
                        {!refFile || refFile.status === "loading" || refFile.status === "idle" ? (
                          <div style={{ color: "#6b7280" }}>Loading {refKey}…</div>
                        ) : refFile.status === "error" ? (
                          <div style={{ display: "grid", gap: 6 }}>
                            <div style={{ color: "#b91c1c", fontSize: 13 }}>Failed to parse {refKey}: {refFile.error}</div>
                            {refFile.rawSnippet ? (
                              <pre style={{ margin: 0, padding: 10, background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, overflowX: "auto", fontSize: 12 }}>
                                {refFile.rawSnippet}
                              </pre>
                            ) : null}
                          </div>
                        ) : (
                          <pre style={{ margin: 0, padding: 10, background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, overflowX: "auto", fontSize: 12 }}>
                            {JSON.stringify(refFile.value, null, 2).slice(0, 6000)}
                          </pre>
                        )}
                      </div>
                    ) : null}
                  </div>
                );
              })}
            </div>
          </Card>
        </div>
      ) : null}

      {/* ------------------------------ Panel: FR-9.5 ------------------------------ */}
      {activeTab === "gates" ? (
        <div role="tabpanel" id="panel-gates" aria-labelledby="tab-gates" style={{ display: "grid", gap: 12 }}>
          <Card
            title="FR-9.5 Gate + structure dashboards"
            subtitle="Show gate2/gate3 status + scores from manifest.gate or plots/gate_dashboard.json; trend lines across selected runs."
          >
            <div style={{ display: "grid", gap: 12 }}>
              {selectedRuns.length === 0 ? <div style={{ color: "#6b7280" }}>Select runs above to see trends.</div> : null}

              {selectedRuns.length >= 2 ? (
                <div style={{ display: "grid", gap: 10 }}>
                  <div style={{ fontWeight: 700 }}>Trends across selected runs</div>
                  <MultiLineChart
                    title="structure_score (gate2) and novelty_score (gate3) across selected runs — FR-9.5"
                    series={[
                      { name: "structure_score", points: panelGates.trendGate2 },
                      { name: "novelty_score", points: panelGates.trendNovelty },
                    ]}
                    xLabel="run index"
                    yLabel="score"
                  />
                  <div style={{ fontSize: 12, color: "#6b7280" }}>
                    X axis uses <code>started_at</code> ordering when available; otherwise falls back to run ID order.
                  </div>
                </div>
              ) : null}

              {selectedRuns.map((r) => {
                const scores = panelGates.perRun[r.runId];
                const artifacts = r.artifacts || [];
                const hasGateJson = !!findArtifact(artifacts, "plots/gate_dashboard.json");
                const hasGatePng = !!findArtifact(artifacts, "plots/gate_dashboard.png");
                const hasCoveragePng = !!findArtifact(artifacts, "plots/log_coverage.png");
                const hasFamilyMass = !!findArtifact(artifacts, "plots/family_mass.png");
                const hasBasisScatter = !!findArtifact(artifacts, "plots/basis_scatter.png");

                return (
                  <div key={r.runId} style={{ display: "grid", gap: 10, borderTop: "1px solid #f3f4f6", paddingTop: 10 }}>
                    <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                      <div style={{ fontWeight: 700 }}>{r.runId}</div>
                      <div style={{ fontSize: 12, color: "#6b7280" }}>
                        source: <b>{scores.source}</b>
                      </div>
                    </div>

                    <div style={{ display: "grid", gap: 6, fontSize: 13 }}>
                      <div>
                        Gate2: <b>{scores.gate2_status ?? "—"}</b> • structure_score:{" "}
                        <b>{scores.structure_score !== undefined ? scores.structure_score.toFixed(4) : "—"}</b>
                      </div>
                      <div>
                        Gate3: <b>{scores.gate3_status ?? "—"}</b> • novelty_score:{" "}
                        <b>{scores.novelty_score !== undefined ? scores.novelty_score.toFixed(4) : "—"}</b>
                      </div>
                    </div>

                    {hasGateJson ? (
                      <div style={{ fontSize: 12, color: "#6b7280" }}>
                        Gate dashboard JSON: {renderArtifactLinkOrPath(r, "plots/gate_dashboard.json")}
                      </div>
                    ) : (
                      <div style={{ fontSize: 12, color: "#b45309" }}>
                        Missing <code>plots/gate_dashboard.json</code>. PlotService can generate it (repo: electrodrive/researched/plot_service.py).
                      </div>
                    )}

                    {(hasGatePng || hasCoveragePng) ? (
                      <div style={{ display: "grid", gap: 10 }}>
                        {hasGatePng ? renderPngIfAvailable(r, "plots/gate_dashboard.png") : null}
                        {hasCoveragePng ? renderPngIfAvailable(r, "plots/log_coverage.png") : null}
                      </div>
                    ) : null}

                    {(hasFamilyMass || hasBasisScatter) ? (
                      <div style={{ display: "grid", gap: 6 }}>
                        <div style={{ fontWeight: 700 }}>Fingerprint hooks / related plots</div>
                        <div style={{ fontSize: 12, color: "#6b7280" }}>
                          These can support FR-9.5 family fingerprint panels if present:
                        </div>
                        {hasFamilyMass ? renderPngIfAvailable(r, "plots/family_mass.png") : null}
                        {hasBasisScatter ? renderPngIfAvailable(r, "plots/basis_scatter.png") : null}
                      </div>
                    ) : (
                      <div style={{ fontSize: 12, color: "#6b7280" }}>
                        No fingerprint PNGs found; if this is a discovery run, ensure PlotService ran (plots/family_mass.png, plots/basis_scatter.png).
                      </div>
                    )}
                  </div>
                );
              })}
            </div>
          </Card>
        </div>
      ) : null}

      {/* ------------------------------ Panel: FR-9.6 ------------------------------ */}
      {activeTab === "audit" ? (
        <div role="tabpanel" id="panel-audit" aria-labelledby="tab-audit" style={{ display: "grid", gap: 12 }}>
          <Card
            title="FR-9.6 Visualization + log consumer audit"
            subtitle="Show log parse coverage: event-name source, residual variants, ingested files, and warnings about schema/filename drift."
          >
            <div style={{ display: "grid", gap: 12 }}>
              {selectedRuns.length === 0 ? <div style={{ color: "#6b7280" }}>Select a run above.</div> : null}

              {selectedRuns.map((r) => {
                const artifacts = r.artifacts || [];
                const hasEvents = !!findArtifact(artifacts, "events.jsonl"); // repo JsonlLogger writes this (electrodrive/utils/logging.py).
                const hasEvidence = !!findArtifact(artifacts, "evidence_log.jsonl"); // legacy (electrodrive/viz/live_console.py).

                const audit = panelAudit[r.runId];
                const fromPayload = (r.upgradesPayload as any)?._client_sampled_coverage as CoverageComputed | undefined;

                // Prefer PlotService coverage; else show artifacts-only coverage; optionally show client-sampled.
                const cov = fromPayload ?? audit?.best;
                const covDisplay: CoverageComputed = cov ?? {
                  files_present: { events_jsonl: false, evidence_log_jsonl: false, other: [] },
                  event_name_source_counts: { event: 0, msg: 0, message: 0, parsed_message_json: 0, missing: 0 },
                  residual_fields_detected: [],
                  total_records: 0,
                  parsed_records: 0,
                  warnings: [],
                };

                const eventCounts = covDisplay.event_name_source_counts ?? {
                  event: 0,
                  msg: 0,
                  message: 0,
                  parsed_message_json: 0,
                  missing: 0,
                };
                const residuals = covDisplay.residual_fields_detected ?? [];

                const fixIts: string[] = [];
                if (eventCounts.msg > 0 && eventCounts.event === 0) fixIts.push("Records labeled via msg (not event). Update consumers or normalize (FR-4).");
                if (!residuals.includes("resid") && (residuals.includes("resid_precond") || residuals.includes("resid_true"))) fixIts.push("Only resid_precond/resid_true present; consider adding resid alias (FR-4).");
                if (hasEvidence && !hasEvents) fixIts.push("Only evidence_log.jsonl exists; bridge to events.jsonl for compatibility (§1.4).");
                if (hasEvents && hasEvidence) fixIts.push("Both log files exist; merge/dedup to avoid double counting (§1.4).");

                return (
                  <div key={r.runId} style={{ display: "grid", gap: 10, borderTop: "1px solid #f3f4f6", paddingTop: 10 }}>
                    <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                      <div style={{ fontWeight: 700 }}>{r.runId}</div>
                      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
                        <div style={{ fontSize: 12, color: "#6b7280" }}>source: <b>{audit?.from ?? "n/a"}{fromPayload ? " + client sample" : ""}</b></div>
                        <SmallButton onClick={() => void computeSampleCoverage(r.runId)} title="Optional: sample manageable JSONL logs to estimate coverage (FR-4).">
                          Sample logs
                        </SmallButton>
                      </div>
                    </div>

                    <div style={{ display: "grid", gap: 6 }}>
                      <div style={{ fontSize: 12, color: "#6b7280" }}>
                        log files present:{" "}
                        <b>
                          {hasEvents ? "events.jsonl" : "—"}
                          {hasEvents && hasEvidence ? " + " : ""}
                          {hasEvidence ? "evidence_log.jsonl" : ""}
                          {!hasEvents && !hasEvidence ? "none" : ""}
                        </b>
                      </div>

                      <div style={{ overflowX: "auto" }}>
                        <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                          <thead>
                            <tr style={{ textAlign: "left" }}>
                              <th style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb" }}>metric</th>
                              <th style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb" }}>value</th>
                            </tr>
                          </thead>
                          <tbody>
                            <tr>
                              <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>total_records (sampled)</td>
                              <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>{covDisplay.total_records}</td>
                            </tr>
                            <tr>
                              <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>parsed_records (sampled)</td>
                              <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>{covDisplay.parsed_records}</td>
                            </tr>
                            <tr>
                              <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>event_name_source_counts</td>
                              <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>
                                event={eventCounts.event}, msg={eventCounts.msg}, message={eventCounts.message}, parsed_message_json={eventCounts.parsed_message_json}, missing={eventCounts.missing}
                              </td>
                            </tr>
                            <tr>
                              <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>residual_fields_detected</td>
                              <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>
                                {residuals.length ? residuals.join(", ") : "—"}
                              </td>
                            </tr>
                          </tbody>
                        </table>
                      </div>

                      {fixIts.length ? (
                        <div style={{ fontSize: 12, color: "#b45309" }}>
                          Fix-it checklist (FR-9.6): {fixIts.join(" • ")}
                        </div>
                      ) : (
                        <div style={{ fontSize: 12, color: "#6b7280" }}>
                          No major compatibility warnings detected from available metadata.
                        </div>
                      )}

                      {covDisplay.warnings.length ? (
                        <div style={{ fontSize: 12, color: "#6b7280" }}>
                          Notes: {covDisplay.warnings.join(" • ")}
                        </div>
                      ) : null}

                      {audit.raw ? (
                        <details>
                          <summary style={{ cursor: "pointer", color: "#2563eb", fontSize: 12 }}>Show raw coverage object</summary>
                          <pre style={{ marginTop: 8, padding: 10, background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, overflowX: "auto", fontSize: 12 }}>
                            {JSON.stringify(audit.raw, null, 2).slice(0, 8000)}
                          </pre>
                        </details>
                      ) : null}
                    </div>

                    <div style={{ fontSize: 12, color: "#6b7280" }}>
                      Repo grounding: live_console demonstrates legacy behavior (evidence_log.jsonl + event/msg/message + resid variants).
                    </div>
                  </div>
                );
              })}
            </div>
          </Card>
        </div>
      ) : null}

      {/* Footer note */}
      <div style={{ fontSize: 12, color: "#6b7280" }}>
        Tip: if PlotService has already generated <code>plots/gate_dashboard.json</code> and <code>plots/log_coverage.png</code>,
        the audit and gates panels become richer (repo: electrodrive/researched/plot_service.py).
      </div>
    </div>
  );
}
