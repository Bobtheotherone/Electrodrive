import { ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";

/**
 * RunDashboards (Design Doc §8 day-one #4; FR-7 post-run dashboards).
 *
 * Intent:
 * - Render post-run dashboards per workflow (solve/images_discover/learn_train/fmm_suite).
 * - Be tolerant of older runs and partial artifacts (FR-3 contract + "open any historical run").
 * - Avoid heavy client-side work; prefer precomputed plots when present (PlotService / ReportService).
 *
 * Safety:
 * - Never assume artifact paths are stable; normalize and check multiple candidates.
 * - Never download huge logs unbounded; use limited/tail reads when deriving convergence from JSONL.
 */

/* ------------------------------ Types ------------------------------ */

type Workflow = "solve" | "images_discover" | "learn_train" | "fmm_suite" | (string & {});
type RunStatus = "running" | "success" | "error" | "killed" | (string & {});
type JsonObject = Record<string, unknown>;

type ManifestLike = {
  run_id?: string;
  workflow?: Workflow;
  started_at?: string;
  ended_at?: string | null;
  status?: RunStatus | string;

  git?: { sha?: string | null; branch?: string | null; dirty?: boolean | null; diff_summary?: string | null; [k: string]: unknown };
  env?: { [k: string]: unknown };
  inputs?: { spec_path?: string | null; config_path?: string | null; command?: string[]; [k: string]: unknown };
  outputs?: {
    metrics_json?: string | null;
    events_jsonl?: string | null;
    evidence_log_jsonl?: string | null;
    viz_dir?: string | null;
    plots_dir?: string | null;
    report_html?: string | null;
    [k: string]: unknown;
  };
  gate?: { [k: string]: unknown };
  spec_digest?: JsonObject;

  // Older solve manifest schema from electrodrive/cli.py (tolerate).
  planner?: JsonObject;
  versions?: JsonObject;
  device?: JsonObject;
  backend?: JsonObject;
  run_status?: string;

  [k: string]: unknown;
};

type ArtifactSummary = {
  path: string; // relative path in run_dir
  is_dir: boolean;
  size?: number;
  mtime?: number;
  url?: string; // optional direct URL if backend provides
  [k: string]: unknown;
};

type DiscoveredSystem = {
  images?: Array<{ type?: string; params?: JsonObject; group_info?: JsonObject; weight?: number; [k: string]: unknown }>;
  metadata?: JsonObject;
  system_metadata?: JsonObject;
  [k: string]: unknown;
};

type CanonicalLogRecord = {
  // Defensive: raw JSONL can vary.
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

type ConvergenceSeries = { points: Array<{ iter: number; resid: number }>; source: string; note?: string };

/* ------------------------------ Path helpers ------------------------------ */

function normRelPath(p: string): string {
  return String(p || "")
    .replace(/\\/g, "/")
    .replace(/^\.\/+/, "")
    .replace(/^\/+/, "")
    .trim();
}

function uniq(xs: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const x of xs) {
    const v = String(x || "").trim();
    if (!v) continue;
    if (seen.has(v)) continue;
    seen.add(v);
    out.push(v);
  }
  return out;
}

/* ------------------------------ API wiring (adaptive) ------------------------------ */

const DEFAULT_REST_PREFIX = "/api";
const REST_PREFIX_CANDIDATES = uniq([
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "",
  "/api/v1",
  DEFAULT_REST_PREFIX,
]).map((s) => String(s || "").trim()).filter(Boolean);

let _cachedBase: string | null = null;

function normalizeBase(base: string): string {
  const b = (base || "").trim();
  if (!b) return "";
  return b.replace(/\/+$/, "");
}

async function readJson(resp: Response): Promise<unknown> {
  const ct = (resp.headers.get("content-type") || "").toLowerCase();
  if (resp.status === 204) return null;
  if (ct.includes("application/json")) {
    try { return await resp.json(); } catch { return null; }
  }
  try { return (await resp.text()) || null; } catch { return null; }
}

function toErrorMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  try { return JSON.stringify(e); } catch { return String(e); }
}

async function fetchJsonWithFallback<T>(
  pathCandidates: string[],
  init: RequestInit & { signal?: AbortSignal } = {},
): Promise<T> {
  const bases = uniq([_cachedBase ?? "", ...REST_PREFIX_CANDIDATES].map(normalizeBase).filter(Boolean));
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
        _cachedBase = base;
        return body as T;
      } catch (e) {
        lastErr = e;
        const st = (e as any)?.status;
        if (st === 404 || st === 405) continue;
        continue;
      }
    }
  }

  throw lastErr ?? new Error("Request failed");
}

async function fetchTextWithFallback(
  pathCandidates: string[],
  init: RequestInit & { signal?: AbortSignal } = {},
): Promise<{ text: string; contentType: string; usedUrl: string }> {
  const bases = uniq([_cachedBase ?? "", ...REST_PREFIX_CANDIDATES].map(normalizeBase).filter(Boolean));
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
        _cachedBase = base;
        return { text, contentType: ct, usedUrl: url };
      } catch (e) {
        lastErr = e;
        const st = (e as any)?.status;
        if (st === 404 || st === 405) continue;
        continue;
      }
    }
  }

  throw lastErr ?? new Error("Request failed");
}

/**
 * Read response body with a strict cap to avoid freezing the browser on huge logs.
 * If the server supports Range requests, we try to request the tail bytes.
 */
async function readTextLimited(resp: Response, maxBytes: number): Promise<string> {
  const body = resp.body;
  if (!body) {
    const t = await resp.text();
    return t.length > maxBytes ? t.slice(-maxBytes) : t;
  }

  const reader = body.getReader();
  const decoder = new TextDecoder();
  let received = 0;
  let out = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    received += value.byteLength;
    out += decoder.decode(value, { stream: true });
    if (received >= maxBytes) {
      try { await reader.cancel(); } catch {}
      break;
    }
  }
  out += decoder.decode();

  if (out.length > maxBytes) out = out.slice(-maxBytes);
  return out;
}

async function fetchTextWithFallbackLimited(
  pathCandidates: string[],
  init: RequestInit & { signal?: AbortSignal } = {},
  opts: { maxBytes: number; preferTail?: boolean } = { maxBytes: 2_000_000, preferTail: true },
): Promise<{ text: string; contentType: string; usedUrl: string }> {
  const bases = uniq([_cachedBase ?? "", ...REST_PREFIX_CANDIDATES].map(normalizeBase).filter(Boolean));
  let lastErr: unknown = null;

  for (const base of bases) {
    for (const p of pathCandidates) {
      const path = p.startsWith("/") ? p : `/${p}`;
      const url = `${base}${path}`;

      const doFetch = async (useRange: boolean) => {
        const hdrs: Record<string, string> = {};
        if (useRange && opts.preferTail) hdrs.Range = `bytes=-${opts.maxBytes}`;
        const resp = await fetch(url, {
          ...init,
          credentials: "same-origin",
          headers: { ...(init.headers as any), ...hdrs },
        });
        const ct = (resp.headers.get("content-type") || "").toLowerCase();
        const text = await readTextLimited(resp, opts.maxBytes);
        return { resp, ct, text };
      };

      try {
        let attempt = await doFetch(true);

        // If the server rejects Range requests (e.g., 416), retry without Range.
        if (!attempt.resp.ok && opts.preferTail && attempt.resp.status === 416) {
          attempt = await doFetch(false);
        }

        if (!attempt.resp.ok) {
          const err = new Error(attempt.text || `HTTP ${attempt.resp.status} for ${url}`);
          (err as any).status = attempt.resp.status;
          throw err;
        }

        _cachedBase = base;
        return { text: attempt.text, contentType: attempt.ct, usedUrl: url };
      } catch (e) {
        lastErr = e;
        const st = (e as any)?.status;
        if (st === 404 || st === 405) continue;
        continue;
      }
    }
  }

  throw lastErr ?? new Error("Request failed");
}

function fileUrl(runId: string, relPath: string): string {
  // Default contract: GET /api/runs/{runId}/files?path=<relative_path>
  const base = normalizeBase(_cachedBase ?? "/api/v1");
  const rid = encodeURIComponent(runId);
  const p = normRelPath(relPath);
  const qs = new URLSearchParams({ path: p }).toString();
  return `${base}/runs/${rid}/files?${qs}`;
}

/* ------------------------------ Small helpers ------------------------------ */

function safeParseJson(text: string): { ok: true; value: unknown } | { ok: false; error: string } {
  const s = (text || "").trim();
  if (!s) return { ok: true, value: null };
  try { return { ok: true, value: JSON.parse(s) }; } catch (e) { return { ok: false, error: toErrorMessage(e) }; }
}

function safeNumber(v: unknown): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string" && v.trim()) {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

function firstPresent(obj: Record<string, unknown>, keys: string[]): unknown {
  for (const k of keys) {
    if (Object.prototype.hasOwnProperty.call(obj, k)) {
      const v = obj[k];
      if (v !== null && v !== undefined) return v;
    }
  }
  return undefined;
}

function parseEmbeddedJsonFromMessage(rec: CanonicalLogRecord): Record<string, unknown> | null {
  const raw =
    typeof rec.msg === "string" ? rec.msg :
    typeof rec.message === "string" ? rec.message :
    "";
  const s = raw.trim();
  if (!(s.startsWith("{") && s.endsWith("}"))) return null;
  try {
    const obj = JSON.parse(s);
    if (obj && typeof obj === "object" && !Array.isArray(obj)) return obj as Record<string, unknown>;
  } catch {
    // ignore
  }
  return null;
}

function firstPresentDeep(rec: CanonicalLogRecord, keys: string[]): unknown {
  const direct = firstPresent(rec as any, keys);
  if (direct !== undefined) return direct;

  const fields = (rec as any).fields;
  if (fields && typeof fields === "object" && !Array.isArray(fields)) {
    const v = firstPresent(fields as Record<string, unknown>, keys);
    if (v !== undefined) return v;
  }

  const embedded = parseEmbeddedJsonFromMessage(rec);
  if (embedded) {
    const v = firstPresent(embedded, keys);
    if (v !== undefined) return v;
  }

  return undefined;
}

function normalizeEventName(rec: CanonicalLogRecord): string {
  const embedded = parseEmbeddedJsonFromMessage(rec);
  const ev = rec.event ?? (embedded ? embedded["event"] : undefined) ?? rec.msg ?? rec.message;
  return String(ev ?? "");
}

function normalizeIter(rec: CanonicalLogRecord): number | undefined {
  const it = firstPresentDeep(rec, ["iter", "iters", "step", "k"]);
  const n = safeNumber(it);
  return n !== undefined ? Math.trunc(n) : undefined;
}

function normalizeResid(rec: CanonicalLogRecord): number | undefined {
  const r = firstPresentDeep(rec, ["resid", "resid_precond", "resid_true", "resid_precond_l2", "resid_true_l2"]);
  return safeNumber(r);
}

function looksLikeGmresProgress(rec: CanonicalLogRecord): boolean {
  const it = normalizeIter(rec);
  const r = normalizeResid(rec);

  // If explicit iter+resid telemetry exists, accept regardless of event naming.
  if (it !== undefined && r !== undefined) return true;

  // Otherwise fall back to name heuristics.
  const msg = normalizeEventName(rec).toLowerCase();
  return msg.includes("gmres") && (msg.includes("iter") || msg.includes("progress"));
}

function findAnyArtifact(
  artifacts: ArtifactSummary[] | undefined,
  relPaths: Array<string | null | undefined>,
): ArtifactSummary | undefined {
  const want = new Set(relPaths.map((p) => normRelPath(String(p || ""))).filter(Boolean));
  if (!want.size) return undefined;
  return (artifacts || []).find((a) => want.has(normRelPath(a.path)));
}

function listArtifactsByPrefix(artifacts: ArtifactSummary[] | undefined, prefix: string, ext?: string): ArtifactSummary[] {
  const pfx = normRelPath(prefix);
  const out = (artifacts || []).filter((a) => !a.is_dir && normRelPath(a.path).startsWith(pfx));
  if (!ext) return out;
  const e = ext.toLowerCase();
  return out.filter((a) => normRelPath(a.path).toLowerCase().endsWith(e));
}

function fmtIso(s?: string | null): string {
  const t = (s || "").trim();
  return t ? t : "—";
}

function extractVizFrameIndex(p: string): number | null {
  const s = normRelPath(p);
  const m = s.match(/(?:^|\/)viz_(\d+)\.png$/i);
  if (!m) return null;
  const n = Number(m[1]);
  return Number.isFinite(n) ? n : null;
}

function sortVizFrames(frames: ArtifactSummary[]): ArtifactSummary[] {
  return [...frames].sort((a, b) => {
    const ai = extractVizFrameIndex(a.path);
    const bi = extractVizFrameIndex(b.path);
    if (ai !== null && bi !== null) return ai - bi;
    if (ai !== null) return -1;
    if (bi !== null) return 1;
    return normRelPath(a.path).localeCompare(normRelPath(b.path));
  });
}

/* ------------------------------ UI components ------------------------------ */

function Card(props: { title: string; subtitle?: string; right?: ReactNode; children: ReactNode }) {
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
  const bg = kind === "primary" ? "#111827" : kind === "danger" ? "#b91c1c" : "#fff";
  const fg = kind === "primary" || kind === "danger" ? "#fff" : "#111827";
  const border = kind === "default" ? "1px solid #e5e7eb" : "1px solid transparent";
  return (
    <button
      type="button"
      title={props.title}
      onClick={props.onClick}
      disabled={props.disabled}
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

function Modal(props: { title: string; onClose: () => void; children: ReactNode }) {
  return (
    <div
      role="dialog"
      aria-modal="true"
      style={{
        position: "fixed",
        inset: 0,
        background: "rgba(0,0,0,0.35)",
        display: "grid",
        placeItems: "center",
        padding: 16,
        zIndex: 50,
      }}
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) props.onClose();
      }}
    >
      <div style={{ width: "min(1100px, 100%)", background: "#fff", borderRadius: 14, border: "1px solid #e5e7eb", boxShadow: "0 8px 24px rgba(0,0,0,0.12)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 12px", borderBottom: "1px solid #e5e7eb" }}>
          <div style={{ fontWeight: 700 }}>{props.title}</div>
          <SmallButton onClick={props.onClose}>Close</SmallButton>
        </div>
        <div style={{ padding: 12 }}>{props.children}</div>
      </div>
    </div>
  );
}

function LineChart(props: { title?: string; points: Array<{ x: number; y: number }>; width?: number; height?: number; yLog?: boolean }) {
  const width = props.width ?? 760;
  const height = props.height ?? 240;
  const pad = { l: 44, r: 14, t: 18, b: 34 };

  const pts = props.points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (pts.length < 2) {
    return (
      <div style={{ height, display: "grid", placeItems: "center", border: "1px solid #e5e7eb", borderRadius: 12, color: "#6b7280" }}>
        No convergence series available
      </div>
    );
  }

  const xMin = Math.min(...pts.map((p) => p.x));
  const xMax = Math.max(...pts.map((p) => p.x));

  const allPositive = pts.every((p) => p.y > 0);
  const yLog = props.yLog ?? allPositive;
  const yTx = (y: number) => (yLog ? Math.log10(Math.max(y, 1e-300)) : y);

  const yVals = pts.map((p) => yTx(p.y));
  let yMin = Math.min(...yVals);
  let yMax = Math.max(...yVals);
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }

  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;

  const sx = (x: number) => pad.l + ((x - xMin) / (xMax - xMin || 1)) * innerW;
  const sy = (y: number) => pad.t + (1 - (yTx(y) - yMin) / (yMax - yMin || 1)) * innerH;

  const d = pts
    .sort((a, b) => a.x - b.x)
    .map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x).toFixed(2)} ${sy(p.y).toFixed(2)}`)
    .join(" ");

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ border: "1px solid #e5e7eb", borderRadius: 12, background: "#fff" }} aria-label={props.title ?? "Convergence chart"}>
      {props.title ? <text x={pad.l} y={12} fontSize="11" fill="#6b7280">{props.title}</text> : null}
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={height - pad.b} stroke="#e5e7eb" />
      <line x1={pad.l} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} stroke="#e5e7eb" />
      <path d={d} fill="none" stroke="#111827" strokeWidth="1.6" />
      <text x={pad.l} y={height - 10} fontSize="10" fill="#6b7280">iter</text>
      <text x={pad.l - 8} y={pad.t + 8} fontSize="10" fill="#6b7280" textAnchor="end">
        {yLog ? "log10(resid)" : "resid"}
      </text>
    </svg>
  );
}

/* ------------------------------ Data extraction ------------------------------ */

async function loadManifest(runId: string, signal?: AbortSignal): Promise<ManifestLike | null> {
  const rid = encodeURIComponent(runId);
  const raw = await fetchJsonWithFallback<unknown>([`/runs/${rid}/manifest`, `/runs/${rid}`], { method: "GET", signal });
  const man = (raw && typeof raw === "object" && (raw as any).manifest) ? (raw as any).manifest : raw;
  return man && typeof man === "object" ? (man as ManifestLike) : null;
}

async function loadArtifacts(runId: string, signal?: AbortSignal): Promise<ArtifactSummary[]> {
  const rid = encodeURIComponent(runId);
  const raw = await fetchJsonWithFallback<unknown>([`/runs/${rid}/artifacts`], { method: "GET", signal });
  const items = (raw && typeof raw === "object" && Array.isArray((raw as any).artifacts)) ? (raw as any).artifacts : raw;

  return Array.isArray(items)
    ? (items as ArtifactSummary[])
        .map((a) => ({ ...a, path: normRelPath(String((a as any).path ?? "")) }))
        .filter((a) => a.path)
    : [];
}

async function loadJsonFile(
  runId: string,
  relPath: string,
  signal?: AbortSignal,
): Promise<{ ok: true; value: unknown; raw: string } | { ok: false; error: string; raw?: string }> {
  const rid = encodeURIComponent(runId);
  const p = normRelPath(relPath);
  const qs = new URLSearchParams({ path: p }).toString();
  try {
    const res = await fetchTextWithFallback([`/runs/${rid}/files?${qs}`], { method: "GET", signal });
    const parsed = safeParseJson(res.text);
    if (!parsed.ok) return { ok: false, error: parsed.error, raw: res.text };
    return { ok: true, value: parsed.value, raw: res.text };
  } catch (e) {
    return { ok: false, error: toErrorMessage(e) };
  }
}


async function loadTextFileLimited(
  runId: string,
  relPath: string,
  maxBytes: number,
  signal?: AbortSignal,
): Promise<{ ok: true; text: string } | { ok: false; error: string }> {
  const rid = encodeURIComponent(runId);
  const p = normRelPath(relPath);
  const qs = new URLSearchParams({ path: p }).toString();
  try {
    const res = await fetchTextWithFallbackLimited([`/runs/${rid}/files?${qs}`], { method: "GET", signal }, { maxBytes, preferTail: true });
    return { ok: true, text: res.text };
  } catch (e) {
    return { ok: false, error: toErrorMessage(e) };
  }
}

function extractMetrics(obj: unknown): Record<string, number | string | boolean | null> {
  if (!obj || typeof obj !== "object") return {};
  const d = obj as any;
  const m = (d.metrics && typeof d.metrics === "object" && !Array.isArray(d.metrics)) ? d.metrics : d;
  const out: Record<string, number | string | boolean | null> = {};
  if (m && typeof m === "object") {
    for (const [k, v] of Object.entries(m as Record<string, unknown>)) {
      if (typeof v === "number" || typeof v === "string" || typeof v === "boolean" || v === null) out[k] = v;
    }
  }
  return out;
}

function deriveConvergenceFromJsonlText(text: string, maxLines: number): Array<{ iter: number; resid: number }> {
  const lines = text.split(/\r?\n/);
  const byIter = new Map<number, number>();
  let parsed = 0;

  for (let i = 0; i < lines.length && parsed < maxLines; i++) {
    const ln = (lines[i] || "").trim();
    if (!ln) continue;

    let obj: unknown;
    try { obj = JSON.parse(ln); } catch { continue; }
    if (!obj || typeof obj !== "object" || Array.isArray(obj)) continue;

    parsed++;
    const rec = obj as CanonicalLogRecord;
    if (!looksLikeGmresProgress(rec)) continue;

    const it = normalizeIter(rec);
    const r = normalizeResid(rec);
    if (it === undefined || r === undefined) continue;

    // last wins if multiple updates for the same iter
    byIter.set(it, r);
  }

  return Array.from(byIter.entries())
    .map(([iter, resid]) => ({ iter, resid }))
    .sort((a, b) => a.iter - b.iter);
}

async function bestEffortConvergence(runId: string, artifacts: ArtifactSummary[], signal?: AbortSignal): Promise<ConvergenceSeries> {
  // Prefer precomputed plot (PNG); else derive from logs with strict client-side guardrails.
  const convPng = findAnyArtifact(artifacts, ["plots/convergence.png", "plots/convergence_curve.png"]);
  if (convPng) return { points: [], source: convPng.path, note: "Rendered as PNG below." };

  const maxBytes = 2_000_000; // strict browser guardrail
  const ev = findAnyArtifact(artifacts, ["events.jsonl", "artifacts/events.jsonl"]);
  const evd = findAnyArtifact(artifacts, ["evidence_log.jsonl", "artifacts/evidence_log.jsonl"]);

  const series: Array<{ iter: number; resid: number }> = [];

  const tryFile = async (art?: ArtifactSummary) => {
    if (!art) return;
    const txt = await loadTextFileLimited(runId, art.path, maxBytes, signal);
    if (!txt.ok) return;
    const pts = deriveConvergenceFromJsonlText(txt.text, 5000);
    for (const p of pts) series.push(p);
  };

  // Prefer events.jsonl but merge evidence_log.jsonl if present.
  await tryFile(ev);
  await tryFile(evd);

  // De-dupe after merge: last wins per iter.
  const byIter = new Map<number, number>();
  for (const p of series) byIter.set(p.iter, p.resid);

  const merged = Array.from(byIter.entries())
    .map(([iter, resid]) => ({ iter, resid }))
    .sort((a, b) => a.iter - b.iter);

  return {
    points: merged,
    source: merged.length ? "jsonl-derived" : "unavailable",
    note: merged.length
      ? "Derived from JSONL logs with normalization (event/msg/message; resid variants; iter variants; fields+embedded JSON)."
      : "No usable log series found or logs too large/unavailable.",
  };
}

/* ----------------------------------- Page ----------------------------------- */

export default function RunDashboards() {
  const { runId } = useParams();
  const id = String(runId || "").trim();

  const [manifest, setManifest] = useState<ManifestLike | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactSummary[] | null>(null);
  const [err, setErr] = useState<string | null>(null);

  const [metricsState, setMetricsState] = useState<{ status: "idle" | "loading" | "ok" | "missing" | "error"; data?: Record<string, any>; error?: string; raw?: string }>({ status: "idle" });
  const [discoveryState, setDiscoveryState] = useState<{ status: "idle" | "loading" | "ok" | "missing" | "error"; sys?: DiscoveredSystem; man?: JsonObject; error?: string }>({ status: "idle" });
  const [convergenceState, setConvergenceState] = useState<{ status: "idle" | "loading" | "ok" | "error"; series?: ConvergenceSeries; error?: string }>({ status: "idle" });

  const [zoomImg, setZoomImg] = useState<{ title: string; src: string } | null>(null);

  const [reportBusy, setReportBusy] = useState(false);
  const [reportMsg, setReportMsg] = useState<string | null>(null);

  const abortRef = useRef<AbortController | null>(null);

  const refreshBasics = async () => {
    if (!id) return;
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    setErr(null);
    try {
      const [m, a] = await Promise.all([loadManifest(id, ac.signal), loadArtifacts(id, ac.signal)]);
      setManifest(m);
      setArtifacts(a);
    } catch (e) {
      setErr(toErrorMessage(e));
      setManifest(null);
      setArtifacts(null);
    }
  };

  useEffect(() => {
    void refreshBasics();
    return () => abortRef.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  // Load metrics.json if present (FR-3 contract; tolerate artifacts/metrics.json too).
  useEffect(() => {
    if (!id || !artifacts) return;
    const ac = new AbortController();

    const metricsArt = findAnyArtifact(artifacts, ["metrics.json", "artifacts/metrics.json"]);
    if (!metricsArt) {
      setMetricsState({ status: "missing" });
      return () => ac.abort();
    }

    setMetricsState({ status: "loading" });
    (async () => {
      const res = await loadJsonFile(id, metricsArt.path, ac.signal);
      if (!res.ok) {
        setMetricsState({ status: "error", error: res.error, raw: res.raw });
        return;
      }
      setMetricsState({ status: "ok", data: extractMetrics(res.value) });
    })();

    return () => ac.abort();
  }, [id, artifacts]);

  // Load discovery artifacts if present (FR-7 images_discover; FR-3 tolerate artifacts/ prefix).
  useEffect(() => {
    if (!id || !artifacts) return;
    const ac = new AbortController();

    const sysArt = findAnyArtifact(artifacts, ["discovered_system.json", "artifacts/discovered_system.json"]);
    const manArt = findAnyArtifact(artifacts, ["discovery_manifest.json", "artifacts/discovery_manifest.json"]);

    if (!sysArt && !manArt) {
      setDiscoveryState({ status: "missing" });
      return () => ac.abort();
    }

    setDiscoveryState({ status: "loading" });
    (async () => {
      try {
        const [sysRes, manRes] = await Promise.all([
          sysArt ? loadJsonFile(id, sysArt.path, ac.signal) : Promise.resolve({ ok: true as const, value: null, raw: "" }),
          manArt ? loadJsonFile(id, manArt.path, ac.signal) : Promise.resolve({ ok: true as const, value: null, raw: "" }),
        ]);
        if (!sysRes.ok) throw new Error(`${sysArt?.path ?? "discovered_system.json"} parse failed: ${sysRes.error}`);
        if (!manRes.ok) throw new Error(`${manArt?.path ?? "discovery_manifest.json"} parse failed: ${manRes.error}`);

        setDiscoveryState({
          status: "ok",
          sys: (sysRes.value as any) ?? undefined,
          man: (manRes.value as any) ?? undefined,
        });
      } catch (e) {
        setDiscoveryState({ status: "error", error: toErrorMessage(e) });
      }
    })();

    return () => ac.abort();
  }, [id, artifacts]);

  // Compute convergence best-effort (FR-7 + FR-4), only if artifacts loaded.
  useEffect(() => {
    if (!id || !artifacts) return;
    const ac = new AbortController();

    setConvergenceState({ status: "loading" });
    (async () => {
      try {
        const series = await bestEffortConvergence(id, artifacts, ac.signal);
        setConvergenceState({ status: "ok", series });
      } catch (e) {
        setConvergenceState({ status: "error", error: toErrorMessage(e) });
      }
    })();

    return () => ac.abort();
  }, [id, artifacts]);

  const workflow = useMemo(() => {
    const wf = manifest?.workflow;
    if (wf && String(wf).trim()) return String(wf);

    if (artifacts) {
      if (findAnyArtifact(artifacts, ["discovered_system.json", "artifacts/discovered_system.json", "discovery_manifest.json", "artifacts/discovery_manifest.json"])) return "images_discover";
      if (findAnyArtifact(artifacts, ["metrics.jsonl", "artifacts/metrics.jsonl", "train_log.jsonl", "artifacts/train_log.jsonl"])) return "learn_train";
      if (listArtifactsByPrefix(artifacts, "viz/", ".png").length) return "solve";
    }

    return "unknown";
  }, [manifest, artifacts]);

  const plots = useMemo(() => {
    const ps = artifacts ? listArtifactsByPrefix(artifacts, "plots/", ".png") : [];
    return [...ps].sort((a, b) => normRelPath(a.path).localeCompare(normRelPath(b.path)));
  }, [artifacts]);

  const vizPngs = useMemo(() => {
    const vs = artifacts ? listArtifactsByPrefix(artifacts, "viz/", ".png") : [];
    return sortVizFrames(vs);
  }, [artifacts]);

  const reportHtml = useMemo(() => {
    if (!artifacts) return undefined;
    const fromOutputs = manifest?.outputs?.report_html ? normRelPath(String(manifest.outputs.report_html)) : "";
    return findAnyArtifact(artifacts, ["report.html", fromOutputs, "artifacts/report.html"]);
  }, [artifacts, manifest]);

  const convergencePng = useMemo(() => findAnyArtifact(artifacts || undefined, ["plots/convergence.png", "plots/convergence_curve.png"]), [artifacts]);

  const lossPlot = useMemo(() => findAnyArtifact(artifacts || undefined, ["plots/loss_curve.png", "plots/loss.png"]), [artifacts]);
  const fmmAccuracyPlot = useMemo(() => findAnyArtifact(artifacts || undefined, ["plots/fmm_accuracy.png"]), [artifacts]);
  const fmmRuntimePlot = useMemo(() => findAnyArtifact(artifacts || undefined, ["plots/fmm_runtime.png"]), [artifacts]);

  const renderImageThumb = (a: ArtifactSummary) => {
    const src = a.url || fileUrl(id, a.path);
    return (
      <button
        key={a.path}
        type="button"
        onClick={() => setZoomImg({ title: a.path, src })}
        style={{
          border: "1px solid #e5e7eb",
          borderRadius: 12,
          background: "#fff",
          padding: 8,
          cursor: "pointer",
          textAlign: "left",
        }}
        title="Click to zoom"
      >
        <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 6 }}>{a.path}</div>
        <img
          alt={a.path}
          src={src}
          style={{ width: "100%", height: 120, objectFit: "cover", borderRadius: 10, background: "#f9fafb" }}
          onError={(e) => {
            // If backend can't serve image bytes at files endpoint, keep UI resilient.
            (e.currentTarget as HTMLImageElement).style.opacity = "0.25";
          }}
        />
      </button>
    );
  };

  const regenerateReport = async () => {
    if (!id) return;
    setReportBusy(true);
    setReportMsg(null);

    try {
      const rid = encodeURIComponent(id);

      await fetchJsonWithFallback<unknown>(
        [
          `/runs/${rid}/report`,
          `/runs/${rid}/report/generate`,
          `/runs/${rid}/report/regenerate`,
        ],
        { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ pdf: false }) },
      );

      setReportMsg("Report regenerated.");
      await refreshBasics();
    } catch (e) {
      setReportMsg(`Failed to regenerate report: ${toErrorMessage(e)}`);
    } finally {
      setReportBusy(false);
    }
  };

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 18 }}>Post-run Dashboards</h1>
          <div style={{ marginTop: 4, fontSize: 12, color: "#6b7280" }}>
            Design Doc §8 day-one • FR-7 dashboards • FR-3 artifact contract • FR-10 report • FR-4 log normalization
          </div>
        </div>
        <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
          <Link to="/runs">Run Library</Link>
          <Link to={`/runs/${encodeURIComponent(id)}/monitor`}>Monitor</Link>
          <SmallButton onClick={() => void refreshBasics()} disabled={!id}>Refresh</SmallButton>
        </div>
      </div>

      {err ? (
        <Card title="Error" subtitle="Failed to load run basics (manifest/artifacts).">
          <div style={{ color: "#b91c1c", fontSize: 13 }}>{err}</div>
        </Card>
      ) : null}

      <Card
        title="Summary"
        subtitle="Run metadata (workflow/status/timestamps/git/spec path). Tolerant of older manifest shapes."
      >
        <div style={{ display: "grid", gap: 8, fontSize: 13 }}>
          <div>
            <b>run_id:</b> {manifest?.run_id || id || "—"}
          </div>
          <div>
            <b>workflow:</b> {workflow}
          </div>
          <div>
            <b>status:</b> {String(manifest?.status ?? manifest?.run_status ?? "—")}
          </div>
          <div>
            <b>started_at:</b> {fmtIso(manifest?.started_at)}
          </div>
          <div>
            <b>ended_at:</b> {fmtIso(manifest?.ended_at ?? null)}
          </div>
          <div>
            <b>git:</b>{" "}
            {manifest?.git?.sha ? (
              <code>{String(manifest.git.sha).slice(0, 10)}</code>
            ) : (
              <span style={{ color: "#6b7280" }}>—</span>
            )}
            {manifest?.git?.branch ? <span style={{ color: "#6b7280" }}> ({String(manifest.git.branch)})</span> : null}
          </div>
          <div>
            <b>spec_path:</b>{" "}
            {manifest?.inputs?.spec_path ? <code>{String(manifest.inputs.spec_path)}</code> : <span style={{ color: "#6b7280" }}>—</span>}
          </div>
        </div>
      </Card>

      <Card
        title="Metrics"
        subtitle="FR-7: final scalar metrics from metrics.json (FR-3 contract)."
      >
        {metricsState.status === "missing" ? (
          <div style={{ color: "#b45309", fontSize: 13 }}>
            metrics.json is missing. (FR-3 says it should exist; ResearchED may create an empty stub when possible.)
          </div>
        ) : metricsState.status === "loading" ? (
          <div style={{ color: "#6b7280" }}>Loading metrics.json…</div>
        ) : metricsState.status === "error" ? (
          <div style={{ display: "grid", gap: 8 }}>
            <div style={{ color: "#b91c1c", fontSize: 13 }}>Failed to parse metrics.json: {metricsState.error}</div>
            {metricsState.raw ? (
              <pre style={{ margin: 0, padding: 10, background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, overflowX: "auto", fontSize: 12 }}>
                {metricsState.raw.slice(0, 3000)}
              </pre>
            ) : null}
          </div>
        ) : metricsState.status === "ok" ? (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ textAlign: "left" }}>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb" }}>key</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb" }}>value</th>
                </tr>
              </thead>
              <tbody>
                {Object.entries(metricsState.data || {}).slice(0, 200).map(([k, v]) => (
                  <tr key={k}>
                    <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}><code>{k}</code></td>
                    <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>
                      <code>{typeof v === "number" ? (Number.isFinite(v) ? v.toPrecision(6) : String(v)) : JSON.stringify(v)}</code>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div style={{ color: "#6b7280" }}>—</div>
        )}
      </Card>

      <Card
        title="Convergence"
        subtitle="FR-7 solve dashboard: show convergence. Prefer plots/convergence.png; else derive from logs using FR-4 normalization."
      >
        {convergencePng ? (
          <div style={{ display: "grid", gap: 8 }}>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              Using precomputed plot from plots/{normRelPath(convergencePng.path).split("/").pop()}.
            </div>
            <img
              alt="Convergence"
              src={convergencePng.url || fileUrl(id, convergencePng.path)}
              style={{ maxWidth: "100%", height: "auto", borderRadius: 12, border: "1px solid #e5e7eb", background: "#f9fafb" }}
              onError={(e) => { (e.currentTarget as HTMLImageElement).style.opacity = "0.25"; }}
            />
          </div>
        ) : convergenceState.status === "loading" ? (
          <div style={{ color: "#6b7280" }}>Deriving convergence series…</div>
        ) : convergenceState.status === "error" ? (
          <div style={{ color: "#b91c1c", fontSize: 13 }}>{convergenceState.error}</div>
        ) : convergenceState.status === "ok" ? (
          <div style={{ display: "grid", gap: 8 }}>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              source: <b>{convergenceState.series?.source}</b>{" "}
              {convergenceState.series?.note ? <span>• {convergenceState.series.note}</span> : null}
            </div>
            {convergenceState.series && convergenceState.series.points.length ? (
              <LineChart
                title="resid vs iter (log10 y when positive)"
                points={convergenceState.series.points.map((p) => ({ x: p.iter, y: p.resid }))}
                yLog={true}
              />
            ) : (
              <div style={{ color: "#b45309", fontSize: 13 }}>
                No numeric convergence series available. Ensure residual telemetry is emitted and accessible via events.jsonl/evidence_log.jsonl.
              </div>
            )}
          </div>
        ) : null}
      </Card>

      <Card
        title="Visualization gallery"
        subtitle="FR-7 solve dashboard: show viz/*.png frames (click to zoom)."
      >
        {vizPngs.length === 0 ? (
          <div style={{ color: "#6b7280" }}>No viz/*.png artifacts found.</div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10 }}>
            {vizPngs.slice(0, 60).map(renderImageThumb)}
          </div>
        )}
        <div style={{ marginTop: 10, fontSize: 12, color: "#6b7280" }}>
          Frames are sorted by numeric index when named <code>viz_####.png</code>.
        </div>
      </Card>

      <Card
        title="Generated plots"
        subtitle="plots/*.png generated by PlotService / ReportService (FR-7/FR-9/FR-10)."
      >
        {plots.length === 0 ? (
          <div style={{ color: "#6b7280" }}>No plots/*.png artifacts found.</div>
        ) : (
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 10 }}>
            {plots.slice(0, 80).map(renderImageThumb)}
          </div>
        )}
      </Card>

      <Card
        title="Workflow-specific dashboards (FR-7)"
        subtitle="Shows discovery/learn/FMM specific artifacts when present; never assumes files exist (FR-3)."
      >
        {workflow === "images_discover" ? (
          <div style={{ display: "grid", gap: 10 }}>
            <div style={{ fontSize: 13 }}>
              images_discover writes <code>discovered_system.json</code> + <code>discovery_manifest.json</code>.
            </div>
            <div>
              <Link to={`/upgrades?runId=${encodeURIComponent(id)}`}>Open Experimental Upgrades (FR-9)</Link>
            </div>

            {discoveryState.status === "missing" ? (
              <div style={{ color: "#b45309", fontSize: 13 }}>
                Discovery artifacts not found. If this is a discovery run, ensure discovered_system.json is saved to the run_dir (FR-3/FR-7).
              </div>
            ) : discoveryState.status === "loading" ? (
              <div style={{ color: "#6b7280" }}>Loading discovery artifacts…</div>
            ) : discoveryState.status === "error" ? (
              <div style={{ color: "#b91c1c", fontSize: 13 }}>{discoveryState.error}</div>
            ) : discoveryState.status === "ok" ? (
              <div style={{ display: "grid", gap: 10 }}>
                <div style={{ display: "grid", gap: 6, fontSize: 13 }}>
                  <div>
                    <b>n_images:</b> {Array.isArray(discoveryState.sys?.images) ? discoveryState.sys!.images!.length : "—"}
                  </div>
                  <div>
                    <b>numeric_status:</b>{" "}
                    {discoveryState.man && typeof discoveryState.man === "object" ? String((discoveryState.man as any).numeric_status ?? "—") : "—"}
                  </div>
                  <div>
                    <b>rel_resid:</b>{" "}
                    {discoveryState.man && typeof discoveryState.man === "object" ? String((discoveryState.man as any).rel_resid ?? "—") : "—"}
                  </div>
                  <div>
                    <b>max_abs_weight:</b>{" "}
                    {discoveryState.man && typeof discoveryState.man === "object" ? String((discoveryState.man as any).max_abs_weight ?? "—") : "—"}
                  </div>
                </div>

                <details>
                  <summary style={{ cursor: "pointer", color: "#2563eb", fontSize: 12 }}>Show discovery_manifest.json</summary>
                  <pre style={{ marginTop: 8, padding: 10, background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, overflowX: "auto", fontSize: 12 }}>
                    {JSON.stringify(discoveryState.man ?? {}, null, 2).slice(0, 6000)}
                  </pre>
                </details>
              </div>
            ) : null}
          </div>
        ) : workflow === "learn_train" ? (
          <div style={{ display: "grid", gap: 10 }}>
            <div style={{ fontSize: 13 }}>
              Learning dashboards (FR-7): expect <code>metrics.jsonl</code> and/or <code>train_log.jsonl</code>; PlotService may generate <code>plots/loss_curve.png</code>.
            </div>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              metrics.jsonl present: <b>{findAnyArtifact(artifacts || undefined, ["metrics.jsonl", "artifacts/metrics.jsonl"]) ? "yes" : "no"}</b> • train_log.jsonl present:{" "}
              <b>{findAnyArtifact(artifacts || undefined, ["train_log.jsonl", "artifacts/train_log.jsonl"]) ? "yes" : "no"}</b>
            </div>
            {lossPlot ? (
              <img
                alt="Loss curve"
                src={lossPlot.url || fileUrl(id, lossPlot.path)}
                style={{ maxWidth: "100%", height: "auto", borderRadius: 12, border: "1px solid #e5e7eb", background: "#f9fafb" }}
                onError={(e) => { (e.currentTarget as HTMLImageElement).style.opacity = "0.25"; }}
              />
            ) : (
              <div style={{ color: "#6b7280", fontSize: 13 }}>
                No loss curve plot found in <code>plots/</code>.
              </div>
            )}
          </div>
        ) : workflow === "fmm_suite" ? (
          <div style={{ display: "grid", gap: 10 }}>
            <div style={{ fontSize: 13 }}>
              FMM dashboards (FR-7): PlotService may generate <code>plots/fmm_accuracy.png</code> and <code>plots/fmm_runtime.png</code>.
            </div>
            {fmmAccuracyPlot ? (
              <img
                alt="FMM accuracy"
                src={fmmAccuracyPlot.url || fileUrl(id, fmmAccuracyPlot.path)}
                style={{ maxWidth: "100%", height: "auto", borderRadius: 12, border: "1px solid #e5e7eb", background: "#f9fafb" }}
                onError={(e) => { (e.currentTarget as HTMLImageElement).style.opacity = "0.25"; }}
              />
            ) : (
              <div style={{ color: "#6b7280", fontSize: 13 }}>No <code>plots/fmm_accuracy.png</code> found.</div>
            )}
            {fmmRuntimePlot ? (
              <img
                alt="FMM runtime"
                src={fmmRuntimePlot.url || fileUrl(id, fmmRuntimePlot.path)}
                style={{ maxWidth: "100%", height: "auto", borderRadius: 12, border: "1px solid #e5e7eb", background: "#f9fafb" }}
                onError={(e) => { (e.currentTarget as HTMLImageElement).style.opacity = "0.25"; }}
              />
            ) : (
              <div style={{ color: "#6b7280", fontSize: 13 }}>No <code>plots/fmm_runtime.png</code> found.</div>
            )}
          </div>
        ) : (
          <div style={{ color: "#6b7280" }}>
            No workflow-specific panel for workflow=<code>{workflow}</code>. This page remains safe for older/unknown runs.
          </div>
        )}
      </Card>

      <Card
        title="Spec inspector (FR-7)"
        subtitle="Show spec_digest summaries when present (SpecInspector; program/spec dashboards)."
      >
        {manifest?.spec_digest && typeof manifest.spec_digest === "object" ? (
          <pre style={{ margin: 0, padding: 10, background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, overflowX: "auto", fontSize: 12 }}>
            {JSON.stringify(manifest.spec_digest, null, 2).slice(0, 8000)}
          </pre>
        ) : (
          <div style={{ color: "#6b7280", fontSize: 13 }}>
            spec_digest not available for this run.
          </div>
        )}
      </Card>

      <Card
        title="Report (FR-10)"
        subtitle="Link to report.html if present and allow regeneration (POST)."
        right={
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <SmallButton kind="primary" onClick={() => void regenerateReport()} disabled={!id || reportBusy} title="POST to regenerate report.html + plots (FR-10)">
              {reportBusy ? "Regenerating…" : "Regenerate report"}
            </SmallButton>
            <SmallButton onClick={() => setReportMsg(null)} disabled={!reportMsg}>Clear message</SmallButton>
          </div>
        }
      >
        <div style={{ display: "grid", gap: 10 }}>
          {reportMsg ? <div style={{ fontSize: 13, color: reportMsg.startsWith("Failed") ? "#b91c1c" : "#065f46" }}>{reportMsg}</div> : null}

          {reportHtml ? (
            <div style={{ display: "grid", gap: 6 }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>report.html</div>
              <a href={reportHtml.url || fileUrl(id, reportHtml.path)} target="_blank" rel="noreferrer">
                Open report.html
              </a>
            </div>
          ) : (
            <div style={{ color: "#b45309", fontSize: 13 }}>
              report.html not found in artifacts list. (The run_dir contract may create a stub report.html.)
            </div>
          )}

          <div style={{ fontSize: 12, color: "#6b7280" }}>
            Plots directory contains {plots.length} PNG(s).
          </div>
        </div>
      </Card>

      {zoomImg ? (
        <Modal title={zoomImg.title} onClose={() => setZoomImg(null)}>
          <div style={{ display: "grid", gap: 10 }}>
            <img
              alt={zoomImg.title}
              src={zoomImg.src}
              style={{ maxWidth: "100%", height: "auto", borderRadius: 12, border: "1px solid #e5e7eb", background: "#f9fafb" }}
            />
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              If images fail to render, the backend may not serve binary files via the <code>/files</code> endpoint; prefer <code>artifact.url</code> when available.
            </div>
          </div>
        </Modal>
      ) : null}
    </div>
  );
}
