import { ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";

/**
 * CompareRuns (Design Doc §8 day-one #5; FR-8 cross-run comparison).
 *
 * Intent:
 * - Compare 2+ runs: overlay convergence curves when numeric series is available, metric deltas,
 *   and a "What changed?" diff of argv/inputs/spec_digest.
 *
 * Safety:
 * - Never assume artifact paths are stable; normalize and check multiple candidates.
 * - Never download huge logs unbounded; use limited/tail reads when deriving convergence from JSONL.
 */

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
  inputs?: { spec_path?: string | null; config_path?: string | null; command?: string[]; [k: string]: unknown };
  outputs?: { [k: string]: unknown };
  gate?: { gate1_status?: string | null; gate2_status?: string | null; gate3_status?: string | null; structure_score?: number | null; novelty_score?: number | null; [k: string]: unknown };
  spec_digest?: JsonObject;
  [k: string]: unknown;
};

type ArtifactSummary = { path: string; is_dir: boolean; size?: number; mtime?: number; url?: string; [k: string]: unknown };

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

type RunBundle = {
  runId: string;
  manifest?: ManifestLike | null;
  metrics?: Record<string, any> | null;
  artifacts?: ArtifactSummary[] | null;
  convergence?: { points: Array<{ iter: number; resid: number }>; note: string } | null;
  errors: string[];
};

type CompareServerResponse = {
  overlays?: unknown;
  deltas?: unknown;
  diffs?: unknown;
  [k: string]: unknown;
};

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

function findAnyArtifact(
  artifacts: ArtifactSummary[] | undefined,
  relPaths: Array<string | null | undefined>,
): ArtifactSummary | undefined {
  const want = new Set(relPaths.map((p) => normRelPath(String(p || ""))).filter(Boolean));
  if (!want.size) return undefined;
  return (artifacts || []).find((a) => want.has(normRelPath(a.path)));
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

async function fetchJsonWithFallback<T>(pathCandidates: string[], init: RequestInit & { signal?: AbortSignal } = {}): Promise<T> {
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

/**
 * Read response body with a strict cap to avoid freezing the browser on huge logs.
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
  maxBytes = 2_000_000,
): Promise<string> {
  const bases = uniq([_cachedBase ?? "", ...REST_PREFIX_CANDIDATES].map(normalizeBase).filter(Boolean));
  let lastErr: unknown = null;

  for (const base of bases) {
    for (const p of pathCandidates) {
      const path = p.startsWith("/") ? p : `/${p}`;
      const url = `${base}${path}`;

      const doFetch = async (useRange: boolean) => {
        const hdrs: Record<string, string> = {};
        if (useRange) hdrs.Range = `bytes=-${maxBytes}`;
        const resp = await fetch(url, {
          ...init,
          credentials: "same-origin",
          headers: { ...(init.headers as any), ...hdrs },
        });
        const text = await readTextLimited(resp, maxBytes);
        return { resp, text };
      };

      try {
        let attempt = await doFetch(true);

        // If the server rejects Range requests (e.g., 416), retry without Range.
        if (!attempt.resp.ok && attempt.resp.status === 416) {
          attempt = await doFetch(false);
        }

        if (!attempt.resp.ok) {
          const err = new Error(attempt.text || `HTTP ${attempt.resp.status} for ${url}`);
          (err as any).status = attempt.resp.status;
          throw err;
        }

        _cachedBase = base;
        return attempt.text;
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

/* ------------------------------ FR-4 normalization helpers ------------------------------ */

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

  const msg = normalizeEventName(rec).toLowerCase();
  return msg.includes("gmres") && (msg.includes("iter") || msg.includes("progress"));
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

    byIter.set(it, r);
  }

  return Array.from(byIter.entries()).map(([iter, resid]) => ({ iter, resid })).sort((a, b) => a.iter - b.iter);
}

/* ------------------------------ Diff helpers (FR-8) ------------------------------ */

function asLines(v: unknown): string[] {
  if (v === null || v === undefined) return ["(null)"];
  if (Array.isArray(v)) return v.map((x) => String(x));
  if (typeof v === "string") return v.split(/\r?\n/);
  try {
    return JSON.stringify(v, null, 2).split(/\r?\n/);
  } catch {
    return [String(v)];
  }
}

/**
 * Simple LCS-based line diff with +/- markers.
 * O(n*m) but constrained by maxLines guardrails.
 */
function diffLines(a: string[], b: string[], maxLines = 220): string {
  const A = a.slice(0, maxLines);
  const B = b.slice(0, maxLines);
  const n = A.length;
  const m = B.length;

  const dp: number[][] = Array.from({ length: n + 1 }, () => new Array(m + 1).fill(0));
  for (let i = n - 1; i >= 0; i--) {
    for (let j = m - 1; j >= 0; j--) {
      dp[i][j] = A[i] === B[j] ? 1 + dp[i + 1][j + 1] : Math.max(dp[i + 1][j], dp[i][j + 1]);
    }
  }

  const out: string[] = [];
  let i = 0;
  let j = 0;
  while (i < n && j < m) {
    if (A[i] === B[j]) {
      out.push(`  ${A[i]}`);
      i++; j++;
    } else if (dp[i + 1][j] >= dp[i][j + 1]) {
      out.push(`- ${A[i]}`);
      i++;
    } else {
      out.push(`+ ${B[j]}`);
      j++;
    }
    if (out.length >= 600) break;
  }
  while (i < n && out.length < 600) { out.push(`- ${A[i++]}`); }
  while (j < m && out.length < 600) { out.push(`+ ${B[j++]}`); }

  const truncated = a.length > maxLines || b.length > maxLines;
  if (truncated) out.push("… (diff truncated)");
  return out.join("\n");
}

/* ------------------------------ Charts (SVG) ------------------------------ */

function colorFor(name: string): string {
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) >>> 0;
  return `hsl(${h % 360}, 65%, 40%)`;
}

function MultiLineChart(props: {
  title?: string;
  series: Array<{ name: string; points: Array<{ x: number; y: number }> }>;
  width?: number;
  height?: number;
  yLog?: boolean;
}) {
  const width = props.width ?? 780;
  const height = props.height ?? 260;
  const pad = { l: 44, r: 16, t: 18, b: 34 };

  const all = props.series.flatMap((s) => s.points).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (all.length < 2) {
    return (
      <div style={{ height, display: "grid", placeItems: "center", border: "1px solid #e5e7eb", borderRadius: 12, color: "#6b7280" }}>
        No series available
      </div>
    );
  }

  const xMin = Math.min(...all.map((p) => p.x));
  const xMax = Math.max(...all.map((p) => p.x));

  const allPositive = all.every((p) => p.y > 0);
  const yLog = props.yLog ?? allPositive;
  const yTx = (y: number) => (yLog ? Math.log10(Math.max(y, 1e-300)) : y);

  const yVals = all.map((p) => yTx(p.y));
  let yMin = Math.min(...yVals);
  let yMax = Math.max(...yVals);
  if (!Number.isFinite(yMin) || !Number.isFinite(yMax) || yMin === yMax) { yMin -= 1; yMax += 1; }

  const innerW = width - pad.l - pad.r;
  const innerH = height - pad.t - pad.b;

  const sx = (x: number) => pad.l + ((x - xMin) / (xMax - xMin || 1)) * innerW;
  const sy = (y: number) => pad.t + (1 - (yTx(y) - yMin) / (yMax - yMin || 1)) * innerH;

  return (
    <svg width="100%" viewBox={`0 0 ${width} ${height}`} style={{ border: "1px solid #e5e7eb", borderRadius: 12, background: "#fff" }}>
      {props.title ? <text x={pad.l} y={12} fontSize="11" fill="#6b7280">{props.title}</text> : null}
      <line x1={pad.l} y1={pad.t} x2={pad.l} y2={height - pad.b} stroke="#e5e7eb" />
      <line x1={pad.l} y1={height - pad.b} x2={width - pad.r} y2={height - pad.b} stroke="#e5e7eb" />

      {props.series.map((s) => {
        const pts = s.points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y)).sort((a, b) => a.x - b.x);
        if (pts.length < 2) return null;
        const d = pts.map((p, i) => `${i === 0 ? "M" : "L"} ${sx(p.x).toFixed(2)} ${sy(p.y).toFixed(2)}`).join(" ");
        return <path key={s.name} d={d} fill="none" stroke={colorFor(s.name)} strokeWidth="1.6" />;
      })}

      {/* legend */}
      <g>
        {props.series.slice(0, 8).map((s, i) => {
          const x = width - pad.r - 190;
          const y = pad.t + 12 + i * 14;
          return (
            <g key={s.name}>
              <line x1={x} y1={y - 4} x2={x + 14} y2={y - 4} stroke={colorFor(s.name)} strokeWidth="2" />
              <text x={x + 18} y={y} fontSize="10" fill="#374151">{s.name}</text>
            </g>
          );
        })}
      </g>

      <text x={pad.l} y={height - 10} fontSize="10" fill="#6b7280">iter</text>
      <text x={pad.l - 8} y={pad.t + 8} fontSize="10" fill="#6b7280" textAnchor="end">{yLog ? "log10(resid)" : "resid"}</text>
    </svg>
  );
}

/* ------------------------------ Data loading for compare ------------------------------ */

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


async function loadMetricsIfPresent(runId: string, artifacts: ArtifactSummary[], signal?: AbortSignal): Promise<Record<string, any> | null> {
  const mArt = findAnyArtifact(artifacts, ["metrics.json", "artifacts/metrics.json"]);
  if (!mArt) return null;

  const qs = new URLSearchParams({ path: mArt.path }).toString();
  const text = await fetchTextWithFallbackLimited([`/runs/${encodeURIComponent(runId)}/files?${qs}`], { method: "GET", signal }, 1_000_000);

  let parsed: any;
  try {
    parsed = JSON.parse(text);
  } catch (e) {
    throw new Error(`metrics.json parse failed: ${toErrorMessage(e)}`);
  }

  const base = (parsed && typeof parsed === "object" && parsed.metrics && typeof parsed.metrics === "object") ? parsed.metrics : parsed;
  const out: Record<string, any> = {};
  if (base && typeof base === "object") {
    for (const [k, v] of Object.entries(base)) {
      if (typeof v === "number" || typeof v === "string" || typeof v === "boolean" || v === null) out[k] = v;
    }
  }
  return out;
}

async function bestEffortConvergence(
  runId: string,
  artifacts: ArtifactSummary[],
  signal?: AbortSignal,
): Promise<{ points: Array<{ iter: number; resid: number }>; note: string } | null> {
  // FR-8: overlay curves when numeric series is available (from logs).
  const maxBytes = 2_000_000;
  const ev = findAnyArtifact(artifacts, ["events.jsonl", "artifacts/events.jsonl"]);
  const evd = findAnyArtifact(artifacts, ["evidence_log.jsonl", "artifacts/evidence_log.jsonl"]);

  const parts: Array<Array<{ iter: number; resid: number }>> = [];

  const readIfOk = async (art?: ArtifactSummary) => {
    if (!art) return;
    const qs = new URLSearchParams({ path: art.path }).toString();
    const text = await fetchTextWithFallbackLimited([`/runs/${encodeURIComponent(runId)}/files?${qs}`], { method: "GET", signal }, maxBytes);
    parts.push(deriveConvergenceFromJsonlText(text, 5000));
  };

  await readIfOk(ev);
  await readIfOk(evd);

  const mergedByIter = new Map<number, number>();
  for (const arr of parts) {
    for (const p of arr) mergedByIter.set(p.iter, p.resid);
  }
  const merged = Array.from(mergedByIter.entries()).map(([iter, resid]) => ({ iter, resid })).sort((a, b) => a.iter - b.iter);
  if (!merged.length) return null;

  return {
    points: merged,
    note: "Derived from JSONL logs with normalization (event/msg/message; resid variants; iter variants; fields+embedded JSON).",
  };
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

/* ----------------------------------- Page ----------------------------------- */

const LS_COMPARE = "researched.compare.selection.v1";

function loadCompareIds(): string[] {
  try {
    const raw = localStorage.getItem(LS_COMPARE);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return [];
    return parsed.map((x) => String(x)).filter(Boolean);
  } catch {
    return [];
  }
}

function saveCompareIds(ids: string[]) {
  try {
    localStorage.setItem(LS_COMPARE, JSON.stringify(uniq(ids)));
  } catch {
    // ignore
  }
}

function fmtIso(s?: string | null): string {
  const t = (s || "").trim();
  return t ? t : "—";
}

function shortSha(sha?: string | null): string {
  const s = (sha || "").trim();
  return s ? s.slice(0, 10) : "—";
}

function pickKeyMetrics(metrics: Record<string, any> | null): Array<[string, any]> {
  if (!metrics) return [];
  const keys = [
    "gmres_resid",
    "bc_residual_linf",
    "dual_route_l2_boundary",
    "pde_residual_linf",
    "energy_rel_diff",
    "gpu_mem_peak_mb",
    "wall_time_s",
    "rel_l2_err",
    "max_abs_err",
    "rel_resid",
    "max_abs_weight",
  ];
  const out: Array<[string, any]> = [];
  for (const k of keys) if (k in metrics) out.push([k, metrics[k]]);
  return out;
}

function buildExportJson(bundles: RunBundle[], server?: CompareServerResponse | null): JsonObject {
  return {
    runIds: bundles.map((b) => b.runId),
    generated_at: new Date().toISOString(),
    server_compare: server ?? null,
    runs: bundles.map((b) => ({
      runId: b.runId,
      manifest: b.manifest ?? null,
      metrics: b.metrics ?? null,
      errors: b.errors,
      convergence_points: b.convergence?.points ?? null,
      convergence_note: b.convergence?.note ?? null,
    })),
  };
}

function downloadJson(name: string, obj: unknown) {
  const text = JSON.stringify(obj, null, 2);
  const blob = new Blob([text], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

export default function CompareRuns() {
  const [sp, setSp] = useSearchParams();
  const initialFromQuery = useMemo(() => {
    const ids = sp.getAll("r").concat(sp.getAll("runId")).concat(sp.getAll("run_id")).map((x) => x.trim()).filter(Boolean);
    return uniq(ids);
  }, [sp]);

  const [runIdInput, setRunIdInput] = useState<string>(() => initialFromQuery.join(", "));
  const [selected, setSelected] = useState<string[]>(() => {
    const stored = loadCompareIds();
    return uniq([...initialFromQuery, ...stored]);
  });

  useEffect(() => {
    if (initialFromQuery.length) setSelected((prev) => uniq([...initialFromQuery, ...prev]));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [initialFromQuery.join("|")]);

  useEffect(() => {
    saveCompareIds(selected);
    const qs = new URLSearchParams();
    for (const id of selected) qs.append("r", id);
    setSp(qs, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [selected.join("|")]);

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  const [serverCompare, setServerCompare] = useState<CompareServerResponse | null>(null);
  const [bundles, setBundles] = useState<RunBundle[]>([]);
  const [loadConvergence, setLoadConvergence] = useState<boolean>(true);

  const abortRef = useRef<AbortController | null>(null);

  const applySelection = () => {
    const ids = runIdInput.split(/[,\s]+/).map((x) => x.trim()).filter(Boolean);
    setSelected(uniq(ids));
  };

  const fetchServerCompare = async (runIds: string[], signal?: AbortSignal): Promise<CompareServerResponse | null> => {
    if (runIds.length < 2) return null;

    const qs1 = new URLSearchParams({ runIds: runIds.join(",") }).toString();
    const qs2 = new URLSearchParams();
    runIds.forEach((r) => qs2.append("r", r));

    try {
      const raw = await fetchJsonWithFallback<unknown>(
        [
          `/runs/compare?${qs1}`,
          `/compare?${qs2.toString()}`,
          `/compare?${qs1}`,
        ],
        { method: "GET", signal },
      );
      if (raw && typeof raw === "object") return raw as CompareServerResponse;
      return null;
    } catch {
      return null;
    }
  };

  const loadAll = async () => {
    setErr(null);
    setBusy(true);
    setServerCompare(null);
    setBundles([]);

    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    try {
      const runIds = selected;
      if (runIds.length < 2) {
        setErr("Select at least 2 runs to compare (FR-8).");
        return;
      }

      const server = await fetchServerCompare(runIds, ac.signal);
      setServerCompare(server);

      const results: RunBundle[] = await Promise.all(
        runIds.map(async (rid) => {
          const b: RunBundle = { runId: rid, errors: [] };

          try {
            b.manifest = await loadManifest(rid, ac.signal);
          } catch (e) {
            b.errors.push(`manifest: ${toErrorMessage(e)}`);
          }

          try {
            b.artifacts = await loadArtifacts(rid, ac.signal);
          } catch (e) {
            b.errors.push(`artifacts: ${toErrorMessage(e)}`);
            b.artifacts = [];
          }

          try {
            b.metrics = b.artifacts ? await loadMetricsIfPresent(rid, b.artifacts, ac.signal) : null;
          } catch (e) {
            b.errors.push(`metrics.json: ${toErrorMessage(e)}`);
            b.metrics = null;
          }

          if (loadConvergence && b.artifacts) {
            try {
              b.convergence = await bestEffortConvergence(rid, b.artifacts, ac.signal);
            } catch (e) {
              b.errors.push(`convergence: ${toErrorMessage(e)}`);
              b.convergence = null;
            }
          }

          return b;
        }),
      );

      setBundles(results);
    } catch (e) {
      setErr(toErrorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  useEffect(() => {
    return () => abortRef.current?.abort();
  }, []);

  const overlaySeries = useMemo(() => {
    return bundles
      .filter((b) => b.convergence && b.convergence.points.length)
      .map((b) => ({
        name: b.runId,
        points: b.convergence!.points.map((p) => ({ x: p.iter, y: p.resid })),
      }));
  }, [bundles]);

  const deltaMetricKeys = useMemo(() => {
    const keys = new Set<string>();
    for (const b of bundles) {
      if (!b.metrics) continue;
      for (const k of Object.keys(b.metrics)) keys.add(k);
    }
    const preferred = ["bc_residual_linf", "dual_route_l2_boundary", "pde_residual_linf", "energy_rel_diff", "gmres_resid", "gpu_mem_peak_mb", "wall_time_s", "rel_l2_err", "max_abs_err"];
    const rest = Array.from(keys).filter((k) => !preferred.includes(k)).sort();
    return uniq([...preferred.filter((k) => keys.has(k)), ...rest]).slice(0, 24);
  }, [bundles]);

  const whatChanged = useMemo(() => {
    if (bundles.length < 2) return null;

    const base = bundles[0];
    const baseMan = base.manifest ?? {};
    const baseCmd = (baseMan.inputs?.command ?? (baseMan as any).command ?? []) as unknown;
    const baseInputs = baseMan.inputs ?? {};
    const baseSpecDigest = (baseMan.spec_digest ?? (baseMan as any).spec_digest ?? null) as unknown;

    const diffs: Array<{ against: string; argvDiff: string; inputsDiff: string; specDigestDiff: string }> = [];

    for (let i = 1; i < bundles.length; i++) {
      const cur = bundles[i];
      const man = cur.manifest ?? {};
      const cmd = (man.inputs?.command ?? (man as any).command ?? []) as unknown;
      const inputs = man.inputs ?? {};
      const specDigest = (man.spec_digest ?? (man as any).spec_digest ?? null) as unknown;

      diffs.push({
        against: cur.runId,
        argvDiff: diffLines(asLines(baseCmd), asLines(cmd)),
        inputsDiff: diffLines(asLines(baseInputs), asLines(inputs)),
        specDigestDiff: diffLines(asLines(baseSpecDigest), asLines(specDigest)),
      });
    }

    return { base: base.runId, diffs };
  }, [bundles]);

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 18 }}>Compare Runs</h1>
          <div style={{ marginTop: 4, fontSize: 12, color: "#6b7280" }}>
            Design Doc §8 day-one • FR-8 compare • FR-4 normalization safety (msg/event + resid variants)
          </div>
        </div>
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
          <Link to="/runs">Run Library</Link>
          <Link to="/upgrades">Upgrades</Link>
          <SmallButton kind="primary" onClick={() => void loadAll()} disabled={busy || selected.length < 2}>
            {busy ? "Loading…" : "Run compare"}
          </SmallButton>
        </div>
      </div>

      <Card
        title="Select runs"
        subtitle="Enter 2+ run IDs. Selection persists in localStorage and in URL query params (?r=...)."
        right={
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            <label style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 12, color: "#6b7280" }}>
              <input type="checkbox" checked={loadConvergence} onChange={(e) => setLoadConvergence(e.target.checked)} />
              load convergence series (FR-8)
            </label>
            <SmallButton onClick={() => downloadJson("researched_compare.json", buildExportJson(bundles, serverCompare))} disabled={bundles.length === 0}>
              Export JSON
            </SmallButton>
          </div>
        }
      >
        <div style={{ display: "grid", gap: 10 }}>
          <label style={{ display: "grid", gap: 6 }}>
            <div style={{ fontSize: 12, color: "#6b7280" }}>Run IDs</div>
            <input
              value={runIdInput}
              onChange={(e) => setRunIdInput(e.target.value)}
              placeholder="runA, runB, runC"
              style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid #e5e7eb" }}
            />
          </label>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
            <SmallButton kind="primary" onClick={applySelection} disabled={!runIdInput.trim()}>Apply</SmallButton>
            <SmallButton
              onClick={() => {
                setSelected([]);
                setRunIdInput("");
                setBundles([]);
                setServerCompare(null);
              }}
              disabled={selected.length === 0}
            >
              Clear
            </SmallButton>
            <div style={{ fontSize: 12, color: "#6b7280" }}>Selected: <b>{selected.length}</b></div>
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {selected.map((id) => (
              <span key={id} style={{ border: "1px solid #e5e7eb", borderRadius: 999, padding: "4px 10px", fontSize: 12, background: "#fff" }}>
                <code>{id}</code>
              </span>
            ))}
          </div>

          {err ? <div style={{ color: "#b91c1c", fontSize: 13 }}>{err}</div> : null}
        </div>
      </Card>

      <Card
        title="Metric delta table (FR-8)"
        subtitle="Client-side fallback table from manifest + metrics.json. If a server-side compare endpoint exists, it will also be shown."
      >
        {bundles.length === 0 ? (
          <div style={{ color: "#6b7280" }}>Run compare to see results.</div>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ textAlign: "left" }}>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb" }}>run</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb" }}>workflow</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb" }}>status</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb" }}>started</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb" }}>ended</th>
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb" }}>git</th>
                  {deltaMetricKeys.map((k) => (
                    <th key={k} style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb" }}>
                      {k}
                    </th>
                  ))}
                  <th style={{ padding: "6px 8px", borderBottom: "1px solid #e5e7eb" }}>gates</th>
                </tr>
              </thead>
              <tbody>
                {bundles.map((b) => {
                  const man = b.manifest ?? {};
                  const g = man.gate ?? {};
                  return (
                    <tr key={b.runId}>
                      <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>
                        <Link to={`/runs/${encodeURIComponent(b.runId)}`}>{b.runId}</Link>
                      </td>
                      <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>{String(man.workflow ?? "—")}</td>
                      <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>{String(man.status ?? "—")}</td>
                      <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>{fmtIso(man.started_at)}</td>
                      <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>{fmtIso(man.ended_at ?? null)}</td>
                      <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>
                        <code>{shortSha(man.git?.sha ?? null)}</code>
                      </td>
                      {deltaMetricKeys.map((k) => (
                        <td key={k} style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>
                          <code>{b.metrics && k in b.metrics ? String(b.metrics[k]) : "—"}</code>
                        </td>
                      ))}
                      <td style={{ padding: "6px 8px", borderBottom: "1px solid #f3f4f6" }}>
                        <div style={{ display: "grid", gap: 2 }}>
                          <div>gate2: <b>{String((g as any).gate2_status ?? "—")}</b></div>
                          <div>structure: <b>{String((g as any).structure_score ?? "—")}</b></div>
                          <div>gate3: <b>{String((g as any).gate3_status ?? "—")}</b></div>
                          <div>novelty: <b>{String((g as any).novelty_score ?? "—")}</b></div>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>

            <div style={{ marginTop: 10, fontSize: 12, color: "#6b7280" }}>
              Notes: manifests are expected to include stable blocks (git/inputs/gate/spec_digest) for comparisons; older runs may be missing fields.
            </div>
          </div>
        )}
      </Card>

      <Card
        title="Convergence overlay (FR-8)"
        subtitle="Shown only when numeric series can be derived from logs."
      >
        {overlaySeries.length >= 2 ? (
          <div style={{ display: "grid", gap: 8 }}>
            <MultiLineChart
              title="GMRES residual vs iteration (merged from events.jsonl/evidence_log.jsonl; normalization applied)"
              series={overlaySeries}
              yLog={true}
            />
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              Extraction is resilient to msg/event naming and residual/iter variant fields (including fields bag and embedded JSON messages).
            </div>
          </div>
        ) : (
          <div style={{ color: "#b45309", fontSize: 13 }}>
            Not available: fewer than two runs have a numeric convergence series (logs missing/too large, or no iter/resid telemetry).
          </div>
        )}
      </Card>

      <Card
        title="What changed? (FR-8)"
        subtitle="Textual diffs: argv diff (manifest.inputs.command) + inputs/spec_digest diffs as JSON fallback."
      >
        {!whatChanged ? (
          <div style={{ color: "#6b7280" }}>Compare at least two runs to see diffs.</div>
        ) : (
          <div style={{ display: "grid", gap: 14 }}>
            <div style={{ fontSize: 13 }}>
              Base run: <b>{whatChanged.base}</b>. Diffs are shown against base.
            </div>

            {whatChanged.diffs.map((d) => (
              <details key={d.against} style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 10, background: "#fff" }}>
                <summary style={{ cursor: "pointer", color: "#111827", fontWeight: 700 }}>
                  Diff vs {d.against}
                </summary>

                <div style={{ display: "grid", gap: 12, marginTop: 10 }}>
                  <div>
                    <div style={{ fontWeight: 700, marginBottom: 6 }}>argv diff</div>
                    <pre style={{ margin: 0, padding: 10, background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, overflowX: "auto", fontSize: 12 }}>
                      {d.argvDiff}
                    </pre>
                  </div>

                  <div>
                    <div style={{ fontWeight: 700, marginBottom: 6 }}>inputs diff</div>
                    <pre style={{ margin: 0, padding: 10, background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, overflowX: "auto", fontSize: 12 }}>
                      {d.inputsDiff}
                    </pre>
                  </div>

                  <div>
                    <div style={{ fontWeight: 700, marginBottom: 6 }}>spec_digest diff</div>
                    <pre style={{ margin: 0, padding: 10, background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, overflowX: "auto", fontSize: 12 }}>
                      {d.specDigestDiff}
                    </pre>
                    <div style={{ marginTop: 6, fontSize: 12, color: "#6b7280" }}>
                      spec_digest is a stable summary produced by SpecInspector; may be missing for older runs.
                    </div>
                  </div>
                </div>
              </details>
            ))}
          </div>
        )}
      </Card>

      <Card
        title="Server-side compare (optional)"
        subtitle="If the backend provides /compare or /runs/compare, we show the raw response."
        right={
          <SmallButton onClick={() => downloadJson("researched_compare_server.json", serverCompare)} disabled={!serverCompare}>
            Download
          </SmallButton>
        }
      >
        {!serverCompare ? (
          <div style={{ color: "#6b7280" }}>
            No server-side compare response available (fallback-only mode).
          </div>
        ) : (
          <pre style={{ margin: 0, padding: 10, background: "#f9fafb", border: "1px solid #e5e7eb", borderRadius: 10, overflowX: "auto", fontSize: 12 }}>
            {JSON.stringify(serverCompare, null, 2).slice(0, 12000)}
          </pre>
        )}
      </Card>

      {bundles.length ? (
        <Card
          title="Per-run notes"
          subtitle="Any per-run fetch/parsing issues are listed here (missing artifacts, parse failures, etc.)."
        >
          <div style={{ display: "grid", gap: 10 }}>
            {bundles.map((b) => (
              <div key={b.runId} style={{ borderTop: "1px solid #f3f4f6", paddingTop: 10 }}>
                <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
                  <div style={{ fontWeight: 700 }}>{b.runId}</div>
                  <div style={{ display: "flex", gap: 10, flexWrap: "wrap", fontSize: 12, color: "#6b7280" }}>
                    <Link to={`/runs/${encodeURIComponent(b.runId)}`}>Dashboards</Link>
                    <Link to={`/runs/${encodeURIComponent(b.runId)}/monitor`}>Monitor</Link>
                  </div>
                </div>

                {b.errors.length ? (
                  <ul style={{ margin: "6px 0 0 16px", color: "#b91c1c", fontSize: 12 }}>
                    {b.errors.slice(0, 12).map((e, i) => <li key={i}>{e}</li>)}
                  </ul>
                ) : (
                  <div style={{ marginTop: 6, fontSize: 12, color: "#065f46" }}>No issues detected.</div>
                )}

                <div style={{ marginTop: 6, fontSize: 12, color: "#6b7280" }}>
                  Key metrics:{" "}
                  {pickKeyMetrics(b.metrics).length ? (
                    pickKeyMetrics(b.metrics).map(([k, v]) => (
                      <span key={k} style={{ marginRight: 10 }}>
                        <code>{k}</code>=<code>{String(v)}</code>
                      </span>
                    ))
                  ) : (
                    <span>—</span>
                  )}
                </div>

                {b.convergence ? (
                  <div style={{ marginTop: 6, fontSize: 12, color: "#6b7280" }}>
                    convergence: <b>{b.convergence.points.length}</b> points • {b.convergence.note}
                  </div>
                ) : (
                  <div style={{ marginTop: 6, fontSize: 12, color: "#6b7280" }}>
                    convergence: — (not available)
                  </div>
                )}
              </div>
            ))}
          </div>
        </Card>
      ) : null}
    </div>
  );
}
