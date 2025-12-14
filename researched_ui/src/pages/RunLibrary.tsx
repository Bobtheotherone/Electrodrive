import { useEffect, useMemo, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";

/**
 * Run Library page (Design Doc §8 day-one #1; FR-3 run dir contract browsing).
 *
 * Design Doc requirements implemented here:
 * - §8 "Run Library": browse runs, open artifacts, search/filter.
 * - FR-3: surface run_dir contract signals (manifest/metrics/events/viz/artifacts/plots/report).
 *
 * Repo anchors inspected:
 * - electrodrive/researched/app.py: REST mounted at /api/v1 (include_router(..., prefix="/api/v1")).
 * - electrodrive/researched/api.py: GET /api/v1/runs returns run summaries with has_events/has_evidence/has_viz etc.
 * - electrodrive/utils/logging.py: JsonlLogger writes events.jsonl and uses "msg" field (not necessarily "event").
 * - electrodrive/viz/live_console.py: legacy evidence_log.jsonl is still tailed; prefers events.jsonl if present.
 */

type Workflow = "solve" | "images_discover" | "learn_train" | "fmm_suite" | (string & {});
type RunStatus = "running" | "success" | "error" | "killed" | (string & {});

type RunSummary = {
  run_id: string;
  workflow?: Workflow;
  status?: RunStatus | null;
  started_at?: string | null;
  ended_at?: string | null;
  path?: string;

  // Repo (electrodrive/researched/api.py _run_summary)
  has_viz?: boolean;
  has_events?: boolean;
  has_evidence?: boolean;
  has_train_log?: boolean;
  has_metrics_jsonl?: boolean;
  has_metrics?: boolean;

  // Future/back-compat
  [k: string]: unknown;
};

type ManifestV1 = {
  // Design Doc §5.1 / FR-3
  run_id: string;
  workflow: Workflow;
  started_at: string;
  ended_at?: string | null;
  status: RunStatus;

  git?: { sha?: string | null; branch?: string | null; dirty?: boolean | null; diff_summary?: string | null; [k: string]: unknown };
  inputs?: { spec_path?: string | null; config_path?: string | null; config?: unknown; command?: string[]; [k: string]: unknown };
  outputs?: { events_jsonl?: string | null; evidence_log_jsonl?: string | null; metrics_json?: string | null; viz_dir?: string | null; plots_dir?: string | null; report_html?: string | null; [k: string]: unknown };

  // Repo may add fields
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

type Filters = {
  workflow: Workflow | "all";
  status: RunStatus | "all";
  query: string;
};

const LS_FILTERS = "researched.runLibrary.filters.v1";
const LS_COMPARE = "researched.compare.selection.v1";

const DEFAULT_REST_PREFIX = "/api"; // prompt default
const REST_PREFIX_CANDIDATES = [
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "",
  "/api/v1", // repo (electrodrive/researched/app.py)
  DEFAULT_REST_PREFIX,
].map((s) => String(s || "").trim()).filter(Boolean);

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

function toErrorMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  try { return JSON.stringify(e); } catch { return String(e); }
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
          (err as any).url = url;
          throw err;
        }
        return body as T;
      } catch (e) {
        lastErr = e;
        const status = (e as any)?.status;
        if (status === 404 || status === 405) continue; // try next candidate
        // Network errors: try next candidate too.
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

function loadFilters(): Filters {
  try {
    const raw = localStorage.getItem(LS_FILTERS);
    if (!raw) return { workflow: "all", status: "all", query: "" };
    const parsed = safeParseJson(raw);
    if (!parsed.ok || !parsed.value || typeof parsed.value !== "object") return { workflow: "all", status: "all", query: "" };
    const v = parsed.value as any;
    return {
      workflow: typeof v.workflow === "string" ? (v.workflow as any) : "all",
      status: typeof v.status === "string" ? (v.status as any) : "all",
      query: typeof v.query === "string" ? v.query : "",
    };
  } catch {
    return { workflow: "all", status: "all", query: "" };
  }
}

function saveFilters(f: Filters) {
  try {
    localStorage.setItem(LS_FILTERS, JSON.stringify(f));
  } catch {
    // ignore
  }
}

function loadCompare(): string[] {
  try {
    const raw = localStorage.getItem(LS_COMPARE);
    if (!raw) return [];
    const parsed = safeParseJson(raw);
    if (!parsed.ok || !Array.isArray(parsed.value)) return [];
    return parsed.value.map((x) => String(x)).filter(Boolean);
  } catch {
    return [];
  }
}

function saveCompare(ids: string[]) {
  try {
    localStorage.setItem(LS_COMPARE, JSON.stringify(uniq(ids.map(String))));
  } catch {
    // ignore
  }
}

function truncate(s: string, n: number): string {
  const t = String(s || "");
  if (t.length <= n) return t;
  return `${t.slice(0, Math.max(0, n - 1))}…`;
}

function fmtIso(ts?: string | null): string {
  const s = (ts || "").trim();
  if (!s) return "—";
  return s;
}

function buildCompareLink(runIds: string[]): string {
  const qs = new URLSearchParams();
  for (const id of runIds) qs.append("r", id);
  return `/compare?${qs.toString()}`;
}

function Modal(props: { title: string; onClose: () => void; children: JSX.Element }) {
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
      <div style={{ width: "min(980px, 100%)", background: "#fff", borderRadius: 14, border: "1px solid #e5e7eb", boxShadow: "0 8px 24px rgba(0,0,0,0.12)" }}>
        <div style={{ display: "flex", alignItems: "center", justifyContent: "space-between", padding: "10px 12px", borderBottom: "1px solid #e5e7eb" }}>
          <div style={{ fontWeight: 700 }}>{props.title}</div>
          <button type="button" onClick={props.onClose} style={{ padding: "6px 10px", borderRadius: 10, border: "1px solid #e5e7eb", background: "#fff", cursor: "pointer" }}>
            Close
          </button>
        </div>
        <div style={{ padding: 12 }}>{props.children}</div>
      </div>
    </div>
  );
}

export default function RunLibrary() {
  const nav = useNavigate();

  const [filters, setFilters] = useState<Filters>(() => loadFilters());
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  // Compare selection persisted (Design Doc §8: Comparison view is day-one too; we store selections here for convenience).
  const [compare, setCompare] = useState<string[]>(() => loadCompare());

  // Extra row details (manifest-backed) for git/spec display (FR-3; Design Doc §5.1).
  const [rowDetails, setRowDetails] = useState<Record<string, { gitSha?: string; specPath?: string }>>({});
  const loadedRowIdsRef = useRef<Set<string>>(new Set());

  // Artifacts modal state
  const [artifactModalRun, setArtifactModalRun] = useState<string | null>(null);
  const [artifactModalItems, setArtifactModalItems] = useState<ArtifactSummary[] | null>(null);
  const [artifactModalErr, setArtifactModalErr] = useState<string | null>(null);
  const [artifactModalLoading, setArtifactModalLoading] = useState(false);

  const abortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    saveFilters(filters);
  }, [filters]);

  useEffect(() => {
    saveCompare(compare);
  }, [compare]);

  const filteredRuns = useMemo(() => {
    if (!runs) return null;
    const q = filters.query.trim().toLowerCase();
    return runs.filter((r) => {
      if (filters.workflow !== "all" && String(r.workflow || "").trim() !== filters.workflow) return false;
      if (filters.status !== "all" && String(r.status || "").trim() !== filters.status) return false;

      if (!q) return true;
      const rid = String(r.run_id || "").toLowerCase();
      const details = rowDetails[r.run_id] || {};
      const spec = String(details.specPath || "").toLowerCase();
      return rid.includes(q) || spec.includes(q);
    });
  }, [runs, filters, rowDetails]);

  const refresh = async () => {
    abortRef.current?.abort();
    const ac = new AbortController();
    abortRef.current = ac;

    setLoading(true);
    setErr(null);

    try {
      // Default contract: GET /api/runs?query=&workflow=&status=
      // Repo: GET /api/v1/runs (electrodrive/researched/app.py + api.py).
      const qs = new URLSearchParams();
      if (filters.query.trim()) qs.set("query", filters.query.trim());
      if (filters.workflow !== "all") qs.set("workflow", filters.workflow);
      if (filters.status !== "all") qs.set("status", filters.status);

      const pathCandidates = [
        `/runs?${qs.toString()}`,
        `/runs`, // fallback if backend ignores params
      ];

      const raw = await fetchJsonWithFallback<unknown>(pathCandidates, { method: "GET", signal: ac.signal });
      const data = (raw && typeof raw === "object" && Array.isArray((raw as any).runs)) ? (raw as any).runs : raw;

      if (!Array.isArray(data)) {
        setRuns([]);
      } else {
        setRuns(data as RunSummary[]);
      }
    } catch (e) {
      setErr(toErrorMessage(e));
      setRuns(null);
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Fetch manifests for visible rows to populate git sha + spec path (FR-3, Design Doc §5.1).
  // IMPORTANT: avoid self-canceling thrash by not depending on rowDetails (each setRowDetails would retrigger).
  useEffect(() => {
    if (!runs) return;

    const ac = new AbortController();
    let cancelled = false;

    const ids = runs
      .slice(0, 50)
      .map((r) => String(r.run_id || "").trim())
      .filter((id) => id && !loadedRowIdsRef.current.has(id));

    if (ids.length === 0) return;

    const concurrency = 4;
    let idx = 0;

    const runOne = async () => {
      while (idx < ids.length && !cancelled) {
        const runId = ids[idx++];
        try {
          // Default contract: GET /api/runs/{runId}/manifest
          // Repo likely returns manifest in GET /api/v1/runs/{runId} (electrodrive/researched/app.py).
          const rid = encodeURIComponent(runId);
          const raw = await fetchJsonWithFallback<unknown>(
            [`/runs/${rid}/manifest`, `/runs/${rid}`],
            { method: "GET", signal: ac.signal },
          );
          const man = (raw && typeof raw === "object" && (raw as any).manifest) ? (raw as any).manifest : raw;
          if (man && typeof man === "object") {
            const m = man as ManifestV1;
            const sha = typeof m.git?.sha === "string" ? m.git?.sha : (typeof (m as any)?.git_sha === "string" ? (m as any).git_sha : undefined);
            const spec = typeof m.inputs?.spec_path === "string" ? m.inputs?.spec_path : undefined;
            setRowDetails((prev) => ({ ...prev, [runId]: { gitSha: sha || prev[runId]?.gitSha, specPath: spec || prev[runId]?.specPath } }));
            loadedRowIdsRef.current.add(runId);
          }
        } catch {
          // Ignore per-row failures; runs table should still render.
        }
      }
    };

    void Promise.all(Array.from({ length: Math.min(concurrency, ids.length) }, () => runOne()));

    return () => {
      cancelled = true;
      ac.abort();
    };
  }, [runs]);

  const toggleCompare = (runId: string) => {
    setCompare((prev) => {
      const set = new Set(prev);
      if (set.has(runId)) set.delete(runId);
      else set.add(runId);
      return Array.from(set);
    });
  };

  const openArtifacts = async (runId: string) => {
    setArtifactModalRun(runId);
    setArtifactModalItems(null);
    setArtifactModalErr(null);
    setArtifactModalLoading(true);

    try {
      const rid = encodeURIComponent(runId);
      // Default: GET /api/runs/{runId}/artifacts
      // Repo: /api/v1/runs/{runId}/artifacts (electrodrive/researched/app.py prefix + api.py FR-3).
      const raw = await fetchJsonWithFallback<unknown>([`/runs/${rid}/artifacts`], { method: "GET" });
      const items = (raw && typeof raw === "object" && Array.isArray((raw as any).artifacts)) ? (raw as any).artifacts : raw;
      setArtifactModalItems(Array.isArray(items) ? (items as ArtifactSummary[]) : []);
    } catch (e) {
      setArtifactModalErr(toErrorMessage(e));
      setArtifactModalItems([]);
    } finally {
      setArtifactModalLoading(false);
    }
  };

  const clearFilters = () => setFilters({ workflow: "all", status: "all", query: "" });

  const compareLink = useMemo(() => buildCompareLink(compare), [compare]);

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 18 }}>Run Library</h1>
          <div style={{ marginTop: 4, fontSize: 12, color: "#6b7280" }}>
            Design Doc §8 day-one • FR-3 run artifacts contract • Repo-aware events.jsonl vs evidence_log.jsonl
          </div>
        </div>

        <div style={{ display: "flex", gap: 8, alignItems: "center", flexWrap: "wrap" }}>
          <Link to="/launch" style={{ fontSize: 13 }}>Launch a run</Link>
          <Link to={compareLink} style={{ fontSize: 13 }}>
            Compare ({compare.length})
          </Link>
          <button
            type="button"
            onClick={() => void refresh()}
            disabled={loading}
            style={{ padding: "6px 10px", borderRadius: 10, border: "1px solid #e5e7eb", background: "#fff", cursor: loading ? "not-allowed" : "pointer" }}
          >
            {loading ? "Refreshing…" : "Refresh"}
          </button>
        </div>
      </div>

      <section style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 10 }}>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "end" }}>
            <label style={{ display: "grid", gap: 6 }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>Workflow</div>
              <select
                value={filters.workflow}
                onChange={(e) => setFilters((f) => ({ ...f, workflow: e.target.value as any }))}
                style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid #e5e7eb", minWidth: 180 }}
              >
                <option value="all">All</option>
                <option value="solve">solve</option>
                <option value="images_discover">images_discover</option>
                <option value="learn_train">learn_train</option>
                <option value="fmm_suite">fmm_suite</option>
              </select>
            </label>

            <label style={{ display: "grid", gap: 6 }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>Status</div>
              <select
                value={filters.status}
                onChange={(e) => setFilters((f) => ({ ...f, status: e.target.value as any }))}
                style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid #e5e7eb", minWidth: 180 }}
              >
                <option value="all">All</option>
                <option value="running">running</option>
                <option value="success">success</option>
                <option value="error">error</option>
                <option value="killed">killed</option>
              </select>
            </label>

            <label style={{ display: "grid", gap: 6, flex: "1 1 260px" }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>Search (run id or spec path)</div>
              <input
                value={filters.query}
                onChange={(e) => setFilters((f) => ({ ...f, query: e.target.value }))}
                placeholder="e.g. plane_point or 2025-12-..."
                style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid #e5e7eb" }}
              />
            </label>

            <button
              type="button"
              onClick={clearFilters}
              style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid #e5e7eb", background: "#fff", cursor: "pointer" }}
            >
              Clear
            </button>
          </div>

          {err ? <div style={{ color: "#b91c1c", fontSize: 13 }}>{err}</div> : null}

          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ textAlign: "left" }}>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e5e7eb" }}>Run</th>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e5e7eb" }}>Workflow</th>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e5e7eb" }}>Status</th>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e5e7eb" }}>Started</th>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e5e7eb" }}>Ended</th>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e5e7eb" }}>Git</th>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e5e7eb" }}>Spec</th>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e5e7eb" }}>Artifacts</th>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e5e7eb" }}>Actions</th>
                </tr>
              </thead>

              <tbody>
                {!filteredRuns ? (
                  <tr><td colSpan={9} style={{ padding: 10, color: "#6b7280" }}>{loading ? "Loading…" : "No data."}</td></tr>
                ) : filteredRuns.length === 0 ? (
                  <tr><td colSpan={9} style={{ padding: 10, color: "#6b7280" }}>No runs match current filters.</td></tr>
                ) : (
                  filteredRuns.map((r) => {
                    const id = String(r.run_id);
                    const det = rowDetails[id] || {};
                    const shaShort = det.gitSha ? truncate(det.gitSha, 10) : "—";
                    const specShort = det.specPath ? truncate(det.specPath, 36) : "—";
                    const isInCompare = compare.includes(id);

                    // FR-3: show contract signals (events.jsonl / evidence_log.jsonl / viz / metrics)
                    const contractBits = [
                      r.has_events ? "events" : null,
                      r.has_evidence ? "evidence" : null,
                      r.has_viz ? "viz" : null,
                      r.has_metrics ? "metrics" : null,
                    ].filter(Boolean).join(", ") || "—";

                    return (
                      <tr key={id}>
                        <td style={{ padding: "8px 6px", borderBottom: "1px solid #f3f4f6" }}>
                          <Link to={`/runs/${encodeURIComponent(id)}`}>{truncate(id, 16)}</Link>
                        </td>
                        <td style={{ padding: "8px 6px", borderBottom: "1px solid #f3f4f6" }}>{String(r.workflow ?? "—")}</td>
                        <td style={{ padding: "8px 6px", borderBottom: "1px solid #f3f4f6" }}>{String(r.status ?? "—")}</td>
                        <td style={{ padding: "8px 6px", borderBottom: "1px solid #f3f4f6" }}>{fmtIso(r.started_at)}</td>
                        <td style={{ padding: "8px 6px", borderBottom: "1px solid #f3f4f6" }}>{fmtIso(r.ended_at)}</td>
                        <td style={{ padding: "8px 6px", borderBottom: "1px solid #f3f4f6" }}>{shaShort}</td>
                        <td title={det.specPath || ""} style={{ padding: "8px 6px", borderBottom: "1px solid #f3f4f6" }}>
                          {specShort}
                        </td>
                        <td style={{ padding: "8px 6px", borderBottom: "1px solid #f3f4f6", color: "#6b7280" }}>
                          {contractBits}
                        </td>
                        <td style={{ padding: "8px 6px", borderBottom: "1px solid #f3f4f6" }}>
                          <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>
                            <Link to={`/runs/${encodeURIComponent(id)}/monitor`}>Monitor</Link>
                            <button
                              type="button"
                              onClick={() => nav(`/runs/${encodeURIComponent(id)}`)}
                              style={{ border: "none", background: "transparent", padding: 0, cursor: "pointer", color: "#2563eb" }}
                            >
                              Dashboards
                            </button>
                            <button
                              type="button"
                              onClick={() => toggleCompare(id)}
                              style={{ border: "none", background: "transparent", padding: 0, cursor: "pointer", color: isInCompare ? "#b91c1c" : "#2563eb" }}
                            >
                              {isInCompare ? "Remove from Compare" : "Add to Compare"}
                            </button>
                            <button
                              type="button"
                              onClick={() => void openArtifacts(id)}
                              style={{ border: "none", background: "transparent", padding: 0, cursor: "pointer", color: "#2563eb" }}
                            >
                              Artifacts…
                            </button>
                          </div>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>

          <div style={{ fontSize: 12, color: "#6b7280" }}>
            Notes: Repo JsonlLogger emits <code>{`{"ts","level","msg"}`}</code> to <code>events.jsonl</code> (electrodrive/utils/logging.py),
            while legacy tools may tail <code>evidence_log.jsonl</code> (electrodrive/viz/live_console.py).
          </div>
        </div>
      </section>

      {artifactModalRun ? (
        <Modal
          title={`Artifacts for ${artifactModalRun}`}
          onClose={() => {
            setArtifactModalRun(null);
            setArtifactModalItems(null);
            setArtifactModalErr(null);
          }}
        >
          <div style={{ display: "grid", gap: 10 }}>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              FR-3 run_dir contract: <code>artifacts/</code>, <code>plots/</code>, <code>report.html</code>, <code>viz/*.png</code>, logs, etc.
            </div>

            {artifactModalErr ? <div style={{ color: "#b91c1c", fontSize: 13 }}>{artifactModalErr}</div> : null}

            {artifactModalLoading ? (
              <div style={{ color: "#6b7280" }}>Loading…</div>
            ) : (
              <div style={{ maxHeight: 420, overflow: "auto", border: "1px solid #e5e7eb", borderRadius: 12 }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr style={{ textAlign: "left" }}>
                      <th style={{ padding: "8px 10px", borderBottom: "1px solid #e5e7eb" }}>Path</th>
                      <th style={{ padding: "8px 10px", borderBottom: "1px solid #e5e7eb" }}>Type</th>
                      <th style={{ padding: "8px 10px", borderBottom: "1px solid #e5e7eb" }}>Size</th>
                    </tr>
                  </thead>
                  <tbody>
                    {(artifactModalItems || []).slice(0, 1000).map((a) => (
                      <tr key={a.path}>
                        <td style={{ padding: "7px 10px", borderBottom: "1px solid #f3f4f6" }}>
                          <code>{a.path}</code>
                        </td>
                        <td style={{ padding: "7px 10px", borderBottom: "1px solid #f3f4f6" }}>
                          {a.is_dir ? "dir" : "file"}
                        </td>
                        <td style={{ padding: "7px 10px", borderBottom: "1px solid #f3f4f6" }}>
                          {typeof a.size === "number" ? a.size : "—"}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}

            <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
              <button
                type="button"
                onClick={() => nav(`/runs/${encodeURIComponent(String(artifactModalRun))}`)}
                style={{ padding: "6px 10px", borderRadius: 10, border: "1px solid #e5e7eb", background: "#fff", cursor: "pointer" }}
              >
                Open Dashboards
              </button>
              <button
                type="button"
                onClick={() => nav(`/runs/${encodeURIComponent(String(artifactModalRun))}/monitor`)}
                style={{ padding: "6px 10px", borderRadius: 10, border: "1px solid #e5e7eb", background: "#fff", cursor: "pointer" }}
              >
                Open Monitor
              </button>
            </div>
          </div>
        </Modal>
      ) : null}
    </div>
  );
}
