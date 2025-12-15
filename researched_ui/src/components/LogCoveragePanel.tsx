import React, { useEffect, useMemo, useState } from "react";

// Source notes:
// - Design Doc: FR-9.6 (Visualization + log consumer audit: parse coverage + fix-it checklist), §1.3 + FR-4 (msg/event mismatch; resid variants), §1.4 (events.jsonl vs evidence_log.jsonl drift)
// - Design Doc: §5.2 (canonical normalized event record: event_source + resid/iter extraction)
// - Repo: electrodrive/researched/plot_service.py (_LogCoverage.snapshot() keys; plots/log_coverage.png path; generate_gate_dashboard embeds log_coverage into gate_dashboard.json)
// - Repo: electrodrive/researched/ws.py (normalize_event(...) sets event_source categories; resid/iter extraction rules; resid_precond/resid_true variants)
// - Repo: electrodrive/researched/api.py (GET /runs/{run_id}/logs/coverage endpoint; artifact download /runs/{run_id}/artifact?path=...; filename drift bridge exists in backend)

type LogCoveragePanelProps = {
  runId: string;
  apiBase?: string; // default "/api"
  className?: string;
};

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function stableClone(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(stableClone);
  if (!isRecord(value)) return value;
  const keys = Object.keys(value).sort((a, b) => a.localeCompare(b));
  const out: Record<string, unknown> = {};
  for (const k of keys) out[k] = stableClone(value[k]);
  return out;
}
function stableStringify(value: unknown, indent = 2): string {
  try {
    return JSON.stringify(stableClone(value), null, indent);
  } catch {
    try {
      return String(value);
    } catch {
      return "<unprintable>";
    }
  }
}

async function fetchText(url: string, signal: AbortSignal): Promise<{ ok: true; text: string } | { ok: false; status: number; message: string }> {
  try {
    const resp = await fetch(url, { method: "GET", signal, headers: { Accept: "*/*" } });
    if (!resp.ok) return { ok: false, status: resp.status, message: `${resp.status} ${resp.statusText}` };
    const text = await resp.text();
    return { ok: true, text };
  } catch (e) {
    if ((e as any)?.name === "AbortError") return { ok: false, status: 0, message: "aborted" };
    return { ok: false, status: 0, message: (e as Error)?.message || "network error" };
  }
}
async function fetchJson(url: string, signal: AbortSignal): Promise<{ ok: true; value: unknown } | { ok: false; status: number; message: string; raw?: string }> {
  const t = await fetchText(url, signal);
  if (!t.ok) return t;
  try {
    return { ok: true, value: JSON.parse(t.text) as unknown };
  } catch (e) {
    return { ok: false, status: 0, message: (e as Error)?.message || "JSON parse error", raw: t.text };
  }
}

type Coverage = {
  total_lines_seen?: number;
  total_records_parsed?: number;
  total_records_emitted?: number;
  total_json_errors?: number;
  total_non_dict_records?: number;
  dropped_by_dedup?: number;
  event_source_counts?: Record<string, number>;
  residual_field_detection_counts?: Record<string, number>;
  ingested_files?: string[];
  per_file?: Record<string, Record<string, number>>;
  last_event_t?: number;
};

function coerceNumber(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

function normalizeCoverage(raw: unknown): Coverage | null {
  if (!isRecord(raw)) return null;
  const c: Coverage = {};
  c.total_lines_seen = coerceNumber((raw as any).total_lines_seen);
  c.total_records_parsed = coerceNumber((raw as any).total_records_parsed);
  c.total_records_emitted = coerceNumber((raw as any).total_records_emitted);
  c.total_json_errors = coerceNumber((raw as any).total_json_errors);
  c.total_non_dict_records = coerceNumber((raw as any).total_non_dict_records);
  c.dropped_by_dedup = coerceNumber((raw as any).dropped_by_dedup);
  c.last_event_t = coerceNumber((raw as any).last_event_t);

  if (isRecord((raw as any).event_source_counts)) {
    const out: Record<string, number> = {};
    for (const [k, v] of Object.entries((raw as any).event_source_counts as Record<string, unknown>)) {
      if (typeof v === "number" && Number.isFinite(v)) out[k] = v;
    }
    c.event_source_counts = out;
  }
  if (isRecord((raw as any).residual_field_detection_counts)) {
    const out: Record<string, number> = {};
    for (const [k, v] of Object.entries((raw as any).residual_field_detection_counts as Record<string, unknown>)) {
      if (typeof v === "number" && Number.isFinite(v)) out[k] = v;
    }
    c.residual_field_detection_counts = out;
  }
  if (Array.isArray((raw as any).ingested_files)) {
    c.ingested_files = ((raw as any).ingested_files as unknown[]).filter((x) => typeof x === "string") as string[];
  }
  if (isRecord((raw as any).per_file)) {
    const out: Record<string, Record<string, number>> = {};
    for (const [fname, stats] of Object.entries((raw as any).per_file as Record<string, unknown>)) {
      if (!isRecord(stats)) continue;
      const row: Record<string, number> = {};
      for (const [k, v] of Object.entries(stats)) {
        if (typeof v === "number" && Number.isFinite(v)) row[k] = v;
      }
      out[fname] = row;
    }
    c.per_file = out;
  }
  return c;
}

type ChecklistItem = {
  severity: "info" | "warn" | "error";
  title: string;
  explanation: string;
  remediation: string;
};

function hasFilename(files: string[] | undefined, filename: string): boolean {
  if (!files) return false;
  return files.some((f) => f === filename || f.endsWith(`/${filename}`) || f.endsWith(`\\${filename}`));
}

function sumCounts(obj: Record<string, number> | undefined): number {
  if (!obj) return 0;
  let s = 0;
  for (const v of Object.values(obj)) s += typeof v === "number" && Number.isFinite(v) ? v : 0;
  return s;
}

function barWidth(value: number, maxValue: number): string {
  if (maxValue <= 0) return "0%";
  const pct = Math.max(0, Math.min(1, value / maxValue));
  return `${Math.round(pct * 100)}%`;
}

export default function LogCoveragePanel(props: LogCoveragePanelProps) {
  const apiBase = props.apiBase ?? "/api";
  const runId = props.runId;
  const base = apiBase.replace(/\/+$/, "");

  const artifactUrl = (relpath: string) => `${base}/runs/${encodeURIComponent(runId)}/artifact?path=${encodeURIComponent(relpath)}`;
  const logsCoverageUrl = `${base}/runs/${encodeURIComponent(runId)}/logs/coverage`;
  const runUrl = `${base}/runs/${encodeURIComponent(runId)}`;
  const artifactsUrl = `${base}/runs/${encodeURIComponent(runId)}/artifacts?recursive=true`;

  const [coverage, setCoverage] = useState<Coverage | null>(null);
  const [coverageRaw, setCoverageRaw] = useState<unknown>(null);
  const [loading, setLoading] = useState(true);
  const [note, setNote] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [artifactPaths, setArtifactPaths] = useState<string[] | null>(null);
  const [imgError, setImgError] = useState(false);

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    setNote(null);
    setCoverage(null);
    setCoverageRaw(null);
    setArtifactPaths(null);
    setImgError(false);

    (async () => {
      // Primary: gate_dashboard.json -> log_coverage (Design Doc FR-9.6; Repo plot_service.generate_gate_dashboard embeds).
      const gateDash = await fetchJson(artifactUrl("plots/gate_dashboard.json"), ac.signal);
      if (gateDash.ok && isRecord(gateDash.value) && isRecord((gateDash.value as any).log_coverage)) {
        const covRaw = (gateDash.value as any).log_coverage as unknown;
        const cov = normalizeCoverage(covRaw);
        if (cov) {
          setCoverage(cov);
          setCoverageRaw(covRaw);
          setLoading(false);
          return;
        }
      }

      // Fallback: backend coverage endpoint (Repo: electrodrive/researched/api.py GET /runs/{run_id}/logs/coverage).
      const covRes = await fetchJson(logsCoverageUrl, ac.signal);
      if (covRes.ok) {
        const cov = normalizeCoverage(covRes.value);
        if (cov) {
          setCoverage(cov);
          setCoverageRaw(covRes.value);
          setNote("Loaded coverage from /runs/{runId}/logs/coverage (gate dashboard artifact not present).");
          setLoading(false);
          return;
        }
      }

      // Final fallback: show hints via artifact list + run info.
      setNote("Coverage unavailable. Showing best-effort hints based on run directory contents.");
      const arts = await fetchJson(artifactsUrl, ac.signal);
      if (arts.ok && Array.isArray(arts.value)) {
        const paths: string[] = [];
        for (const item of arts.value) {
          if (isRecord(item) && typeof (item as any).path === "string") paths.push((item as any).path);
          else if (typeof item === "string") paths.push(item);
        }
        setArtifactPaths(paths);
      } else {
        // Try run endpoint for artifacts list if any
        const runRes = await fetchJson(runUrl, ac.signal);
        if (runRes.ok && isRecord(runRes.value) && Array.isArray((runRes.value as any).artifacts)) {
          const paths: string[] = [];
          for (const a of (runRes.value as any).artifacts as unknown[]) {
            if (isRecord(a) && typeof (a as any).path === "string") paths.push((a as any).path);
          }
          setArtifactPaths(paths);
        }
      }

      setLoading(false);
    })().catch((e) => {
      if ((e as any)?.name === "AbortError") return;
      setError((e as Error)?.message || "Failed to load coverage");
      setLoading(false);
    });

    return () => ac.abort();
  }, [runId, base]);

  const checklist = useMemo<ChecklistItem[]>(() => {
    // Design Doc FR-9.6 fix-it checklist; §1.3 + FR-4 mismatch realities; §1.4 filename drift.
    const items: ChecklistItem[] = [];
    const src = coverage?.event_source_counts || {};
    const resid = coverage?.residual_field_detection_counts || {};
    const ingested = coverage?.ingested_files;

    const eventCount = src.event ?? 0;
    const msgCount = src.msg ?? 0;
    const embeddedCount = src.embedded_json ?? 0;
    const totalSrc = sumCounts(src);

    if (eventCount === 0 && msgCount > 0) {
      items.push({
        severity: "warn",
        title: "Event name comes from msg (event==0)",
        explanation:
          "Structured logger records often use “msg” rather than “event”. Older parsers that only read rec.event may miss important signals.",
        remediation:
          "Update consumers to normalize event name via event ?? msg ?? message and/or use ResearchED’s normalized event stream (Design Doc §1.3, FR-4).",
      });
    }

    const residCount = resid.resid ?? 0;
    const residVariants = (resid.resid_precond ?? 0) + (resid.resid_true ?? 0) + (resid.resid_precond_l2 ?? 0) + (resid.resid_true_l2 ?? 0);
    if (residCount === 0 && residVariants > 0) {
      items.push({
        severity: "warn",
        title: "Residuals present but “resid” missing",
        explanation:
          "Some runs only emit resid_precond/resid_true (and *_l2 variants). Tools expecting a single “resid” field may fail to plot convergence.",
        remediation:
          "Normalize residual via resid ?? resid_precond ?? resid_true (and l2 variants). Ensure dashboards read residual variants (Design Doc §1.3, FR-4; Repo: electrodrive/researched/ws.py normalize_event).",
      });
    }

    const hasEvents = hasFilename(ingested, "events.jsonl");
    const hasEvidence = hasFilename(ingested, "evidence_log.jsonl");
    if ((hasEvents && !hasEvidence) || (!hasEvents && hasEvidence)) {
      items.push({
        severity: "warn",
        title: "Filename drift: only one log filename ingested",
        explanation:
          "Repo/tooling may write or expect either events.jsonl or evidence_log.jsonl. If only one exists, some consumers may go blank unless bridged.",
        remediation:
          "Bridge filenames so both exist (copy/symlink) or ingest both and merge streams (Design Doc §1.4; Repo: electrodrive/researched/api.py implements filename drift bridge).",
      });
    }

    const jsonErrors = coverage?.total_json_errors ?? 0;
    if (jsonErrors > 0) {
      items.push({
        severity: "error",
        title: "JSON parse errors detected",
        explanation:
          "Some JSONL lines could not be parsed. This can happen with partial writes, concurrent writers, or corrupted logs during live runs.",
        remediation:
          "Check writer concurrency and atomic append behavior; consider buffering/flushing behavior for JSONL emitters. Inspect the offending log files near the end of the run.",
      });
    }

    // Helpful info items (not errors)
    if (embeddedCount > 0) {
      items.push({
        severity: "info",
        title: "Embedded JSON messages detected",
        explanation:
          "Some emitters serialize JSON into the log message string; normalization parsed embedded JSON successfully.",
        remediation:
          "Prefer structured JSONL records when possible, but keep embedded parsing enabled for legacy logs (Design Doc FR-4; Repo: electrodrive/researched/ws.py normalize_event).",
      });
    }

    if (totalSrc === 0 && !coverage) {
      items.push({
        severity: "info",
        title: "Coverage not available",
        explanation: "No coverage snapshot found in gate_dashboard.json and no coverage endpoint response.",
        remediation: "Generate/refresh plots and coverage via ResearchED backend plot service, or ensure plot_service writes plots/log_coverage.png and embeds coverage.",
      });
    }

    // Dedup info
    const dropped = coverage?.dropped_by_dedup ?? 0;
    if (dropped > 0) {
      items.push({
        severity: "info",
        title: "Deduplication dropped records",
        explanation: `Log merger deduplicated ${dropped} records (common when merging events.jsonl + evidence_log.jsonl).`,
        remediation: "If drops seem excessive, review dedup key and ensure log producers do not duplicate identical lines unintentionally.",
      });
    }

    // If no issues detected, add a friendly “all good” item.
    if (items.length === 0 && coverage) {
      items.push({
        severity: "info",
        title: "No major ingestion issues detected",
        explanation: "Coverage snapshot indicates logs were parsed and normalized successfully.",
        remediation: "Keep using normalized log ingestion for downstream dashboards.",
      });
    }

    return items;
  }, [coverage]);

  const ingestedFiles = useMemo(() => {
    if (coverage?.ingested_files?.length) return coverage.ingested_files;
    const keys = coverage?.per_file ? Object.keys(coverage.per_file) : [];
    if (keys.length) return keys;
    return [];
  }, [coverage]);

  const eventSourceCounts = coverage?.event_source_counts || {};
  const maxEventSource = Math.max(0, ...Object.values(eventSourceCounts).map((v) => (typeof v === "number" ? v : 0)));

  const residCounts = coverage?.residual_field_detection_counts || {};
  const residKeys = ["resid", "resid_precond", "resid_precond_l2", "resid_true", "resid_true_l2"];

  const styleRoot: React.CSSProperties = {
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif",
    fontSize: 14,
    lineHeight: 1.4,
    color: "#111",
  };

  const styleCard: React.CSSProperties = {
    border: "1px solid #ddd",
    borderRadius: 8,
    padding: 12,
    background: "#fff",
  };

  const styleMono: React.CSSProperties = {
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
    fontSize: 12,
  };

  const badge = (sev: ChecklistItem["severity"]): React.CSSProperties => {
    const base: React.CSSProperties = { display: "inline-block", padding: "2px 8px", borderRadius: 999, fontSize: 12 };
    if (sev === "error") return { ...base, background: "#fdecec", color: "#8a1212" };
    if (sev === "warn") return { ...base, background: "#fff3cd", color: "#664d03" };
    return { ...base, background: "#f1f1f1", color: "#555" };
  };

  return (
    <div className={props.className} style={styleRoot}>
      <div style={styleCard}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
          <div>
            <h3 style={{ margin: 0 }}>Log coverage</h3>
            <div style={{ marginTop: 4, color: "#555" }}>
              Run <span style={styleMono}>{runId}</span> · primary artifact <span style={styleMono}>plots/gate_dashboard.json</span> →{" "}
              <span style={styleMono}>log_coverage</span>
            </div>
          </div>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", justifyContent: "flex-end" }}>
            <a href={artifactUrl("plots/gate_dashboard.json")} target="_blank" rel="noreferrer">
              gate_dashboard.json
            </a>
            <a href={artifactUrl("plots/log_coverage.png")} target="_blank" rel="noreferrer">
              log_coverage.png
            </a>
          </div>
        </div>

        {loading ? <div style={{ marginTop: 10, color: "#555" }}>Loading…</div> : null}
        {error ? (
          <div role="alert" style={{ marginTop: 10, color: "#8a1212" }}>
            {error}
          </div>
        ) : null}
        {note ? (
          <div style={{ marginTop: 10, padding: 10, borderRadius: 6, background: "#f8f9fa", border: "1px solid #eee", color: "#555" }}>
            {note}
          </div>
        ) : null}

        {!loading && !error ? (
          <>
            {coverage ? (
              <>
                <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12, marginTop: 12 }}>
                  <div style={{ border: "1px solid #eee", borderRadius: 6, padding: 10 }}>
                    <div style={{ fontWeight: 700, marginBottom: 6 }}>Summary</div>
                    <div>
                      total_lines_seen: <span style={styleMono}>{coverage.total_lines_seen ?? "—"}</span>
                    </div>
                    <div>
                      total_records_parsed: <span style={styleMono}>{coverage.total_records_parsed ?? "—"}</span>
                    </div>
                    <div>
                      total_records_emitted: <span style={styleMono}>{coverage.total_records_emitted ?? "—"}</span>
                    </div>
                    <div style={{ marginTop: 6 }}>
                      dropped_by_dedup: <span style={styleMono}>{coverage.dropped_by_dedup ?? "—"}</span>
                    </div>
                  </div>

                  <div style={{ border: "1px solid #eee", borderRadius: 6, padding: 10 }}>
                    <div style={{ fontWeight: 700, marginBottom: 6 }}>Errors</div>
                    <div>
                      total_json_errors: <span style={styleMono}>{coverage.total_json_errors ?? "—"}</span>
                    </div>
                    <div>
                      total_non_dict_records: <span style={styleMono}>{coverage.total_non_dict_records ?? "—"}</span>
                    </div>
                    <div style={{ marginTop: 6 }}>
                      last_event_t: <span style={styleMono}>{coverage.last_event_t ?? "—"}</span>
                    </div>
                  </div>

                  <div style={{ border: "1px solid #eee", borderRadius: 6, padding: 10 }}>
                    <div style={{ fontWeight: 700, marginBottom: 6 }}>Ingested files</div>
                    {ingestedFiles.length ? (
                      <ul style={{ margin: 0, paddingLeft: 18 }}>
                        {ingestedFiles.map((f) => (
                          <li key={f}>
                            <span style={styleMono}>{f}</span>
                          </li>
                        ))}
                      </ul>
                    ) : (
                      <div style={{ color: "#666" }}>No ingested_files recorded.</div>
                    )}
                  </div>
                </div>

                <div style={{ marginTop: 14 }}>
                  <div style={{ fontWeight: 700, marginBottom: 8 }}>Event-source breakdown</div>
                  <div style={{ color: "#666", fontSize: 12, marginBottom: 8 }}>
                    event_source categories come from normalization (Repo: electrodrive/researched/ws.py normalize_event). (Design Doc §5.2)
                  </div>
                  <div style={{ display: "grid", gap: 8 }}>
                    {Object.keys(eventSourceCounts).length ? (
                      Object.entries(eventSourceCounts)
                        .sort((a, b) => (b[1] ?? 0) - (a[1] ?? 0))
                        .map(([k, v]) => (
                          <div key={k} style={{ display: "grid", gridTemplateColumns: "160px 1fr 80px", gap: 10, alignItems: "center" }}>
                            <div style={styleMono}>{k}</div>
                            <div style={{ height: 10, border: "1px solid #eee", borderRadius: 6, overflow: "hidden", background: "#fafafa" }}>
                              <div style={{ height: "100%", width: barWidth(v, maxEventSource), background: "#1b4f9c" }} />
                            </div>
                            <div style={{ ...styleMono, textAlign: "right" }}>{v}</div>
                          </div>
                        ))
                    ) : (
                      <div style={{ color: "#666" }}>No event_source_counts recorded.</div>
                    )}
                  </div>
                </div>

                <div style={{ marginTop: 14 }}>
                  <div style={{ fontWeight: 700, marginBottom: 8 }}>Residual variant detection</div>
                  <div style={{ overflowX: "auto", border: "1px solid #eee", borderRadius: 6 }}>
                    <table style={{ width: "100%", borderCollapse: "collapse" }}>
                      <thead>
                        <tr style={{ background: "#fafafa" }}>
                          <th style={{ textAlign: "left", padding: "8px 6px", borderBottom: "1px solid #eee" }}>field</th>
                          <th style={{ textAlign: "right", padding: "8px 6px", borderBottom: "1px solid #eee" }}>count</th>
                        </tr>
                      </thead>
                      <tbody>
                        {residKeys.map((k) => (
                          <tr key={k}>
                            <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3" }}>
                              <span style={styleMono}>{k}</span>
                            </td>
                            <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3", textAlign: "right" }}>
                              <span style={styleMono}>{(residCounts as any)[k] ?? 0}</span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                {coverage.per_file && Object.keys(coverage.per_file).length ? (
                  <div style={{ marginTop: 14 }}>
                    <div style={{ fontWeight: 700, marginBottom: 8 }}>Per-file coverage</div>
                    <div style={{ overflowX: "auto", border: "1px solid #eee", borderRadius: 6 }}>
                      <table style={{ width: "100%", borderCollapse: "collapse" }}>
                        <thead>
                          <tr style={{ background: "#fafafa" }}>
                            <th style={{ textAlign: "left", padding: "8px 6px", borderBottom: "1px solid #eee" }}>file</th>
                            <th style={{ textAlign: "right", padding: "8px 6px", borderBottom: "1px solid #eee" }}>lines_seen</th>
                            <th style={{ textAlign: "right", padding: "8px 6px", borderBottom: "1px solid #eee" }}>records_parsed</th>
                            <th style={{ textAlign: "right", padding: "8px 6px", borderBottom: "1px solid #eee" }}>json_errors</th>
                            <th style={{ textAlign: "right", padding: "8px 6px", borderBottom: "1px solid #eee" }}>non_dict</th>
                            <th style={{ textAlign: "right", padding: "8px 6px", borderBottom: "1px solid #eee" }}>emitted</th>
                          </tr>
                        </thead>
                        <tbody>
                          {Object.entries(coverage.per_file).map(([fname, stats]) => (
                            <tr key={fname}>
                              <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3" }}>
                                <span style={styleMono}>{fname}</span>
                              </td>
                              <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3", textAlign: "right" }}>
                                <span style={styleMono}>{(stats as any).lines_seen ?? (stats as any).total_lines_seen ?? "—"}</span>
                              </td>
                              <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3", textAlign: "right" }}>
                                <span style={styleMono}>{(stats as any).records_parsed ?? (stats as any).total_records_parsed ?? "—"}</span>
                              </td>
                              <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3", textAlign: "right" }}>
                                <span style={styleMono}>{(stats as any).json_errors ?? (stats as any).total_json_errors ?? "—"}</span>
                              </td>
                              <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3", textAlign: "right" }}>
                                <span style={styleMono}>{(stats as any).non_dict ?? (stats as any).total_non_dict_records ?? "—"}</span>
                              </td>
                              <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3", textAlign: "right" }}>
                                <span style={styleMono}>{(stats as any).emitted ?? (stats as any).total_records_emitted ?? "—"}</span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                ) : null}

                <div style={{ marginTop: 14 }}>
                  <div style={{ fontWeight: 700, marginBottom: 6 }}>Coverage plot</div>
                  {!imgError ? (
                    <img
                      src={artifactUrl("plots/log_coverage.png")}
                      alt="Log coverage plot"
                      style={{ width: "100%", maxHeight: 360, objectFit: "contain", border: "1px solid #eee", borderRadius: 6, background: "#fafafa" }}
                      onError={() => setImgError(true)}
                    />
                  ) : (
                    <div style={{ color: "#666" }}>plots/log_coverage.png not available for this run.</div>
                  )}
                </div>

                <div style={{ marginTop: 14 }}>
                  <div style={{ fontWeight: 700, marginBottom: 8 }}>Fix-it checklist</div>
                  <div style={{ display: "grid", gap: 10 }}>
                    {checklist.map((it, idx) => (
                      <div key={idx} style={{ border: "1px solid #eee", borderRadius: 6, padding: 10 }}>
                        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12 }}>
                          <div style={{ fontWeight: 700 }}>{it.title}</div>
                          <span style={badge(it.severity)}>{it.severity}</span>
                        </div>
                        <div style={{ marginTop: 6, color: "#555" }}>{it.explanation}</div>
                        <div style={{ marginTop: 6 }}>
                          <strong>Remediation:</strong> <span style={{ color: "#333" }}>{it.remediation}</span>
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <details style={{ marginTop: 14 }}>
                  <summary style={{ cursor: "pointer", fontWeight: 700 }}>Raw coverage JSON</summary>
                  <pre style={{ marginTop: 8, padding: 10, border: "1px solid #eee", borderRadius: 6, background: "#fafafa", overflowX: "auto", ...styleMono }}>
                    {coverageRaw ? stableStringify(coverageRaw, 2) : "—"}
                  </pre>
                </details>
              </>
            ) : (
              <>
                <div style={{ marginTop: 12, color: "#666" }}>
                  Coverage snapshot not available. (Design Doc FR-9.6 expects coverage via plot_service or a backend endpoint.)
                </div>

                {artifactPaths && artifactPaths.length ? (
                  <div style={{ marginTop: 12 }}>
                    <div style={{ fontWeight: 700, marginBottom: 6 }}>Detected run_dir files (best-effort)</div>
                    <div style={{ color: "#666", fontSize: 12, marginBottom: 8 }}>
                      This list is derived from artifact indexing. Use it to confirm presence of events.jsonl vs evidence_log.jsonl (Design Doc §1.4).
                    </div>
                    <ul style={{ margin: 0, paddingLeft: 18 }}>
                      {artifactPaths
                        .filter((p) => /events\.jsonl$|evidence_log\.jsonl$|train_log\.jsonl$|metrics\.jsonl$|stdout\.log$|stderr\.log$/.test(p))
                        .slice(0, 25)
                        .map((p) => (
                          <li key={p}>
                            <span style={styleMono}>{p}</span>
                          </li>
                        ))}
                    </ul>
                    <div style={{ marginTop: 6, color: "#666", fontSize: 12 }}>
                      (Showing up to 25 log-ish files. If you don’t see expected logs, check the run directory contract and the backend’s artifact indexer.)
                    </div>
                  </div>
                ) : null}

                <div style={{ marginTop: 14 }}>
                  <div style={{ fontWeight: 700, marginBottom: 8 }}>Fix-it checklist (limited)</div>
                  <div style={{ color: "#555" }}>
                    Without coverage counters, use the Gate dashboard generator (plot_service) or enable coverage snapshot export in the backend.
                  </div>
                </div>
              </>
            )}
          </>
        ) : null}
      </div>
    </div>
  );
}
