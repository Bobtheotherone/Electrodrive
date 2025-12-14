import React, { useEffect, useMemo, useState } from "react";

// Source notes:
// - Design Doc: FR-9.5 (Gate + structure dashboards; stable gate_dashboard.json/png), FR-7 (post-run dashboards reference gate summary), §8 (day-one dashboards include gate dashboard)
// - Repo: electrodrive/researched/plot_service.py (generate_gate_dashboard -> plots/gate_dashboard.json shape; paths plots/gate_dashboard.json, plots/gate_dashboard.png, plots/log_coverage.png)
// - Repo: tools/images_gate2.py (compute_structural_summary output: n_images, families, degeneracies, structure_score, gate2_status, note, fingerprint)
// - Repo: electrodrive/discovery/novelty.py (compute_gate3_status statuses: "pass" / "non_novel" / "n/a"; novelty_score thresholds)
// - Repo: electrodrive/researched/api.py (GET /runs/{run_id} returns manifest; artifact download via /runs/{run_id}/artifact?path=...)
// - Repo: electrodrive/researched/templates/report.html.j2 (report links gate_dashboard.json/png and log_coverage.png)

type GateDashboardPanelProps = {
  runId: string;
  apiBase?: string; // default "/api"
  className?: string;
};

function isRecord(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}
function isString(v: unknown): v is string {
  return typeof v === "string";
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

function coerceFiniteNumber(v: unknown): number | undefined {
  return typeof v === "number" && Number.isFinite(v) ? v : undefined;
}

async function fetchText(
  url: string,
  signal: AbortSignal,
): Promise<{ ok: true; text: string } | { ok: false; status: number; message: string }> {
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
async function fetchJson(
  url: string,
  signal: AbortSignal,
): Promise<{ ok: true; value: unknown } | { ok: false; status: number; message: string; raw?: string }> {
  const t = await fetchText(url, signal);
  if (!t.ok) return t;
  try {
    return { ok: true, value: JSON.parse(t.text) as unknown };
  } catch (e) {
    return { ok: false, status: 0, message: (e as Error)?.message || "JSON parse error", raw: t.text };
  }
}

type FamilyRow = {
  family_name: string;
  count?: number;
  weight_l1?: number;
  weight_linf?: number;
  z_min?: number;
  z_max?: number;
};

function normalizeFamilies(families: unknown): FamilyRow[] {
  if (Array.isArray(families)) {
    const out: FamilyRow[] = [];
    for (const f of families) {
      if (!isRecord(f)) continue;
      const family_name = (isString((f as any).family_name) && (f as any).family_name) || (isString((f as any).name) && (f as any).name) || "—";
      const row: FamilyRow = {
        family_name,
        count: coerceFiniteNumber((f as any).count),
        weight_l1: coerceFiniteNumber((f as any).weight_l1),
        weight_linf: coerceFiniteNumber((f as any).weight_linf),
        z_min: coerceFiniteNumber((f as any).z_min),
        z_max: coerceFiniteNumber((f as any).z_max),
      };
      out.push(row);
    }
    return out.sort((a, b) => (b.weight_l1 ?? 0) - (a.weight_l1 ?? 0));
  }
  if (isRecord(families)) {
    const out: FamilyRow[] = [];
    for (const [name, stats] of Object.entries(families)) {
      if (!isRecord(stats)) continue;
      out.push({
        family_name: name,
        count: coerceFiniteNumber((stats as any).count),
        weight_l1: coerceFiniteNumber((stats as any).weight_l1),
        weight_linf: coerceFiniteNumber((stats as any).weight_linf),
        z_min: coerceFiniteNumber((stats as any).z_min),
        z_max: coerceFiniteNumber((stats as any).z_max),
      });
    }
    return out.sort((a, b) => (b.weight_l1 ?? 0) - (a.weight_l1 ?? 0));
  }
  return [];
}

function pillStyle(status: string | undefined): React.CSSProperties {
  const s = (status || "").toLowerCase();
  let bg = "#eee";
  let fg = "#111";
  if (s === "pass" || s === "ok" || s === "success") {
    bg = "#e7f7ec";
    fg = "#0b5d1e";
  } else if (s === "fail" || s === "error") {
    bg = "#fdecec";
    fg = "#8a1212";
  } else if (s === "borderline" || s === "warn" || s === "warning" || s === "non_novel" || s === "non-novel") {
    // Gate3 "non_novel" is not a crash; treat it like a warning/borderline outcome.
    bg = "#fff3cd";
    fg = "#664d03";
  } else if (s === "n/a" || s === "na") {
    bg = "#f1f1f1";
    fg = "#555";
  }
  return { display: "inline-block", padding: "2px 8px", borderRadius: 999, background: bg, color: fg, fontSize: 12 };
}

function pickFromManifest(man: Record<string, unknown> | null) {
  const gate = man && isRecord((man as any).gate) ? ((man as any).gate as Record<string, unknown>) : null;

  const gate2_status =
    (gate && isString((gate as any).gate2_status) && String((gate as any).gate2_status)) ||
    (man && isString((man as any).gate2_status) && String((man as any).gate2_status)) ||
    "n/a";

  const gate3_status =
    (gate && isString((gate as any).gate3_status) && String((gate as any).gate3_status)) ||
    (man && isString((man as any).gate3_status) && String((man as any).gate3_status)) ||
    "n/a";

  const structure_score = coerceFiniteNumber(gate ? (gate as any).structure_score : undefined) ?? coerceFiniteNumber(man ? (man as any).structure_score : undefined);

  const novelty_score = coerceFiniteNumber(gate ? (gate as any).novelty_score : undefined) ?? coerceFiniteNumber(man ? (man as any).novelty_score : undefined);

  return { gate2_status, gate3_status, structure_score, novelty_score };
}

export default function GateDashboardPanel(props: GateDashboardPanelProps) {
  const apiBase = props.apiBase ?? "/api";
  const runId = props.runId;
  const base = apiBase.replace(/\/+$/, "");

  const runUrl = `${base}/runs/${encodeURIComponent(runId)}`;
  const artifactUrl = (relpath: string) => `${base}/runs/${encodeURIComponent(runId)}/artifact?path=${encodeURIComponent(relpath)}`;

  const [manifest, setManifest] = useState<Record<string, unknown> | null>(null);
  const [dashboard, setDashboard] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);
  const [warn, setWarn] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [imgGateError, setImgGateError] = useState(false);
  const [imgCovError, setImgCovError] = useState(false);

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    setError(null);
    setWarn(null);
    setManifest(null);
    setDashboard(null);
    setImgGateError(false);
    setImgCovError(false);

    (async () => {
      const [runRes, dashRes] = await Promise.all([fetchJson(runUrl, ac.signal), fetchJson(artifactUrl("plots/gate_dashboard.json"), ac.signal)]);
      if (runRes.ok && isRecord(runRes.value) && isRecord((runRes.value as any).manifest)) {
        setManifest((runRes.value as any).manifest as Record<string, unknown>);
      } else if (runRes.ok && isRecord(runRes.value)) {
        // fallback: response may itself be manifest-like
        setManifest(runRes.value as Record<string, unknown>);
      }

      if (dashRes.ok && isRecord(dashRes.value)) {
        setDashboard(dashRes.value);
      } else {
        // Not fatal: use manifest.gate where possible.
        setWarn("Gate dashboard artifact not found; showing gate fields from manifest when available.");
      }

      setLoading(false);
    })().catch((e) => {
      if ((e as any)?.name === "AbortError") return;
      setError((e as Error)?.message || "Failed to load gate dashboard");
      setLoading(false);
    });

    return () => ac.abort();
  }, [runUrl, base, runId]);

  const derived = useMemo(() => {
    const man = manifest;
    const fromMan = pickFromManifest(man);
    const d = dashboard;

    const gate2_status = (d && isString((d as any).gate2_status) && String((d as any).gate2_status)) || fromMan.gate2_status;
    const gate3_status = (d && isString((d as any).gate3_status) && String((d as any).gate3_status)) || fromMan.gate3_status;

    const structure_score = coerceFiniteNumber(d ? (d as any).structure_score : undefined) ?? fromMan.structure_score;
    const novelty_score = coerceFiniteNumber(d ? (d as any).novelty_score : undefined) ?? fromMan.novelty_score;

    const generated_at = d ? ((d as any).generated_at as unknown) : undefined;
    const workflow = (d && isString((d as any).workflow) && String((d as any).workflow)) || (man && isString((man as any).workflow) && String((man as any).workflow)) || undefined;

    const gate2_summary = d && isRecord((d as any).gate2_summary) ? ((d as any).gate2_summary as Record<string, unknown>) : null;
    const warnings = d && Array.isArray((d as any).warnings) ? (((d as any).warnings as unknown[]) ?? null) : null;

    return { gate2_status, gate3_status, structure_score, novelty_score, generated_at, workflow, gate2_summary, warnings };
  }, [manifest, dashboard]);

  const families = useMemo(() => {
    const s = derived.gate2_summary;
    if (!s) return [];
    return normalizeFamilies((s as any).families);
  }, [derived.gate2_summary]);

  const degeneracies = useMemo(() => {
    const s = derived.gate2_summary;
    if (!s || !isRecord((s as any).degeneracies)) return null;
    return (s as any).degeneracies as Record<string, unknown>;
  }, [derived.gate2_summary]);

  const note = useMemo(() => {
    const s = derived.gate2_summary;
    if (!s) return null;
    const n = (s as any).note;
    return isString(n) && n.trim() ? n.trim() : null;
  }, [derived.gate2_summary]);

  const nImages = useMemo(() => {
    const s = derived.gate2_summary;
    if (!s) return null;
    const n = (s as any).n_images;
    return typeof n === "number" ? n : null;
  }, [derived.gate2_summary]);

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

  return (
    <div className={props.className} style={styleRoot}>
      <div style={styleCard}>
        <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12 }}>
          <div>
            <h3 style={{ margin: 0 }}>Gate dashboard</h3>
            <div style={{ color: "#555", marginTop: 4 }}>
              Run <span style={styleMono}>{runId}</span>
              {derived.workflow ? (
                <>
                  {" "}
                  · workflow <span style={{ fontWeight: 700 }}>{derived.workflow}</span>
                </>
              ) : null}
              {derived.generated_at ? (
                <>
                  {" "}
                  · generated <span style={styleMono}>{String(derived.generated_at)}</span>
                </>
              ) : null}
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", justifyContent: "flex-end" }}>
            <a href={artifactUrl("plots/gate_dashboard.json")} target="_blank" rel="noreferrer">
              gate_dashboard.json
            </a>
            <a href={artifactUrl("plots/gate_dashboard.png")} target="_blank" rel="noreferrer">
              gate_dashboard.png
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
        {warn ? (
          <div style={{ marginTop: 10, padding: 10, borderRadius: 6, background: "#fff8e1", border: "1px solid #f1e2a6", color: "#664d03" }}>
            {warn}
          </div>
        ) : null}

        {!loading && !error ? (
          <>
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12, marginTop: 12 }}>
              <div style={{ border: "1px solid #eee", borderRadius: 6, padding: 10 }}>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>Gate2</div>
                <div>
                  status: <span style={pillStyle(derived.gate2_status)}>{derived.gate2_status}</span>
                </div>
                <div style={{ marginTop: 6 }}>
                  structure_score:{" "}
                  <span style={styleMono}>{typeof derived.structure_score === "number" ? derived.structure_score.toFixed(6) : "—"}</span>
                </div>
                {typeof nImages === "number" ? (
                  <div style={{ marginTop: 6 }}>
                    n_images: <span style={styleMono}>{nImages}</span>
                  </div>
                ) : null}
                {note ? (
                  <div style={{ marginTop: 8, color: "#555" }}>
                    <strong>note:</strong> {note}
                  </div>
                ) : null}
              </div>

              <div style={{ border: "1px solid #eee", borderRadius: 6, padding: 10 }}>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>Gate3</div>
                <div>
                  status: <span style={pillStyle(derived.gate3_status)}>{derived.gate3_status}</span>
                </div>
                <div style={{ marginTop: 6 }}>
                  novelty_score:{" "}
                  <span style={styleMono}>{typeof derived.novelty_score === "number" ? derived.novelty_score.toFixed(6) : "—"}</span>
                </div>
                <div style={{ marginTop: 8, color: "#666", fontSize: 12 }}>
                  Gate3 status semantics follow repo gating logic (Repo: electrodrive/discovery/novelty.py compute_gate3_status).
                </div>
              </div>

              <div style={{ border: "1px solid #eee", borderRadius: 6, padding: 10 }}>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>Artifacts</div>
                <div style={{ color: "#555" }}>
                  Expected (Design Doc FR-9.5): <span style={styleMono}>plots/gate_dashboard.json</span> + <span style={styleMono}>plots/gate_dashboard.png</span>
                </div>
                <div style={{ marginTop: 8, color: "#666", fontSize: 12 }}>
                  This panel is workflow-agnostic; for non-discovery runs, statuses may be “n/a”.
                </div>
              </div>
            </div>

            {derived.warnings && derived.warnings.length ? (
              <div style={{ marginTop: 12, padding: 10, borderRadius: 6, background: "#f8f9fa", border: "1px solid #eee" }}>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>Warnings</div>
                <ul style={{ margin: 0, paddingLeft: 18 }}>
                  {derived.warnings.map((w, i) => (
                    <li key={i} style={{ color: "#555" }}>
                      {typeof w === "string" ? w : stableStringify(w, 0)}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}

            <div style={{ marginTop: 12 }}>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>Gate dashboard plot</div>
              {!imgGateError ? (
                <img
                  src={artifactUrl("plots/gate_dashboard.png")}
                  alt="Gate dashboard plot"
                  style={{ width: "100%", maxHeight: 420, objectFit: "contain", border: "1px solid #eee", borderRadius: 6, background: "#fafafa" }}
                  onError={() => setImgGateError(true)}
                />
              ) : (
                <div style={{ color: "#666" }}>plots/gate_dashboard.png not available for this run.</div>
              )}
            </div>

            {families.length ? (
              <div style={{ marginTop: 14 }}>
                <div style={{ fontWeight: 700, marginBottom: 8 }}>Family summary</div>
                <div style={{ overflowX: "auto", border: "1px solid #eee", borderRadius: 6 }}>
                  <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                      <tr style={{ background: "#fafafa" }}>
                        <th style={{ textAlign: "left", padding: "8px 6px", borderBottom: "1px solid #eee" }}>family_name</th>
                        <th style={{ textAlign: "right", padding: "8px 6px", borderBottom: "1px solid #eee" }}>count</th>
                        <th style={{ textAlign: "right", padding: "8px 6px", borderBottom: "1px solid #eee" }}>weight_l1</th>
                        <th style={{ textAlign: "right", padding: "8px 6px", borderBottom: "1px solid #eee" }}>weight_linf</th>
                        <th style={{ textAlign: "right", padding: "8px 6px", borderBottom: "1px solid #eee" }}>z_min</th>
                        <th style={{ textAlign: "right", padding: "8px 6px", borderBottom: "1px solid #eee" }}>z_max</th>
                      </tr>
                    </thead>
                    <tbody>
                      {families.map((f) => (
                        <tr key={f.family_name}>
                          <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3" }}>
                            <span style={styleMono}>{f.family_name}</span>
                          </td>
                          <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3", textAlign: "right" }}>
                            {typeof f.count === "number" ? f.count : "—"}
                          </td>
                          <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3", textAlign: "right" }}>
                            {typeof f.weight_l1 === "number" ? f.weight_l1.toFixed(6) : "—"}
                          </td>
                          <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3", textAlign: "right" }}>
                            {typeof f.weight_linf === "number" ? f.weight_linf.toFixed(6) : "—"}
                          </td>
                          <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3", textAlign: "right" }}>
                            {typeof f.z_min === "number" ? f.z_min.toFixed(6) : "—"}
                          </td>
                          <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3", textAlign: "right" }}>
                            {typeof f.z_max === "number" ? f.z_max.toFixed(6) : "—"}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
                <div style={{ marginTop: 6, color: "#666", fontSize: 12 }}>
                  Family stats come from the Gate2 structural summary contract (Repo: tools/images_gate2.py compute_structural_summary).
                </div>
              </div>
            ) : null}

            {degeneracies ? (
              <div style={{ marginTop: 14 }}>
                <div style={{ fontWeight: 700, marginBottom: 8 }}>Degeneracies</div>
                <div style={{ overflowX: "auto", border: "1px solid #eee", borderRadius: 6 }}>
                  <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                      <tr style={{ background: "#fafafa" }}>
                        <th style={{ textAlign: "left", padding: "8px 6px", borderBottom: "1px solid #eee" }}>key</th>
                        <th style={{ textAlign: "left", padding: "8px 6px", borderBottom: "1px solid #eee" }}>value</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(degeneracies).map(([k, v]) => (
                        <tr key={k}>
                          <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3" }}>
                            <span style={styleMono}>{k}</span>
                          </td>
                          <td style={{ padding: "7px 6px", borderBottom: "1px solid #f3f3f3" }}>{typeof v === "string" ? v : stableStringify(v, 0)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            ) : null}

            <details style={{ marginTop: 14 }}>
              <summary style={{ cursor: "pointer", fontWeight: 700 }}>Raw gate_dashboard.json</summary>
              <pre style={{ marginTop: 8, padding: 10, border: "1px solid #eee", borderRadius: 6, background: "#fafafa", overflowX: "auto", ...styleMono }}>
                {dashboard ? stableStringify(dashboard, 2) : "No gate_dashboard.json loaded."}
              </pre>
            </details>

            <details style={{ marginTop: 10 }}>
              <summary style={{ cursor: "pointer" }}>Raw manifest gate block</summary>
              <pre style={{ marginTop: 8, padding: 10, border: "1px solid #eee", borderRadius: 6, background: "#fafafa", overflowX: "auto", ...styleMono }}>
                {manifest ? stableStringify((manifest as any).gate ?? pickFromManifest(manifest), 2) : "No manifest loaded."}
              </pre>
            </details>

            <div style={{ marginTop: 14 }}>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>Log coverage plot</div>
              {!imgCovError ? (
                <img
                  src={artifactUrl("plots/log_coverage.png")}
                  alt="Log coverage plot"
                  style={{ width: "100%", maxHeight: 320, objectFit: "contain", border: "1px solid #eee", borderRadius: 6, background: "#fafafa" }}
                  onError={() => setImgCovError(true)}
                />
              ) : (
                <div style={{ color: "#666" }}>plots/log_coverage.png not available for this run.</div>
              )}
            </div>
          </>
        ) : null}
      </div>
    </div>
  );
}
