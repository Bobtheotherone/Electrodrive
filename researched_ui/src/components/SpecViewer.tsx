import React, { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";

// Source notes:
// - Design Doc: FR-7 (Program/spec dashboards: spec inspector view + inputs/model view), §5.1 (manifest schema), §5.2 (event shape referenced by downstream tooling)
// - Repo: electrodrive/orchestration/parser.py (CanonicalSpec minimal shape + fields; domain/conductors/dielectrics/charges/BCs/symmetry/queries/symbols; summary/to_json)
// - Repo: electrodrive/researched/api.py (run manifest retrieval via GET /runs/{run_id}; artifact download via /runs/{run_id}/artifact?path=... and /runs/{run_id}/artifacts/{relpath:path}; manifests may be manifest.researched.json or manifest.json)

type SpecViewerProps = {
  runId: string;
  apiBase?: string; // default "/api"
  className?: string;
};

type RunEnvelope = {
  // server may return run metadata plus `manifest`
  manifest?: unknown;
  run_id?: string;
  workflow?: string;
  status?: string;
  dir?: string;
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
    // Fall back to best-effort stringification.
    try {
      return String(value);
    } catch {
      return "<unprintable>";
    }
  }
}

function formatTimestamp(ts: unknown): string {
  if (typeof ts === "number" && Number.isFinite(ts)) {
    try {
      const d = new Date(ts * (ts > 1e12 ? 1 : 1000));
      return d.toISOString();
    } catch {
      return String(ts);
    }
  }
  if (typeof ts === "string") return ts;
  return "—";
}

function normalizeRelPath(raw: string): string {
  // Normalize Windows paths to forward slashes and trim.
  return raw.trim().replace(/\\/g, "/").replace(/^\.\/+/, "");
}

function isSafeRelPath(raw: string): boolean {
  // Do not allow absolute paths or traversal. (SpecViewer requirement: only fetch artifacts safely.)
  // This mirrors the safety intent in electrodrive/researched/api.py, which serves run_dir artifacts by relpath.
  const p = normalizeRelPath(raw);
  if (!p) return false;
  if (p.startsWith("/") || p.startsWith("~")) return false;
  // Windows drive letter or URL scheme.
  if (/^[A-Za-z]:\//.test(p)) return false;
  if (/^[A-Za-z][A-Za-z0-9+.-]*:\/\//.test(p)) return false;
  // Disallow traversal segments only (don’t block benign "a..b").
  const parts = p.split("/");
  if (parts.some((seg) => seg === "..")) return false;
  return true;
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
  const res = await fetchText(url, signal);
  if (!res.ok) return res;
  try {
    return { ok: true, value: JSON.parse(res.text) as unknown };
  } catch (e) {
    return { ok: false, status: 0, message: (e as Error)?.message || "JSON parse error", raw: res.text };
  }
}

function pickSpecDisplayPath(manifest: Record<string, unknown>): string | undefined {
  // For display: prefer the original spec_path if present, even if absolute.
  // For fetching, we use pickSpecFetchCandidates() instead.
  const inputs = manifest.inputs;
  if (isRecord(inputs)) {
    const sp = inputs.spec_path;
    if (isString(sp) && sp.trim()) return sp.trim();
    const rel = (inputs as any).spec_relpath;
    if (isString(rel) && rel.trim()) return normalizeRelPath(rel);
    const spec = inputs.spec;
    if (isString(spec) && spec.trim()) return spec.trim();
  }
  const sp2 = (manifest as any).spec_path;
  if (isString(sp2) && sp2.trim()) return sp2.trim();
  const sp3 = (manifest as any).spec;
  if (isString(sp3) && sp3.trim()) return sp3.trim();
  return undefined;
}

function pickSpecFetchCandidates(manifest: Record<string, unknown>): string[] {
  // For fetching artifacts: prefer run_dir-relative relpaths, and probe a small set of legacy fallbacks.
  const candidates: string[] = [];
  const push = (v: unknown) => {
    if (isString(v) && v.trim()) candidates.push(normalizeRelPath(v));
  };

  const inputs = manifest.inputs;
  if (isRecord(inputs)) {
    // Prefer explicit relpaths first
    push((inputs as any).spec_relpath);
    push((inputs as any).spec_path_in_run);
    push((inputs as any).spec_artifact);
    push(inputs.spec);
    // Only use spec_path if it is itself a safe relpath (not absolute)
    push(inputs.spec_path);
  }

  const outputs = (manifest as any).outputs;
  if (isRecord(outputs)) {
    push((outputs as any).spec_relpath);
    push((outputs as any).spec_path);
    push((outputs as any).spec);
  }

  // Common legacy fallbacks (harmless if missing; we’ll probe)
  candidates.push(
    "artifacts/spec.json",
    "spec.json",
    "artifacts/canonical_spec.json",
    "canonical_spec.json",
    "artifacts/spec.yaml",
    "spec.yaml",
    "artifacts/spec.yml",
    "spec.yml",
  );

  const seen = new Set<string>();
  return candidates.filter((p) => isSafeRelPath(p) && !seen.has(p) && (seen.add(p), true));
}

function validateCanonicalSpecMinimal(spec: unknown): { ok: boolean; errors: string[] } {
  // Repo: electrodrive/orchestration/parser.py CanonicalSpec.from_json expects at least a dict; "domain" is the key we require here.
  const errors: string[] = [];
  if (!isRecord(spec)) errors.push("Spec JSON must be an object.");
  else {
    if (!("domain" in spec)) errors.push('Missing required key "domain".');
    else if (!isRecord((spec as any).domain)) errors.push('"domain" must be an object.');
  }
  return { ok: errors.length === 0, errors };
}

function summarizeConductorTypes(spec: Record<string, unknown>): Array<{ type: string; count: number }> {
  const conductors = (spec as any).conductors;
  if (!Array.isArray(conductors)) return [];
  const counts = new Map<string, number>();
  for (const c of conductors) {
    let t = "unknown";
    if (isRecord(c)) {
      const cand =
        (isString((c as any).type) && (c as any).type) ||
        (isString((c as any).kind) && (c as any).kind) ||
        (isString((c as any).type_name) && (c as any).type_name) ||
        (isString((c as any).shape) && (c as any).shape) ||
        "unknown";
      t = cand;
    }
    counts.set(t, (counts.get(t) || 0) + 1);
  }
  return Array.from(counts.entries())
    .map(([type, count]) => ({ type, count }))
    .sort((a, b) => b.count - a.count || a.type.localeCompare(b.type));
}

type DielectricRow = { name: string; epsilon?: number; z_min?: number; z_max?: number };

function summarizeDielectrics(spec: Record<string, unknown>): DielectricRow[] {
  const dielectrics = (spec as any).dielectrics;
  if (!Array.isArray(dielectrics)) return [];
  const rows: DielectricRow[] = [];
  for (const d of dielectrics) {
    if (!isRecord(d)) continue;
    const name = (isString((d as any).name) && (d as any).name) || (isString((d as any).id) && (d as any).id) || "—";
    const epsilon = typeof (d as any).epsilon === "number" ? (d as any).epsilon : typeof (d as any).eps === "number" ? (d as any).eps : undefined;
    const z_min = typeof (d as any).z_min === "number" ? (d as any).z_min : undefined;
    const z_max = typeof (d as any).z_max === "number" ? (d as any).z_max : undefined;
    rows.push({ name, epsilon, z_min, z_max });
  }
  return rows;
}

type ChargeRow = { type: string; q?: number; pos?: unknown; meta?: Record<string, unknown> };

function summarizeCharges(spec: Record<string, unknown>, limit = 8): ChargeRow[] {
  const charges = (spec as any).charges;
  if (!Array.isArray(charges)) return [];
  const out: ChargeRow[] = [];
  for (const ch of charges.slice(0, Math.max(0, limit))) {
    if (!isRecord(ch)) continue;
    const type = (isString((ch as any).type) && (ch as any).type) || (isString((ch as any).kind) && (ch as any).kind) || "—";
    const q = typeof (ch as any).q === "number" ? (ch as any).q : typeof (ch as any).charge === "number" ? (ch as any).charge : undefined;
    const pos = (ch as any).pos ?? (ch as any).position;
    const meta: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(ch)) {
      if (k === "type" || k === "kind" || k === "q" || k === "charge" || k === "pos" || k === "position") continue;
      if (typeof v === "string" || typeof v === "number" || typeof v === "boolean") meta[k] = v;
    }
    out.push({ type, q, pos, meta: Object.keys(meta).length ? meta : undefined });
  }
  return out;
}

function toHumanList(v: unknown): string {
  if (v === null || v === undefined) return "—";
  if (Array.isArray(v)) return v.map((x) => (typeof x === "string" ? x : stableStringify(x, 0))).join(", ") || "—";
  if (typeof v === "string") return v || "—";
  return stableStringify(v, 0);
}

function useLatestRef<T>(value: T) {
  const ref = useRef(value);
  ref.current = value;
  return ref;
}

export default function SpecViewer(props: SpecViewerProps) {
  const apiBase = props.apiBase ?? "/api";
  const runId = props.runId;

  const [manifest, setManifest] = useState<Record<string, unknown> | null>(null);
  const [manifestMeta, setManifestMeta] = useState<Record<string, unknown> | null>(null);
  const [manifestError, setManifestError] = useState<string | null>(null);
  const [manifestLoading, setManifestLoading] = useState<boolean>(true);

  const [specLoadState, setSpecLoadState] = useState<"idle" | "loading" | "loaded" | "error">("idle");
  const [specRawText, setSpecRawText] = useState<string>("");
  const [specObj, setSpecObj] = useState<Record<string, unknown> | null>(null);
  const [specError, setSpecError] = useState<string | null>(null);

  const [copyStatus, setCopyStatus] = useState<string>("");

  const digestId = useId();
  const specId = useId();

  const fetchBase = apiBase.replace(/\/+$/, "");

  const runUrl = `${fetchBase}/runs/${encodeURIComponent(runId)}`;

  const artifactUrl = useCallback(
    (relpath: string) => {
      // Repo: electrodrive/researched/api.py exposes both:
      // - GET /runs/{run_id}/artifact?path=<relpath>
      // - GET /runs/{run_id}/artifacts/{relpath:path}
      // Prefer query-form for easier encoding.
      return `${fetchBase}/runs/${encodeURIComponent(runId)}/artifact?path=${encodeURIComponent(relpath)}`;
    },
    [fetchBase, runId],
  );

  useEffect(() => {
    // Design Doc FR-7: spec inspector depends on manifest + inputs; tolerate missing/old manifests.
    // Repo: electrodrive/researched/api.py provides GET /runs/{run_id} returning {manifest: ...}.
    const ac = new AbortController();
    setManifestLoading(true);
    setManifestError(null);
    setManifest(null);
    setManifestMeta(null);

    (async () => {
      const res = await fetchJson(runUrl, ac.signal);
      if (!res.ok) {
        setManifestError(`Failed to load run: ${res.message}`);
        setManifestLoading(false);
        return;
      }
      const data = res.value;
      if (isRecord(data)) {
        const env = data as RunEnvelope;
        const m = env.manifest;
        if (isRecord(m)) {
          setManifest(m);
        } else if (isRecord(data)) {
          // Fallback if endpoint returned manifest-like object directly.
          setManifest(data);
        } else {
          setManifest(null);
        }
        setManifestMeta(data);
        setManifestLoading(false);
        return;
      }
      setManifestError("Run response was not an object.");
      setManifestLoading(false);
    })().catch((e) => {
      if ((e as any)?.name === "AbortError") return;
      setManifestError((e as Error)?.message || "Unknown error");
      setManifestLoading(false);
    });

    return () => ac.abort();
  }, [runUrl]);

  const specDigest = useMemo(() => {
    if (!manifest) return null;
    const d = (manifest as any).spec_digest;
    return isRecord(d) ? d : null;
  }, [manifest]);

  const specPathDisplay = useMemo(() => {
    if (!manifest) return undefined;
    return pickSpecDisplayPath(manifest);
  }, [manifest]);

  const specFetchCandidates = useMemo(() => {
    if (!manifest) return [];
    return pickSpecFetchCandidates(manifest);
  }, [manifest]);

  const canLoadSpec = specFetchCandidates.length > 0;

  const specValidation = useMemo(() => validateCanonicalSpecMinimal(specObj), [specObj]);

  const conductorTypes = useMemo(() => (specObj ? summarizeConductorTypes(specObj) : []), [specObj]);
  const dielectricRows = useMemo(() => (specObj ? summarizeDielectrics(specObj) : []), [specObj]);
  const chargeRows = useMemo(() => (specObj ? summarizeCharges(specObj, 8) : []), [specObj]);

  const copyText = useCallback(async (label: string, text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopyStatus(`Copied ${label}`);
      window.setTimeout(() => setCopyStatus(""), 1500);
    } catch {
      setCopyStatus("Clipboard unavailable");
      window.setTimeout(() => setCopyStatus(""), 2000);
    }
  }, []);

  const manifestRef = useLatestRef(manifest);

  // Abort support for spec loads (click handler cannot return cleanup).
  const specLoadAbortRef = useRef<AbortController | null>(null);
  useEffect(() => {
    return () => {
      specLoadAbortRef.current?.abort();
    };
  }, []);

  const onLoadSpec = useCallback(async () => {
    if (!specFetchCandidates.length) return;

    // Abort any previous in-flight load.
    specLoadAbortRef.current?.abort();

    const ac = new AbortController();
    specLoadAbortRef.current = ac;

    setSpecLoadState("loading");
    setSpecError(null);
    setSpecObj(null);
    setSpecRawText("");

    const timeout = window.setTimeout(() => ac.abort(), 30_000);

    try {
      let lastErr: string | null = null;

      for (const rel of specFetchCandidates) {
        const res = await fetchText(artifactUrl(rel), ac.signal);

        if (!res.ok) {
          if (ac.signal.aborted) return;
          lastErr = `Failed to fetch spec artifact "${rel}": ${res.message}`;
          continue; // probe next candidate
        }

        setSpecRawText(res.text);

        let parsed: unknown;
        try {
          parsed = JSON.parse(res.text) as unknown;
        } catch (e) {
          // Avoid adding a hard dependency on a YAML parser here.
          // If your spec is YAML, the backend should copy a JSON version into the run_dir artifacts,
          // or your manifest should provide inputs.spec_relpath pointing at a JSON artifact.
          setSpecLoadState("error");
          setSpecError(
            `Spec parse error for "${rel}": ${(e as Error)?.message || "invalid JSON"}. ` +
              `This UI expects JSON artifacts. If your original spec is YAML, ensure the run saves a JSON copy within run_dir.`,
          );
          return;
        }

        if (!isRecord(parsed)) {
          setSpecLoadState("error");
          setSpecError(`Spec at "${rel}" parsed, but is not an object.`);
          return;
        }

        const v = validateCanonicalSpecMinimal(parsed);

        // Always set specObj if we parsed an object, even if validation warns.
        setSpecObj(parsed);

        if (!v.ok) {
          setSpecLoadState("error");
          setSpecError(`Spec at "${rel}" failed minimal validation: ${v.errors.join(" ")}`);
          return;
        }

        setSpecLoadState("loaded");
        return;
      }

      if (ac.signal.aborted) return;
      setSpecLoadState("error");
      setSpecError(lastErr ?? "No spec candidates found in run artifacts.");
    } finally {
      window.clearTimeout(timeout);
      if (specLoadAbortRef.current === ac) specLoadAbortRef.current = null;
    }
  }, [artifactUrl, specFetchCandidates]);

  const containerStyle: React.CSSProperties = {
    fontFamily: "system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif",
    fontSize: 14,
    lineHeight: 1.4,
    color: "#111",
  };

  const cardStyle: React.CSSProperties = {
    border: "1px solid #ddd",
    borderRadius: 8,
    padding: 12,
    margin: "12px 0",
    background: "#fff",
  };

  const codeStyle: React.CSSProperties = {
    fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
    fontSize: 12,
    whiteSpace: "pre-wrap",
    wordBreak: "break-word",
    background: "#fafafa",
    border: "1px solid #eee",
    borderRadius: 6,
    padding: 10,
    overflowX: "auto",
  };

  const pillStyle = (status: string | undefined): React.CSSProperties => {
    const s = (status || "").toLowerCase();
    let bg = "#eee";
    let fg = "#111";
    if (s === "pass" || s === "ok" || s === "success") {
      bg = "#e7f7ec";
      fg = "#0b5d1e";
    } else if (s === "fail" || s === "error" || s === "killed") {
      bg = "#fdecec";
      fg = "#8a1212";
    } else if (s === "borderline" || s === "warn" || s === "warning") {
      bg = "#fff3cd";
      fg = "#664d03";
    } else if (s === "running") {
      bg = "#eaf2ff";
      fg = "#1b4f9c";
    }
    return { display: "inline-block", padding: "2px 8px", borderRadius: 999, background: bg, color: fg, fontSize: 12 };
  };

  const workflow = useMemo(() => {
    const m = manifestMeta;
    if (m && isString((m as any).workflow)) return (m as any).workflow as string;
    if (manifest && isString((manifest as any).workflow)) return (manifest as any).workflow as string;
    return undefined;
  }, [manifest, manifestMeta]);

  const status = useMemo(() => {
    const m = manifestMeta;
    if (m && isString((m as any).status)) return (m as any).status as string;
    if (manifest && isString((manifest as any).status)) return (manifest as any).status as string;
    return undefined;
  }, [manifest, manifestMeta]);

  const startedAt = useMemo(() => {
    if (!manifest) return undefined;
    return formatTimestamp((manifest as any).started_at);
  }, [manifest]);

  const endedAt = useMemo(() => {
    if (!manifest) return undefined;
    return formatTimestamp((manifest as any).ended_at);
  }, [manifest]);

  return (
    <div className={props.className} style={containerStyle}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12 }}>
        <div>
          <h3 style={{ margin: "0 0 4px 0" }}>Spec Viewer</h3>
          <div style={{ color: "#555" }}>
            Run <span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" }}>{runId}</span>{" "}
            {workflow ? (
              <>
                · workflow <span style={{ fontWeight: 600 }}>{workflow}</span>
              </>
            ) : null}{" "}
            {status ? (
              <>
                · <span style={pillStyle(status)}>{status}</span>
              </>
            ) : null}
            {startedAt || endedAt ? (
              <>
                {" "}
                · <span style={{ color: "#666" }}>{startedAt || "—"} → {endedAt || "—"}</span>
              </>
            ) : null}
          </div>
        </div>
        {copyStatus ? (
          <div aria-live="polite" style={{ color: "#0b5d1e", fontSize: 12 }}>
            {copyStatus}
          </div>
        ) : null}
      </div>

      <section aria-labelledby={digestId} style={cardStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
          <h4 id={digestId} style={{ margin: 0 }}>
            Spec digest
          </h4>
          <button
            type="button"
            onClick={() => copyText("spec_digest", stableStringify(specDigest ?? {}, 2))}
            disabled={!specDigest}
            style={{ padding: "6px 10px" }}
          >
            Copy
          </button>
        </div>
        {manifestLoading ? (
          <div style={{ marginTop: 8, color: "#555" }}>Loading manifest…</div>
        ) : manifestError ? (
          <div role="alert" style={{ marginTop: 8, color: "#8a1212" }}>
            {manifestError}
          </div>
        ) : specDigest ? (
          <pre style={{ ...codeStyle, marginTop: 10 }}>{stableStringify(specDigest, 2)}</pre>
        ) : (
          <div style={{ marginTop: 8, color: "#555" }}>
            Spec digest not available in this run’s manifest. (Design Doc §5.1: spec_digest is optional for older runs.)
          </div>
        )}
      </section>

      <section aria-labelledby={specId} style={cardStyle}>
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
          <h4 id={specId} style={{ margin: 0 }}>
            Spec inspector
          </h4>
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button
              type="button"
              onClick={onLoadSpec}
              disabled={!canLoadSpec || specLoadState === "loading"}
              style={{ padding: "6px 10px" }}
            >
              {specLoadState === "loading" ? "Loading…" : "Load spec JSON"}
            </button>
            <button
              type="button"
              onClick={() => copyText("spec JSON", specRawText || stableStringify(specObj ?? {}, 2))}
              disabled={!specRawText && !specObj}
              style={{ padding: "6px 10px" }}
            >
              Copy JSON
            </button>
          </div>
        </div>

        <div style={{ marginTop: 8, color: "#333" }}>
          <div>
            <strong>spec_path:</strong>{" "}
            <span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" }}>
              {specPathDisplay || "—"}
            </span>
          </div>

          {!specPathDisplay && !canLoadSpec ? (
            <div style={{ marginTop: 6, color: "#666" }}>
              No spec path found in manifest inputs, and no default spec artifact paths detected. (Design Doc §5.1 expects inputs.spec_path when applicable; older runs may omit.)
            </div>
          ) : !canLoadSpec ? (
            <div style={{ marginTop: 6, padding: 10, borderRadius: 6, background: "#fff8e1", border: "1px solid #f1e2a6", color: "#664d03" }}>
              Spec path appears to be external/absolute, and no run_dir-relative spec artifact was found. For safety, this UI only fetches run_dir-relative artifacts.
              If you want the inspector to work for YAML/absolute specs, ensure the backend saves a JSON copy of the spec inside the run directory (and records a relpath in inputs.spec_relpath).
            </div>
          ) : null}
        </div>

        {specError ? (
          <div
            role="alert"
            style={{
              marginTop: 10,
              padding: 10,
              borderRadius: 6,
              background: "#fdecec",
              border: "1px solid #f5c2c7",
              color: "#8a1212",
            }}
          >
            <div style={{ fontWeight: 700, marginBottom: 4 }}>Spec load/validation error</div>
            <div>{specError}</div>
          </div>
        ) : null}

        {specObj ? (
          <div style={{ marginTop: 12 }}>
            {!specValidation.ok ? (
              <div style={{ marginBottom: 10, padding: 10, borderRadius: 6, background: "#fff3cd", border: "1px solid #f1e2a6", color: "#664d03" }}>
                <div style={{ fontWeight: 700, marginBottom: 4 }}>Spec minimal validation warnings</div>
                <ul style={{ margin: "6px 0 0 18px" }}>
                  {specValidation.errors.map((e, i) => (
                    <li key={i}>{e}</li>
                  ))}
                </ul>
              </div>
            ) : null}

            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
              <div style={{ border: "1px solid #eee", borderRadius: 6, padding: 10 }}>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>Counts</div>
                <div>Conductors: {Array.isArray((specObj as any).conductors) ? (specObj as any).conductors.length : 0}</div>
                <div>Dielectrics: {Array.isArray((specObj as any).dielectrics) ? (specObj as any).dielectrics.length : 0}</div>
                <div>Charges: {Array.isArray((specObj as any).charges) ? (specObj as any).charges.length : 0}</div>
              </div>

              <div style={{ border: "1px solid #eee", borderRadius: 6, padding: 10 }}>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>Domain</div>
                <div>
                  bbox:{" "}
                  <span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" }}>
                    {isRecord((specObj as any).domain) && "bbox" in ((specObj as any).domain as Record<string, unknown>)
                      ? stableStringify(((specObj as any).domain as Record<string, unknown>).bbox, 0)
                      : "—"}
                  </span>
                </div>
                <div style={{ marginTop: 6 }}>
                  BCs:{" "}
                  <span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" }}>
                    {toHumanList((specObj as any).BCs)}
                  </span>
                </div>
              </div>

              <div style={{ border: "1px solid #eee", borderRadius: 6, padding: 10 }}>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>Symmetry / queries</div>
                <div>symmetry: {toHumanList((specObj as any).symmetry)}</div>
                <div style={{ marginTop: 6 }}>queries: {toHumanList((specObj as any).queries)}</div>
              </div>

              <div style={{ border: "1px solid #eee", borderRadius: 6, padding: 10 }}>
                <div style={{ fontWeight: 700, marginBottom: 6 }}>Conductor types</div>
                {conductorTypes.length ? (
                  <ul style={{ margin: 0, paddingLeft: 18 }}>
                    {conductorTypes.map((r) => (
                      <li key={r.type}>
                        <span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" }}>{r.type}</span>: {r.count}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div style={{ color: "#666" }}>No conductors listed.</div>
                )}
              </div>
            </div>

            <div style={{ marginTop: 12, display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(340px, 1fr))", gap: 12 }}>
              <div style={{ border: "1px solid #eee", borderRadius: 6, padding: 10 }}>
                <div style={{ fontWeight: 700, marginBottom: 8 }}>Dielectrics</div>
                {dielectricRows.length ? (
                  <div style={{ overflowX: "auto" }}>
                    <table style={{ width: "100%", borderCollapse: "collapse" }}>
                      <thead>
                        <tr>
                          <th style={{ textAlign: "left", borderBottom: "1px solid #eee", padding: "6px 4px" }}>name</th>
                          <th style={{ textAlign: "right", borderBottom: "1px solid #eee", padding: "6px 4px" }}>epsilon</th>
                          <th style={{ textAlign: "right", borderBottom: "1px solid #eee", padding: "6px 4px" }}>z_min</th>
                          <th style={{ textAlign: "right", borderBottom: "1px solid #eee", padding: "6px 4px" }}>z_max</th>
                        </tr>
                      </thead>
                      <tbody>
                        {dielectricRows.map((r, idx) => (
                          <tr key={`${r.name}-${idx}`}>
                            <td style={{ padding: "6px 4px", borderBottom: "1px solid #f5f5f5" }}>{r.name}</td>
                            <td style={{ padding: "6px 4px", borderBottom: "1px solid #f5f5f5", textAlign: "right" }}>
                              {typeof r.epsilon === "number" ? r.epsilon : "—"}
                            </td>
                            <td style={{ padding: "6px 4px", borderBottom: "1px solid #f5f5f5", textAlign: "right" }}>
                              {typeof r.z_min === "number" ? r.z_min : "—"}
                            </td>
                            <td style={{ padding: "6px 4px", borderBottom: "1px solid #f5f5f5", textAlign: "right" }}>
                              {typeof r.z_max === "number" ? r.z_max : "—"}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                ) : (
                  <div style={{ color: "#666" }}>No dielectrics listed.</div>
                )}
              </div>

              <div style={{ border: "1px solid #eee", borderRadius: 6, padding: 10 }}>
                <div style={{ fontWeight: 700, marginBottom: 8 }}>Charges (first {chargeRows.length})</div>
                {chargeRows.length ? (
                  <ul style={{ margin: 0, paddingLeft: 18 }}>
                    {chargeRows.map((c, idx) => (
                      <li key={idx} style={{ marginBottom: 6 }}>
                        <span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" }}>{c.type}</span>{" "}
                        {typeof c.q === "number" ? <>q={c.q}</> : null}{" "}
                        {c.pos !== undefined ? (
                          <>
                            pos=
                            <span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" }}>{stableStringify(c.pos, 0)}</span>
                          </>
                        ) : null}
                        {c.meta ? (
                          <span style={{ color: "#666" }}>
                            {" "}
                            · <span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" }}>{stableStringify(c.meta, 0)}</span>
                          </span>
                        ) : null}
                      </li>
                    ))}
                  </ul>
                ) : (
                  <div style={{ color: "#666" }}>No charges listed.</div>
                )}
              </div>
            </div>

            <details style={{ marginTop: 12 }}>
              <summary style={{ cursor: "pointer" }}>Raw spec JSON</summary>
              <pre style={{ ...codeStyle, marginTop: 8 }}>{specRawText || stableStringify(specObj, 2)}</pre>
            </details>
          </div>
        ) : (
          <div style={{ marginTop: 12, color: "#666" }}>
            {specLoadState === "idle"
              ? "Load the spec JSON (if available within the run directory) to view an inspector summary."
              : specLoadState === "loading"
                ? "Loading spec…"
                : "Spec not loaded."}
          </div>
        )}
      </section>

      <details style={cardStyle}>
        <summary style={{ cursor: "pointer", fontWeight: 700 }}>Raw manifest JSON</summary>
        {manifestLoading ? (
          <div style={{ marginTop: 8, color: "#555" }}>Loading…</div>
        ) : manifestError ? (
          <div role="alert" style={{ marginTop: 8, color: "#8a1212" }}>
            {manifestError}
          </div>
        ) : manifest ? (
          <div style={{ marginTop: 10 }}>
            <button
              type="button"
              onClick={() => copyText("manifest", stableStringify(manifestRef.current ?? {}, 2))}
              style={{ padding: "6px 10px", marginBottom: 8 }}
            >
              Copy manifest
            </button>
            <pre style={codeStyle}>{stableStringify(manifest, 2)}</pre>
          </div>
        ) : (
          <div style={{ marginTop: 8, color: "#555" }}>No manifest available.</div>
        )}
      </details>
    </div>
  );
}
