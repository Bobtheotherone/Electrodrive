import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";

/**
 * Run Launcher page (Design Doc §8 day-one #2; FR-1 + FR-2).
 *
 * Design Doc requirements implemented here:
 * - §8 "Run Launcher": start runs and navigate to Live Monitor.
 * - FR-1: workflow discovery/launch supports solve/images_discover/learn_train/fmm_suite.
 * - FR-2: config editor + validation + presets + "Explain exact CLI command" panel.
 *
 * Repo anchors inspected (to keep UI aligned with current Electrodrive behavior):
 * - electrodrive/researched/workflows/__init__.py registers workflows: solve/images_discover/learn_train/fmm_suite.
 * - electrodrive/researched/workflows/solve.py builds CLI: python -m electrodrive.cli solve ... --out OUT (supports controls).
 * - electrodrive/researched/workflows/images_discover.py builds CLI: python -m electrodrive.tools.images_discover discover --spec ... --out ...
 * - electrodrive/researched/workflows/learn_train.py requires config_path (YAML); rejects inline dict config.
 * - electrodrive/researched/workflows/fmm_suite.py builds CLI: python -m electrodrive.fmm3d.sanity_suite; JSONL via env vars.
 * - electrodrive/researched/app.py mounts REST at /api/v1 and indicates launch is POST /api/v1/runs.
 */

type Workflow = "solve" | "images_discover" | "learn_train" | "fmm_suite";
type LaunchResponse = { runId?: string; run_id?: string; id?: string; [k: string]: unknown };
type Preset = {
  name: string;
  workflow: Workflow;
  specPath: string;
  requestJson: string; // advanced request overrides JSON object
  argvOverride: string; // optional list[str] JSON or shell-like string (used as extra_args)
};

const LS_PRESETS = "researched.presets.local.v1";

// Default endpoints (prompt); we also adapt to repo (/api/v1/runs).
const DEFAULT_REST_PREFIX = "/api";
const REST_PREFIX_CANDIDATES = [
  (import.meta.env.VITE_API_BASE as string | undefined) ?? "",
  "/api/v1", // repo
  DEFAULT_REST_PREFIX,
].map((s) => String(s || "").trim()).filter(Boolean);

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

function safeParseJson(text: string): { ok: true; value: unknown } | { ok: false; error: string } {
  const s = (text || "").trim();
  if (!s) return { ok: true, value: null };
  try { return { ok: true, value: JSON.parse(s) }; } catch (e) { return { ok: false, error: toErrorMessage(e) }; }
}

async function fetchJsonWithFallback<T>(
  pathCandidates: string[],
  init: RequestInit = {},
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
        continue;
      }
    }
  }

  throw lastErr ?? new Error("Request failed");
}

function loadLocalPresets(): Preset[] {
  try {
    const raw = localStorage.getItem(LS_PRESETS);
    if (!raw) return [];
    const parsed = safeParseJson(raw);
    if (!parsed.ok || !Array.isArray(parsed.value)) return [];
    return parsed.value
      .map((x) => x as any)
      .filter((p) => p && typeof p.name === "string" && typeof p.workflow === "string")
      .map((p) => ({
        name: String(p.name),
        workflow: p.workflow as Workflow,
        specPath: String(p.specPath ?? ""),
        requestJson: String(p.requestJson ?? ""),
        argvOverride: String(p.argvOverride ?? ""),
      }));
  } catch {
    return [];
  }
}

function saveLocalPresets(presets: Preset[]) {
  try {
    localStorage.setItem(LS_PRESETS, JSON.stringify(presets));
  } catch {
    // ignore
  }
}

function splitArgv(input: string): string[] {
  const s = (input || "").trim();
  if (!s) return [];
  if (s.startsWith("[")) {
    const parsed = safeParseJson(s);
    if (parsed.ok && Array.isArray(parsed.value)) return (parsed.value as unknown[]).map((x) => String(x));
  }
  // Minimal shell-like splitting: whitespace only (good enough for a scaffold).
  return s.split(/\s+/).filter(Boolean);
}

function describeWorkflowInput(workflow: Workflow): { label: string; placeholder: string; keyHint: string } {
  if (workflow === "learn_train") {
    // Repo: LearnTrainWorkflow requires config_path YAML (electrodrive/researched/workflows/learn_train.py).
    return { label: "Config path (YAML)", placeholder: "e.g. configs/train_example.yaml", keyHint: "config_path" };
  }
  if (workflow === "fmm_suite") {
    return { label: "Optional spec/config path", placeholder: "(none required for fmm_suite)", keyHint: "(optional)" };
  }
  return { label: "Spec path", placeholder: "e.g. specs/plane_point.json", keyHint: "spec_path" };
}

function buildExplainPanel(workflow: Workflow, specPath: string, requestOverrides: unknown, extraArgs: string[]): { cmd: string; env: Record<string, string> } {
  // This is a best-effort "Explain" panel (FR-2) aligned to repo workflows:
  // - solve: electrodrive/researched/workflows/solve.py builds: python -m electrodrive.cli solve ... --out OUT
  // - images_discover: electrodrive/researched/workflows/images_discover.py builds: python -m electrodrive.tools.images_discover discover --spec ... --out OUT
  // - learn_train: electrodrive/researched/workflows/learn_train.py builds: python -m electrodrive.cli train --config ... --out OUT
  // - fmm_suite: electrodrive/researched/workflows/fmm_suite.py builds: python -m electrodrive.fmm3d.sanity_suite (JSONL via env vars)
  const OUT = "<RUN_DIR>";
  const PY = "python";
  const rid = "<RUN_ID>";

  const envBase: Record<string, string> = {
    EDE_RUN_DIR: OUT,
    EDE_RUN_ID: rid,
    PYTHONUNBUFFERED: "1",
  };

  const extra = extraArgs.length ? ` ${extraArgs.map((x) => JSON.stringify(x)).join(" ")}` : "";

  if (workflow === "solve") {
    const spec = specPath ? ` --problem ${JSON.stringify(specPath)}` : " --problem <SPEC_PATH>";
    return {
      cmd: `${PY} -m electrodrive.cli solve${spec} --mode auto --out ${JSON.stringify(OUT)}${extra}`,
      env: envBase,
    };
  }

  if (workflow === "images_discover") {
    const spec = specPath ? ` --spec ${JSON.stringify(specPath)}` : " --spec <SPEC_PATH>";
    return {
      cmd: `${PY} -m electrodrive.tools.images_discover discover${spec} --out ${JSON.stringify(OUT)}${extra}`,
      env: envBase,
    };
  }

  if (workflow === "learn_train") {
    const cfg = specPath ? ` --config ${JSON.stringify(specPath)}` : " --config <CONFIG_YAML>";
    return {
      cmd: `${PY} -m electrodrive.cli train${cfg} --out ${JSON.stringify(OUT)}${extra}`,
      env: envBase,
    };
  }

  // fmm_suite
  return {
    cmd: `${PY} -m electrodrive.fmm3d.sanity_suite${extra}`,
    env: {
      ...envBase,
      // Repo: fmm_suite enables JSONL via env vars (electrodrive/researched/workflows/fmm_suite.py).
      EDE_FMM_ENABLE_JSONL: "1",
      EDE_FMM_JSONL_PATH: `${OUT}/events.jsonl`,
      EDE_FMM_JSONL_NO_STDOUT: "1",
    },
  };
}

export default function RunLauncher() {
  const nav = useNavigate();

  const [workflow, setWorkflow] = useState<Workflow>("solve");
  const [specPath, setSpecPath] = useState<string>("");
  const [requestJson, setRequestJson] = useState<string>("{}");
  const [argvOverride, setArgvOverride] = useState<string>("");

  const [presetName, setPresetName] = useState<string>("");
  const [presets, setPresets] = useState<Preset[]>(() => loadLocalPresets());
  const [selectedPreset, setSelectedPreset] = useState<string>("");

  const [busy, setBusy] = useState(false);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    saveLocalPresets(presets);
  }, [presets]);

  // When selecting a preset, apply it.
  useEffect(() => {
    if (!selectedPreset) return;
    const p = presets.find((x) => x.name === selectedPreset);
    if (!p) return;
    setWorkflow(p.workflow);
    setSpecPath(p.specPath);
    setRequestJson(p.requestJson || "{}");
    setArgvOverride(p.argvOverride || "");
  }, [selectedPreset, presets]);

  const parsedRequest = useMemo(() => safeParseJson(requestJson), [requestJson]);
  const requestOverrides = parsedRequest.ok ? parsedRequest.value : null;

  const extraArgs = useMemo(() => splitArgv(argvOverride), [argvOverride]);

  const inputDesc = useMemo(() => describeWorkflowInput(workflow), [workflow]);

  const explain = useMemo(() => buildExplainPanel(workflow, specPath, requestOverrides, extraArgs), [workflow, specPath, requestOverrides, extraArgs]);

  const validationError = useMemo(() => {
    // FR-2: validation guardrails before launch.
    if (!parsedRequest.ok) return `Invalid JSON in request overrides: ${parsedRequest.error}`;

    if (parsedRequest.ok) {
      const v = parsedRequest.value;
      if (v !== null && (typeof v !== "object" || Array.isArray(v))) {
        return "Advanced request overrides must be a JSON object (e.g. {}).";
      }
    }

    if (workflow !== "fmm_suite") {
      // Solve/images_discover require spec_path; learn_train requires config_path (repo).
      if (!specPath.trim()) return `${inputDesc.label} is required for ${workflow}.`;
    }
    // For learn_train, repo rejects inline dict config; we keep UI flexible but warn early.
    if (workflow === "learn_train") {
      if (requestOverrides && typeof requestOverrides === "object" && requestOverrides !== null) {
        const cfg = (requestOverrides as any).config;
        if (cfg && typeof cfg === "object") {
          return "learn_train: backend expects a YAML config path (config_path), not an inline config dict (repo: electrodrive/researched/workflows/learn_train.py).";
        }
      }
    }
    return null;
  }, [parsedRequest, workflow, specPath, requestOverrides, inputDesc.label]);

  const savePreset = () => {
    setErr(null);
    const name = presetName.trim();
    if (!name) {
      setErr("Preset name is required.");
      return;
    }
    const next: Preset = { name, workflow, specPath, requestJson: requestJson || "{}", argvOverride: argvOverride || "" };
    setPresets((prev) => {
      const without = prev.filter((p) => p.name !== name);
      return [...without, next].sort((a, b) => a.name.localeCompare(b.name));
    });
    setSelectedPreset(name);
  };

  const deletePreset = () => {
    const name = selectedPreset.trim();
    if (!name) return;
    setPresets((prev) => prev.filter((p) => p.name !== name));
    setSelectedPreset("");
  };

  const launch = async () => {
    setErr(null);
    if (validationError) {
      setErr(validationError);
      return;
    }

    setBusy(true);
    try {
      const overrides =
        (requestOverrides && typeof requestOverrides === "object" && requestOverrides !== null && !Array.isArray(requestOverrides))
          ? (requestOverrides as Record<string, unknown>)
          : {};

      // Build the repo-shaped body ONLY (avoid extra keys → prevent 422 on strict backends).
      const repoBody: Record<string, unknown> = { ...overrides, workflow };

      if (workflow === "learn_train") {
        repoBody.config_path = specPath.trim();
      } else if (workflow === "solve" || workflow === "images_discover") {
        repoBody.spec_path = specPath.trim();
      } else {
        // fmm_suite: do NOT send spec_path unless backend explicitly supports it.
      }

      if (extraArgs.length) repoBody.extra_args = extraArgs;

      // Try repo endpoint first (POST /api/v1/runs).
      let res: LaunchResponse;
      try {
        res = await fetchJsonWithFallback<LaunchResponse>(
          [`/runs`],
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(repoBody),
          },
        );
      } catch (e) {
        // If endpoint truly doesn’t exist, fall back to legacy/prompt payload.
        const status = (e as any)?.status;
        if (status !== 404 && status !== 405) throw e;

        const legacyBody: Record<string, unknown> = {
          workflow,
          specPath: specPath.trim() || undefined,
          argvOverride: argvOverride.trim() || undefined,
          presetName: presetName.trim() || undefined,
          // Only include configInline if user actually provided non-empty overrides.
          configInline: Object.keys(overrides).length ? overrides : undefined,
        };

        res = await fetchJsonWithFallback<LaunchResponse>(
          [`/runs/launch`],
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(legacyBody),
          },
        );
      }

      const runId = String(res.runId || res.run_id || res.id || "").trim();
      if (!runId) throw new Error("Launch succeeded but response did not include runId/run_id.");

      // Design Doc §8: navigate to Live Monitor immediately after launch.
      nav(`/runs/${encodeURIComponent(runId)}/monitor`);
    } catch (e) {
      setErr(toErrorMessage(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <div>
          <h1 style={{ margin: 0, fontSize: 18 }}>Run Launcher</h1>
          <div style={{ marginTop: 4, fontSize: 12, color: "#6b7280" }}>
            Design Doc §8 day-one • FR-1 launch • FR-2 validation + presets + explain panel
          </div>
        </div>

        <button
          type="button"
          onClick={() => nav("/runs")}
          style={{ padding: "6px 10px", borderRadius: 10, border: "1px solid #e5e7eb", background: "#fff", cursor: "pointer" }}
        >
          Back to Runs
        </button>
      </div>

      <section style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 12, background: "#fff" }}>
        <div style={{ display: "grid", gap: 12, maxWidth: 980 }}>
          <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(auto-fit, minmax(240px, 1fr))" }}>
            <label style={{ display: "grid", gap: 6 }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>Workflow (FR-1)</div>
              <select
                value={workflow}
                onChange={(e) => setWorkflow(e.target.value as Workflow)}
                style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid #e5e7eb" }}
              >
                <option value="solve">solve</option>
                <option value="images_discover">images_discover</option>
                <option value="learn_train">learn_train</option>
                <option value="fmm_suite">fmm_suite</option>
              </select>
            </label>

            <label style={{ display: "grid", gap: 6 }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>
                {inputDesc.label}{" "}
                <span style={{ color: "#9ca3af" }}>
                  (request key hint: <code>{inputDesc.keyHint}</code>)
                </span>
              </div>
              <input
                value={specPath}
                onChange={(e) => setSpecPath(e.target.value)}
                placeholder={inputDesc.placeholder}
                style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid #e5e7eb" }}
              />
            </label>

            <label style={{ display: "grid", gap: 6 }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>Preset</div>
              <select
                value={selectedPreset}
                onChange={(e) => setSelectedPreset(e.target.value)}
                style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid #e5e7eb" }}
              >
                <option value="">(none)</option>
                {presets.map((p) => (
                  <option key={p.name} value={p.name}>{p.name}</option>
                ))}
              </select>
            </label>
          </div>

          <div style={{ display: "grid", gap: 10 }}>
            <label style={{ display: "grid", gap: 6 }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>
                Advanced request overrides (JSON object) — validated (FR-2)
              </div>
              <textarea
                value={requestJson}
                onChange={(e) => setRequestJson(e.target.value)}
                rows={8}
                spellCheck={false}
                style={{
                  padding: "8px 10px",
                  borderRadius: 10,
                  border: `1px solid ${parsedRequest.ok ? "#e5e7eb" : "#b91c1c"}`,
                  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                  fontSize: 12,
                }}
                placeholder='e.g. {"mode":"auto","viz_enable":true,"extra_args":["--fast"]}'
              />
              {!parsedRequest.ok ? (
                <div style={{ fontSize: 12, color: "#b91c1c" }}>{parsedRequest.error}</div>
              ) : (
                <div style={{ fontSize: 12, color: "#6b7280" }}>
                  Tip: put power-user flags in <code>extra_args</code> (repo: electrodrive/researched/workflows/*).
                </div>
              )}
            </label>

            <label style={{ display: "grid", gap: 6 }}>
              <div style={{ fontSize: 12, color: "#6b7280" }}>
                argv override (extra args) — JSON array or whitespace-separated (FR-2)
              </div>
              <textarea
                value={argvOverride}
                onChange={(e) => setArgvOverride(e.target.value)}
                rows={3}
                spellCheck={false}
                style={{
                  padding: "8px 10px",
                  borderRadius: 10,
                  border: "1px solid #e5e7eb",
                  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                  fontSize: 12,
                }}
                placeholder='e.g. ["--fast","--cert"] or --fast --cert'
              />
            </label>
          </div>

          <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))" }}>
            <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 10 }}>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>Presets (FR-2)</div>
              <div style={{ display: "grid", gap: 8 }}>
                <input
                  value={presetName}
                  onChange={(e) => setPresetName(e.target.value)}
                  placeholder="Preset name (localStorage fallback)"
                  style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid #e5e7eb" }}
                />
                <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                  <button
                    type="button"
                    onClick={savePreset}
                    style={{ padding: "6px 10px", borderRadius: 10, border: "1px solid #e5e7eb", background: "#fff", cursor: "pointer" }}
                  >
                    Save preset
                  </button>
                  <button
                    type="button"
                    onClick={deletePreset}
                    disabled={!selectedPreset}
                    style={{
                      padding: "6px 10px",
                      borderRadius: 10,
                      border: "1px solid #e5e7eb",
                      background: "#fff",
                      cursor: selectedPreset ? "pointer" : "not-allowed",
                      opacity: selectedPreset ? 1 : 0.6,
                    }}
                  >
                    Delete selected
                  </button>
                </div>
                <div style={{ fontSize: 12, color: "#6b7280" }}>
                  Backend may also support presets on disk at <code>~/.researched/presets</code> (repo: electrodrive/researched/api.py, FR-2).
                  This UI keeps a localStorage fallback to stay usable if backend presets endpoints change.
                </div>
              </div>
            </div>

            <div style={{ border: "1px solid #e5e7eb", borderRadius: 12, padding: 10 }}>
              <div style={{ fontWeight: 700, marginBottom: 6 }}>Explain (FR-2)</div>
              <div style={{ fontSize: 12, color: "#6b7280", marginBottom: 8 }}>
                Command preview mirrors repo workflow builders (electrodrive/researched/workflows/*). Actual run captures command/env in run_dir/command.txt (FR-3).
              </div>
              <div style={{ display: "grid", gap: 10 }}>
                <div>
                  <div style={{ fontSize: 12, color: "#6b7280" }}>Command preview</div>
                  <pre
                    style={{
                      margin: 0,
                      padding: 10,
                      background: "#f9fafb",
                      border: "1px solid #e5e7eb",
                      borderRadius: 10,
                      overflowX: "auto",
                      fontSize: 12,
                    }}
                  >
                    {explain.cmd}
                  </pre>
                </div>

                <div>
                  <div style={{ fontSize: 12, color: "#6b7280" }}>Environment preview</div>
                  <pre
                    style={{
                      margin: 0,
                      padding: 10,
                      background: "#f9fafb",
                      border: "1px solid #e5e7eb",
                      borderRadius: 10,
                      overflowX: "auto",
                      fontSize: 12,
                    }}
                  >
                    {Object.entries(explain.env)
                      .map(([k, v]) => `${k}=${JSON.stringify(v)}`)
                      .join("\n")}
                  </pre>
                </div>
              </div>
            </div>
          </div>

          {validationError ? <div style={{ color: "#b91c1c", fontSize: 13 }}>{validationError}</div> : null}
          {err ? <div style={{ color: "#b91c1c", fontSize: 13 }}>{err}</div> : null}

          <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
            <button
              type="button"
              onClick={() => void launch()}
              disabled={busy || !!validationError}
              style={{
                padding: "8px 12px",
                borderRadius: 10,
                border: "1px solid transparent",
                background: "#111827",
                color: "#fff",
                cursor: busy || !!validationError ? "not-allowed" : "pointer",
                opacity: busy || !!validationError ? 0.6 : 1,
              }}
            >
              {busy ? "Launching…" : "Launch run"}
            </button>

            <div style={{ fontSize: 12, color: "#6b7280" }}>
              On success, you’ll be redirected to <code>/runs/:runId/monitor</code> (Design Doc §8 Live Monitor).
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
