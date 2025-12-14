import React, { useMemo, useState } from "react";
import ConfigEditor from "./ConfigEditor";

// Source notes:
// - Design Doc: FR-1 (launch workflows), FR-2 (validation + Explain exact CLI command + advanced args), FR-3 (run_dir contract; logs/metrics/artifacts recorded).
// - Repo: electrodrive/fmm3d/sanity_suite.py (CLI args: --device, --dtype, --n-points, --tol-p2p, --tol-fmm, --tol-bem, --jsonl).
// - Repo: electrodrive/fmm3d/logging_utils.py (JSONL control via EDE_FMM_ENABLE_JSONL and output path via EDE_FMM_JSONL_PATH).
// - Repo: electrodrive/live/controls.py (run_dir context; include EDE_RUN_DIR env placeholder).

const OBJECT_SCHEMA = Object.freeze({ type: "object" });

type WorkflowId = "solve" | "images_discover" | "learn_train" | "fmm_suite";

interface PresetRef {
  id: string;
  name: string;
  workflow: WorkflowId;
}

interface LaunchRequest {
  workflow: WorkflowId;
  runsRoot: string;
  runName?: string;
  specPath?: string;
  configJson?: unknown;
  configPath?: string;
  argv: string[];
  env?: Record<string, string>;
  extraArgs?: string;
}

interface WorkflowFormProps {
  defaultRunsRoot?: string;
  presets?: PresetRef[];
  onLaunch: (req: LaunchRequest) => void | Promise<void>;
  onSavePreset?: (presetName: string, req: Omit<LaunchRequest, "argv"> & { argvPreview?: string[] }) => void | Promise<void>;
  onLoadPreset?: (presetId: string) => Promise<Partial<LaunchRequest>> | Partial<LaunchRequest>;
  disabled?: boolean;
}

type ShellSplitResult = { args: string[]; error?: string };

function shellSplit(input: string): ShellSplitResult {
  const out: string[] = [];
  let cur = "";
  let quote: "'" | '"' | null = null;

  const pushCur = () => {
    if (cur.length > 0) out.push(cur);
    cur = "";
  };

  for (let i = 0; i < input.length; i++) {
    const ch = input[i];
    if (quote) {
      if (ch === quote) {
        quote = null;
        continue;
      }
      if (ch === "\\" && quote === '"' && i + 1 < input.length) {
        cur += input[i + 1];
        i++;
        continue;
      }
      cur += ch;
      continue;
    }

    if (ch === "'" || ch === '"') {
      quote = ch;
      continue;
    }
    if (/\s/.test(ch)) {
      pushCur();
      while (i + 1 < input.length && /\s/.test(input[i + 1])) i++;
      continue;
    }
    if (ch === "\\" && i + 1 < input.length) {
      cur += input[i + 1];
      i++;
      continue;
    }
    cur += ch;
  }

  if (quote) return { args: out, error: "Unclosed quote in Advanced CLI args." };
  pushCur();
  return { args: out };
}

function quoteArg(a: string): string {
  if (a.length === 0) return '""';
  if (/[\s"'\\]/.test(a)) return `"${a.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
  return a;
}

function argvToPretty(argv: string[]): string {
  return argv.map(quoteArg).join(" ");
}

function toIntOrNull(s: string): number | null {
  if (s.trim() === "") return null;
  const n = Number(s);
  return Number.isFinite(n) && Number.isInteger(n) ? n : null;
}

function toNumOrNull(s: string): number | null {
  if (s.trim() === "") return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

function isEmptyOptionalConfigText(text: string): boolean {
  const t = text.trim();
  return t === "" || t === "{}" || t === "null";
}

export default function WorkflowFormFmmSuite(props: WorkflowFormProps) {
  const disabled = Boolean(props.disabled);

  const [runsRoot, setRunsRoot] = useState<string>(props.defaultRunsRoot ?? "./runs");
  const [runName, setRunName] = useState<string>("");

  const [presetId, setPresetId] = useState<string>("");
  const [presetName, setPresetName] = useState<string>("");

  // Repo defaults (electrodrive/fmm3d/sanity_suite.py)
  const [device, setDevice] = useState<string>("cpu");
  const [dtype, setDtype] = useState<string>("float64");
  const [nPoints, setNPoints] = useState<string>("2048");
  const [tolP2p, setTolP2p] = useState<string>("1e-9");
  const [tolFmm, setTolFmm] = useState<string>("1e-6");
  const [tolBem, setTolBem] = useState<string>("1e-4");
  const [enableJsonl, setEnableJsonl] = useState<boolean>(true);

  const [extraArgs, setExtraArgs] = useState<string>("");

  // Full override for argv (design-doc hardening: CLI flags may vary; this bypasses mismatches).
  const [fullCommandOverride, setFullCommandOverride] = useState<string>("");

  // Optional inline JSON (stored by backend into manifest; not consumed by fmm suite CLI)
  const [configJson, setConfigJson] = useState<unknown>({});
  const [configMeta, setConfigMeta] = useState<{ isValid: boolean; text: string; parseError?: string; validationErrors?: string[] }>({
    isValid: true,
    text: "",
  });

  const presetOptions = (props.presets ?? []).filter((p) => p.workflow === "fmm_suite");
  const extraSplit = useMemo(() => shellSplit(extraArgs), [extraArgs]);

  const computed = useMemo(() => {
    const errors: Record<string, string> = {};
    const hasOverride = fullCommandOverride.trim().length > 0;

    if (!runsRoot.trim()) errors.runsRoot = "Runs root is required.";
    if (!configMeta.isValid) errors.configJson = configMeta.parseError ?? "Config JSON is invalid.";

    // Always include run_dir context for downstream tooling.
    const env: Record<string, string> = { EDE_RUN_DIR: "${RUN_DIR}" };

    if (enableJsonl) {
      // Repo: electrodrive/fmm3d/logging_utils.py uses EDE_FMM_ENABLE_JSONL and EDE_FMM_JSONL_PATH for output location.
      env.EDE_FMM_ENABLE_JSONL = "1";
      env.EDE_FMM_JSONL_PATH = "${RUN_DIR}/events.jsonl";
    }

    if (hasOverride) {
      const overrideSplit = shellSplit(fullCommandOverride);

      if (overrideSplit.error) errors.fullCommandOverride = overrideSplit.error;
      if (overrideSplit.args.length === 0) errors.fullCommandOverride = "Override command is empty.";

      const hasRunDirToken = overrideSplit.args.some((a) => a.includes("${RUN_DIR}"));
      if (!hasRunDirToken) errors.fullCommandOverride = "Override must include ${RUN_DIR} so the backend can substitute the run directory.";

      const canLaunch = Object.keys(errors).length === 0 && !disabled;

      return { errors, argv: overrideSplit.args, env, canLaunch, argvPretty: argvToPretty(overrideSplit.args) };
    }

    const nPointsNum = toIntOrNull(nPoints);
    if (nPointsNum === null || nPointsNum <= 0) errors.nPoints = "n-points must be a positive integer.";

    const tolP2pNum = toNumOrNull(tolP2p);
    const tolFmmNum = toNumOrNull(tolFmm);
    const tolBemNum = toNumOrNull(tolBem);
    if (tolP2pNum === null || tolP2pNum <= 0) errors.tolP2p = "tol-p2p must be a positive number.";
    if (tolFmmNum === null || tolFmmNum <= 0) errors.tolFmm = "tol-fmm must be a positive number.";
    if (tolBemNum === null || tolBemNum <= 0) errors.tolBem = "tol-bem must be a positive number.";

    if (!device.trim()) errors.device = "device is required.";
    if (!dtype.trim()) errors.dtype = "dtype is required.";

    if (extraSplit.error) errors.extraArgs = extraSplit.error;

    // Build argv aligned to electrodrive/fmm3d/sanity_suite.py
    const argv: string[] = [
      "python",
      "-m",
      "electrodrive.fmm3d.sanity_suite",
      "--device",
      device.trim(),
      "--dtype",
      dtype.trim(),
      "--n-points",
      String(nPointsNum ?? 2048),
      "--tol-p2p",
      String(tolP2pNum ?? 1e-9),
      "--tol-fmm",
      String(tolFmmNum ?? 1e-6),
      "--tol-bem",
      String(tolBemNum ?? 1e-4),
    ];

    if (enableJsonl) argv.push("--jsonl");
    if (extraSplit.args.length > 0) argv.push(...extraSplit.args);

    const canLaunch = Object.keys(errors).length === 0 && !disabled;

    return { errors, argv, env, canLaunch, argvPretty: argvToPretty(argv) };
  }, [
    configMeta.isValid,
    configMeta.parseError,
    device,
    disabled,
    dtype,
    enableJsonl,
    extraSplit.args,
    extraSplit.error,
    fullCommandOverride,
    nPoints,
    runsRoot,
    tolBem,
    tolFmm,
    tolP2p,
  ]);

  const onLoadPreset = async () => {
    if (!props.onLoadPreset || !presetId) return;
    const patch = await props.onLoadPreset(presetId);

    if (patch.runsRoot !== undefined) setRunsRoot(String(patch.runsRoot));
    if (patch.runName !== undefined) setRunName(String(patch.runName ?? ""));
    if (patch.configJson !== undefined) setConfigJson(patch.configJson);
    if (patch.extraArgs !== undefined) setExtraArgs(String(patch.extraArgs ?? ""));
    if (patch.argv && Array.isArray(patch.argv) && patch.argv.length > 0) {
      setFullCommandOverride(argvToPretty(patch.argv));
    }
  };

  const onSavePreset = async () => {
    if (!props.onSavePreset) return;
    const name = presetName.trim();
    if (!name) return;

    const reqBase: Omit<LaunchRequest, "argv"> & { argvPreview?: string[] } = {
      workflow: "fmm_suite",
      runsRoot: runsRoot.trim(),
      runName: runName.trim() || undefined,
      configJson: configMeta.isValid && !isEmptyOptionalConfigText(configMeta.text) ? configJson : undefined,
      argvPreview: computed.argv,
      env: computed.env,
      extraArgs: extraArgs.trim() || undefined,
    };

    await props.onSavePreset(name, reqBase);
    setPresetName("");
  };

  const onSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!computed.canLaunch) return;

    const req: LaunchRequest = {
      workflow: "fmm_suite",
      runsRoot: runsRoot.trim(),
      runName: runName.trim() || undefined,
      configJson: configMeta.isValid && !isEmptyOptionalConfigText(configMeta.text) ? configJson : undefined,
      argv: computed.argv,
      env: computed.env,
      extraArgs: extraArgs.trim() || undefined,
    };

    await props.onLaunch(req);
  };

  const overrideActive = fullCommandOverride.trim().length > 0;

  return (
    <form onSubmit={onSubmit} style={{ display: "grid", gap: 14, maxWidth: 980 }}>
      <h2 style={{ margin: 0 }}>FMM Sanity Suite (electrodrive.fmm3d.sanity_suite)</h2>

      <Section title="Run target">
        <Field label="Runs root (backend creates run_dir here)" error={computed.errors.runsRoot}>
          <input value={runsRoot} onChange={(e) => setRunsRoot(e.target.value)} disabled={disabled} style={inputStyle(Boolean(computed.errors.runsRoot))} placeholder="./runs" />
        </Field>

        <Field label="Run name (optional override)" hint="If empty, backend can generate a timestamp/UUID name.">
          <input value={runName} onChange={(e) => setRunName(e.target.value)} disabled={disabled} style={inputStyle(false)} placeholder="optional-name" />
        </Field>

        {props.presets ? (
          <div style={{ display: "grid", gap: 8 }}>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "end" }}>
              <Field label="Preset">
                <select value={presetId} onChange={(e) => setPresetId(e.target.value)} disabled={disabled} style={inputStyle(false)}>
                  <option value="">(none)</option>
                  {presetOptions.map((p) => (
                    <option key={p.id} value={p.id}>
                      {p.name}
                    </option>
                  ))}
                </select>
              </Field>

              <button type="button" onClick={onLoadPreset} disabled={disabled || !presetId || !props.onLoadPreset} style={buttonStyle()}>
                Load
              </button>

              <div style={{ flex: 1 }} />

              <Field label="Save as preset name">
                <input value={presetName} onChange={(e) => setPresetName(e.target.value)} disabled={disabled || !props.onSavePreset} style={inputStyle(false)} placeholder="my-fmm-preset" />
              </Field>
              <button type="button" onClick={onSavePreset} disabled={disabled || !props.onSavePreset || presetName.trim().length === 0} style={buttonStyle()}>
                Save
              </button>
            </div>

            <div style={{ fontSize: 12, color: "#444" }}>
              Presets are workflow-scoped (FR-2). The FMM suite supports JSONL logging when <code>--jsonl</code> is passed and uses <code>EDE_FMM_ENABLE_JSONL</code>{" "}
              and <code>EDE_FMM_JSONL_PATH</code> for output path (repo: electrodrive/fmm3d/sanity_suite.py + electrodrive/fmm3d/logging_utils.py).
            </div>
          </div>
        ) : null}
      </Section>

      <Section title="Suite options (from repo CLI)">
        <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
          <Field label="Device (--device)" error={computed.errors.device}>
            <input value={device} onChange={(e) => setDevice(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.device))} placeholder="cpu" />
          </Field>

          <Field label="Dtype (--dtype)" error={computed.errors.dtype}>
            <select value={dtype} onChange={(e) => setDtype(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.dtype))}>
              <option value="float64">float64 (default)</option>
              <option value="float32">float32</option>
            </select>
          </Field>

          <Field label="n-points (--n-points)" error={computed.errors.nPoints}>
            <input value={nPoints} onChange={(e) => setNPoints(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.nPoints))} inputMode="numeric" />
          </Field>

          <Field label="tol-p2p (--tol-p2p)" error={computed.errors.tolP2p}>
            <input value={tolP2p} onChange={(e) => setTolP2p(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.tolP2p))} inputMode="decimal" />
          </Field>

          <Field label="tol-fmm (--tol-fmm)" error={computed.errors.tolFmm}>
            <input value={tolFmm} onChange={(e) => setTolFmm(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.tolFmm))} inputMode="decimal" />
          </Field>

          <Field label="tol-bem (--tol-bem)" error={computed.errors.tolBem}>
            <input value={tolBem} onChange={(e) => setTolBem(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.tolBem))} inputMode="decimal" />
          </Field>
        </div>

        <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
          <Checkbox label="Enable JSONL emission (--jsonl) and write to ${RUN_DIR}/events.jsonl" checked={enableJsonl} onChange={setEnableJsonl} disabled={disabled} />
        </div>
      </Section>

      <Section title="Inline JSON config (optional)">
        <div style={{ fontSize: 12, color: "#444" }}>
          This inline JSON is validated client-side (FR-2) and intended to be saved into the run manifest by the backend (FR-3). The repo FMM suite CLI primarily
          accepts flags.
        </div>

        <ConfigEditor
          label="Config JSON (stored in manifest.json)"
          value={configJson}
          schema={OBJECT_SCHEMA}
          required={false}
          height={170}
          onChange={(val, meta) => {
            setConfigJson(val ?? {});
            setConfigMeta({ isValid: meta.isValid, text: meta.text, parseError: meta.parseError, validationErrors: meta.validationErrors });
          }}
        />

        {computed.errors.configJson ? <div style={{ color: "#b00020", fontSize: 12 }}>{computed.errors.configJson}</div> : null}
      </Section>

      <Section title="Full command override (advanced)">
        <div style={{ fontSize: 12, color: "#444" }}>
          This bypasses the form’s argv builder (useful if repo CLI flags differ from what the UI assumes). If set, this becomes the exact argv executed. Include{" "}
          <code>${"{RUN_DIR}"}</code> so the backend can substitute the run directory and preserve the artifact contract.
        </div>

        <Field label="Command override (replaces argv entirely)" error={(computed.errors as any).fullCommandOverride}>
          <input
            value={fullCommandOverride}
            onChange={(e) => setFullCommandOverride(e.target.value)}
            disabled={disabled}
            style={inputStyle(Boolean((computed.errors as any).fullCommandOverride))}
            placeholder='python -m electrodrive.fmm3d.sanity_suite --device cpu --dtype float64 --n-points 2048 --jsonl'
          />
        </Field>
      </Section>

      <Section title="Advanced CLI args (escape hatch)">
        <Field label="Advanced CLI args (appended to argv)" error={computed.errors.extraArgs} hint={overrideActive ? "Disabled because a full command override is active." : "Shell-like splitting with quotes (FR-2)."}>
          <input value={extraArgs} onChange={(e) => setExtraArgs(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.extraArgs))} placeholder="e.g. --device cuda:0" />
        </Field>
      </Section>

      <Section title="Explain / Command preview (reproducibility)">
        <div style={{ fontSize: 12, color: "#444" }}>
          Per FR-2, this is the exact argv that would be executed. The backend should replace <code>${"{RUN_DIR}"}</code> and record <code>command.txt</code> and
          <code>manifest.json</code> (FR-3).
        </div>

        <div style={{ display: "grid", gap: 10 }}>
          <CodeBlock title="argv[]">{safeJson(computed.argv)}</CodeBlock>
          <CodeBlock title="command">{computed.argvPretty}</CodeBlock>
          <CodeBlock title="env (preview)">{safeJson(computed.env)}</CodeBlock>
        </div>
      </Section>

      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <button type="submit" disabled={!computed.canLaunch} style={primaryButtonStyle(computed.canLaunch)}>
          Launch FMM Suite Run
        </button>
        {!computed.canLaunch ? <span style={{ fontSize: 12, color: "#444" }}>Fix validation errors above to enable launch (FR-2).</span> : null}
      </div>
    </form>
  );
}

function safeJson(v: unknown): string {
  try {
    return JSON.stringify(v, null, 2);
  } catch {
    return String(v);
  }
}

function inputStyle(hasError: boolean): React.CSSProperties {
  return {
    width: "100%",
    padding: "6px 8px",
    borderRadius: 6,
    border: hasError ? "1px solid #b00020" : "1px solid #bbb",
    fontSize: 14,
    background: "#fff",
  };
}

function buttonStyle(): React.CSSProperties {
  return {
    padding: "7px 10px",
    borderRadius: 6,
    border: "1px solid #bbb",
    background: "#fff",
    cursor: "pointer",
  };
}

function primaryButtonStyle(enabled: boolean): React.CSSProperties {
  return {
    padding: "9px 12px",
    borderRadius: 8,
    border: "1px solid #111",
    background: enabled ? "#111" : "#777",
    color: "#fff",
    cursor: enabled ? "pointer" : "not-allowed",
    fontWeight: 600,
  };
}

function Section(props: { title: string; children: React.ReactNode }) {
  return (
    <section style={{ border: "1px solid #ddd", borderRadius: 10, padding: 12, background: "#fafafa" }}>
      <h3 style={{ margin: "0 0 10px 0", fontSize: 16 }}>{props.title}</h3>
      <div style={{ display: "grid", gap: 10 }}>{props.children}</div>
    </section>
  );
}

function Field(props: { label: string; children: React.ReactNode; error?: string; hint?: string }) {
  const id = React.useId();
  const errId = `${id}-err`;
  const hintId = `${id}-hint`;
  const describedBy = props.error ? `${hintId} ${errId}` : hintId;

  return (
    <div style={{ display: "grid", gap: 4 }}>
      <label htmlFor={id} style={{ fontSize: 13, fontWeight: 600 }}>
        {props.label}
      </label>
      <div aria-describedby={describedBy}>
        {React.isValidElement(props.children) ? React.cloneElement(props.children as any, { id, "aria-describedby": describedBy }) : props.children}
      </div>
      {props.hint ? (
        <div id={hintId} style={{ fontSize: 12, color: "#444" }}>
          {props.hint}
        </div>
      ) : (
        <div id={hintId} style={{ display: "none" }} />
      )}
      {props.error ? (
        <div id={errId} role="alert" style={{ fontSize: 12, color: "#b00020" }}>
          {props.error}
        </div>
      ) : null}
    </div>
  );
}

function Checkbox(props: { label: string; checked: boolean; onChange: (v: boolean) => void; disabled?: boolean }) {
  const id = React.useId();
  return (
    <label htmlFor={id} style={{ display: "flex", gap: 8, alignItems: "center", fontSize: 13 }}>
      <input id={id} type="checkbox" checked={props.checked} onChange={(e) => props.onChange(e.target.checked)} disabled={props.disabled} style={{ width: 16, height: 16 }} />
      <span>{props.label}</span>
    </label>
  );
}

function CodeBlock(props: { title: string; children: string }) {
  const onCopy = async () => {
    try {
      await navigator.clipboard.writeText(props.children);
    } catch {
      // ignore
    }
  };

  return (
    <div style={{ border: "1px solid #ddd", borderRadius: 8, overflow: "hidden", background: "#fff" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "6px 10px", borderBottom: "1px solid #eee" }}>
        <div style={{ fontSize: 12, fontWeight: 600, color: "#333" }}>{props.title}</div>
        <button type="button" onClick={onCopy} style={{ ...buttonStyle(), padding: "4px 8px", fontSize: 12 }}>
          Copy
        </button>
      </div>
      <pre
        style={{
          margin: 0,
          padding: 10,
          fontSize: 12,
          overflowX: "auto",
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
        }}
      >
        {props.children}
      </pre>
    </div>
  );
}
