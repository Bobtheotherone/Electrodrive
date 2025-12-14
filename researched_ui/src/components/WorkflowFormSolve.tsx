import React, { useMemo, useState } from "react";
import ConfigEditor from "./ConfigEditor";

// Source notes:
// - Design Doc: FR-1 (launch workflows), FR-2 (validation + presets + Explain exact CLI command), FR-3 (run_dir contract + command.txt/manifest.json).
// - Repo: electrodrive/cli.py (argparse for `solve`: required --problem, required --out, --mode, --cert/--cert-strong/--cert-fast, --fast, --fail-on-backend-fallback, --viz*; global flags --seed/--amp/--no-amp/--train-dtype/--compile/--tf32).
// - Repo: electrodrive/live/controls.py (control protocol expects run_dir context; we include EDE_RUN_DIR env placeholder for later control compatibility; snapshot is string token).

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
  if (/[\s"'\\]/.test(a)) {
    // Simple double-quote escape
    return `"${a.replace(/\\/g, "\\\\").replace(/"/g, '\\"')}"`;
  }
  return a;
}

function argvToPretty(argv: string[]): string {
  return argv.map(quoteArg).join(" ");
}

function toNumberOrNull(s: string): number | null {
  if (s.trim() === "") return null;
  const n = Number(s);
  return Number.isFinite(n) ? n : null;
}

function isEmptyOptionalConfigText(text: string): boolean {
  const t = text.trim();
  return t === "" || t === "{}" || t === "null";
}

export default function WorkflowFormSolve(props: WorkflowFormProps) {
  const disabled = Boolean(props.disabled);

  const [runsRoot, setRunsRoot] = useState<string>(props.defaultRunsRoot ?? "./runs");
  const [runName, setRunName] = useState<string>("");

  const [presetId, setPresetId] = useState<string>("");
  const [presetName, setPresetName] = useState<string>("");

  // CLI-aligned fields (electrodrive/cli.py)
  const [problemPath, setProblemPath] = useState<string>("");
  const [mode, setMode] = useState<"auto" | "analytic" | "bem" | "pinn">("auto");
  const [cert, setCert] = useState<boolean>(false);
  const [certStrong, setCertStrong] = useState<boolean>(false);
  const [certFast, setCertFast] = useState<boolean>(false);
  const [fast, setFast] = useState<boolean>(false);
  const [failOnBackendFallback, setFailOnBackendFallback] = useState<boolean>(false);
  const [evalPdf, setEvalPdf] = useState<boolean>(false);
  const [evalSha256, setEvalSha256] = useState<boolean>(false);

  const [vizEnabled, setVizEnabled] = useState<boolean>(true);
  const [vizAnimate, setVizAnimate] = useState<boolean>(false);
  const [vizPlane, setVizPlane] = useState<string>("xz");
  const [vizSize, setVizSize] = useState<string>("4.0");
  const [vizRes, setVizRes] = useState<string>("200");

  // Global flags (electrodrive/cli.py)
  const [seed, setSeed] = useState<string>("0");
  const [amp, setAmp] = useState<boolean>(true);
  const [trainDtype, setTrainDtype] = useState<string>("bf16");
  const [compile, setCompile] = useState<boolean>(false);
  const [tf32, setTf32] = useState<boolean>(false);

  const [extraArgs, setExtraArgs] = useState<string>("");

  // Full override for argv (design-doc hardening: CLI flags may vary; this bypasses mismatches).
  const [fullCommandOverride, setFullCommandOverride] = useState<string>("");

  // Optional inline JSON config (stored by backend into manifest.json; not passed to electrodrive.cli solve)
  const [configJson, setConfigJson] = useState<unknown>({});
  const [configMeta, setConfigMeta] = useState<{ isValid: boolean; text: string; parseError?: string; validationErrors?: string[] }>({
    isValid: true,
    text: "",
  });

  const presetOptions = (props.presets ?? []).filter((p) => p.workflow === "solve");

  const extraSplit = useMemo(() => shellSplit(extraArgs), [extraArgs]);

  const computed = useMemo(() => {
    const errors: Record<string, string> = {};
    const hasOverride = fullCommandOverride.trim().length > 0;

    if (!runsRoot.trim()) errors.runsRoot = "Runs root is required.";
    if (!configMeta.isValid) errors.configJson = configMeta.parseError ?? "Config JSON is invalid.";

    const env: Record<string, string> = {
      // Useful for later control protocol alignment (electrodrive/live/controls.py uses run_dir context).
      EDE_RUN_DIR: "${RUN_DIR}",
    };

    if (hasOverride) {
      const overrideSplit = shellSplit(fullCommandOverride);

      if (overrideSplit.error) errors.fullCommandOverride = overrideSplit.error;
      if (overrideSplit.args.length === 0) errors.fullCommandOverride = "Override command is empty.";

      // Ensure placeholder exists so backend can substitute run_dir and preserve artifact contract.
      const hasRunDirToken = overrideSplit.args.some((a) => a.includes("${RUN_DIR}"));
      if (!hasRunDirToken) {
        errors.fullCommandOverride = "Override must include ${RUN_DIR} so the backend can substitute the run directory.";
      }

      const canLaunch = Object.keys(errors).length === 0 && !disabled;

      return {
        errors,
        argv: overrideSplit.args,
        env,
        canLaunch,
        argvPretty: argvToPretty(overrideSplit.args),
      };
    }

    // Normal structured validation/build (best effort aligned to current repo CLI)
    if (!problemPath.trim()) errors.problemPath = "Problem spec path is required (--problem).";

    const seedNum = toNumberOrNull(seed);
    if (seedNum === null || !Number.isInteger(seedNum) || seedNum < 0) errors.seed = "Seed must be a non-negative integer.";

    const vizSizeNum = toNumberOrNull(vizSize);
    if (vizSizeNum === null || vizSizeNum <= 0) errors.vizSize = "viz-size must be a positive number.";

    const vizResNum = toNumberOrNull(vizRes);
    if (vizResNum === null || !Number.isInteger(vizResNum) || vizResNum <= 0) errors.vizRes = "viz-res must be a positive integer.";

    if (extraSplit.error) errors.extraArgs = extraSplit.error;

    // Build argv aligned to electrodrive/cli.py
    const argv: string[] = ["python", "-m", "electrodrive.cli", "solve", "--problem", problemPath.trim(), "--out", "${RUN_DIR}"];

    // Global flags
    if (seedNum !== null) argv.push("--seed", String(seedNum));
    if (!amp) argv.push("--no-amp"); // default is AMP enabled in electrodrive/cli.py
    if (trainDtype.trim() && trainDtype.trim() !== "bf16") argv.push("--train-dtype", trainDtype.trim());
    if (compile) argv.push("--compile");
    if (tf32) argv.push("--tf32");

    // Solve flags
    if (mode !== "auto") argv.push("--mode", mode);
    if (cert) argv.push("--cert");
    if (certStrong) argv.push("--cert-strong");
    if (certFast) argv.push("--cert-fast");
    if (fast) argv.push("--fast");
    if (failOnBackendFallback) argv.push("--fail-on-backend-fallback");
    if (evalPdf) argv.push("--eval-pdf");
    if (evalSha256) argv.push("--eval-sha256");

    // Visualization flags
    if (vizEnabled) {
      argv.push("--viz");
      if (vizAnimate) argv.push("--viz-animate");
      if (vizPlane.trim() && vizPlane.trim() !== "xz") argv.push("--viz-plane", vizPlane.trim());
      if (vizSizeNum !== null && vizSizeNum !== 4.0) argv.push("--viz-size", String(vizSizeNum));
      if (vizResNum !== null && vizResNum !== 200) argv.push("--viz-res", String(vizResNum));
    }

    // Advanced args appended at end (FR-2 escape hatch)
    if (extraSplit.args.length > 0) argv.push(...extraSplit.args);

    const canLaunch = Object.keys(errors).length === 0 && !disabled;

    return {
      errors,
      argv,
      env,
      canLaunch,
      argvPretty: argvToPretty(argv),
    };
  }, [
    amp,
    cert,
    certFast,
    certStrong,
    compile,
    configMeta.isValid,
    configMeta.parseError,
    disabled,
    evalPdf,
    evalSha256,
    extraSplit.args,
    extraSplit.error,
    failOnBackendFallback,
    fast,
    fullCommandOverride,
    mode,
    problemPath,
    runsRoot,
    seed,
    tf32,
    trainDtype,
    vizAnimate,
    vizEnabled,
    vizPlane,
    vizRes,
    vizSize,
  ]);

  const onLoadPreset = async () => {
    if (!props.onLoadPreset || !presetId) return;
    const patch = await props.onLoadPreset(presetId);

    if (patch.runsRoot !== undefined) setRunsRoot(String(patch.runsRoot));
    if (patch.runName !== undefined) setRunName(String(patch.runName ?? ""));
    if (patch.specPath !== undefined) setProblemPath(String(patch.specPath ?? ""));
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
      workflow: "solve",
      runsRoot: runsRoot.trim(),
      runName: runName.trim() || undefined,
      specPath: problemPath.trim() || undefined,
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
      workflow: "solve",
      runsRoot: runsRoot.trim(),
      runName: runName.trim() || undefined,
      specPath: problemPath.trim() || undefined,
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
      <h2 style={{ margin: 0 }}>Solve (electrodrive.cli solve)</h2>

      <Section title="Run target">
        <Field label="Runs root (backend creates run_dir here)" error={computed.errors.runsRoot}>
          <input
            value={runsRoot}
            onChange={(e) => setRunsRoot(e.target.value)}
            disabled={disabled}
            style={inputStyle(Boolean(computed.errors.runsRoot))}
            placeholder="./runs"
          />
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
                <input value={presetName} onChange={(e) => setPresetName(e.target.value)} disabled={disabled || !props.onSavePreset} style={inputStyle(false)} placeholder="my-solve-preset" />
              </Field>
              <button type="button" onClick={onSavePreset} disabled={disabled || !props.onSavePreset || presetName.trim().length === 0} style={buttonStyle()}>
                Save
              </button>
            </div>

            <div style={{ fontSize: 12, color: "#444" }}>
              Presets are workflow-scoped (FR-2). Saved presets should include enough info for reproducible re-runs (FR-3).
            </div>
          </div>
        ) : null}
      </Section>

      <Section title="Required inputs">
        <Field label="Problem spec path (--problem)" error={computed.errors.problemPath} hint="Repo CLI requires --problem (electrodrive/cli.py).">
          <input
            value={problemPath}
            onChange={(e) => setProblemPath(e.target.value)}
            disabled={disabled}
            style={inputStyle(Boolean(computed.errors.problemPath))}
            placeholder="path/to/problem.json"
          />
        </Field>
      </Section>

      <Section title="Solve options (from repo CLI)">
        <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
          <Field label="Mode (--mode)">
            <select value={mode} onChange={(e) => setMode(e.target.value as any)} disabled={disabled || overrideActive} style={inputStyle(false)}>
              <option value="auto">auto</option>
              <option value="analytic">analytic</option>
              <option value="bem">bem</option>
              <option value="pinn">pinn</option>
            </select>
          </Field>

          <Field label="Seed (--seed)" error={computed.errors.seed}>
            <input value={seed} onChange={(e) => setSeed(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.seed))} inputMode="numeric" />
          </Field>

          <Field label="Train dtype (--train-dtype)">
            <select value={trainDtype} onChange={(e) => setTrainDtype(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(false)}>
              <option value="bf16">bf16 (default)</option>
              <option value="fp16">fp16</option>
              <option value="fp32">fp32</option>
              <option value="float64">float64</option>
            </select>
          </Field>
        </div>

        <div style={{ display: "grid", gap: 8 }}>
          <Checkbox label="AMP enabled (default true; uses --no-amp to disable)" checked={amp} onChange={setAmp} disabled={disabled || overrideActive} />
          <Checkbox label="Compile (--compile)" checked={compile} onChange={setCompile} disabled={disabled || overrideActive} />
          <Checkbox label="TF32 (--tf32)" checked={tf32} onChange={setTf32} disabled={disabled || overrideActive} />
        </div>

        <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
          <Checkbox label="Enable certification checks (--cert)" checked={cert} onChange={setCert} disabled={disabled || overrideActive} />
          <Checkbox label="Stricter certification gates (--cert-strong)" checked={certStrong} onChange={setCertStrong} disabled={disabled || overrideActive} />
          <Checkbox label="Faster certification settings (--cert-fast)" checked={certFast} onChange={setCertFast} disabled={disabled || overrideActive} />
          <Checkbox label="Fast mode (--fast)" checked={fast} onChange={setFast} disabled={disabled || overrideActive} />
          <Checkbox label="Fail on backend fallback (--fail-on-backend-fallback)" checked={failOnBackendFallback} onChange={setFailOnBackendFallback} disabled={disabled || overrideActive} />
          <Checkbox label="Evaluate PDF (--eval-pdf)" checked={evalPdf} onChange={setEvalPdf} disabled={disabled || overrideActive} />
          <Checkbox label="Compute SHA256 of outputs (--eval-sha256)" checked={evalSha256} onChange={setEvalSha256} disabled={disabled || overrideActive} />
        </div>
      </Section>

      <Section title="Visualization (from repo CLI)">
        <div style={{ display: "grid", gap: 8 }}>
          <Checkbox label="Enable visualization output (--viz)" checked={vizEnabled} onChange={setVizEnabled} disabled={disabled || overrideActive} />
          <Checkbox label="Create animation from frames (--viz-animate)" checked={vizAnimate} onChange={setVizAnimate} disabled={disabled || !vizEnabled || overrideActive} />
        </div>

        <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", marginTop: 8 }}>
          <Field label="Plane (--viz-plane)">
            <select value={vizPlane} onChange={(e) => setVizPlane(e.target.value)} disabled={disabled || !vizEnabled || overrideActive} style={inputStyle(false)}>
              <option value="xz">xz (default)</option>
              <option value="xy">xy</option>
              <option value="yz">yz</option>
            </select>
          </Field>

          <Field label="Domain size (--viz-size)" error={computed.errors.vizSize}>
            <input value={vizSize} onChange={(e) => setVizSize(e.target.value)} disabled={disabled || !vizEnabled || overrideActive} style={inputStyle(Boolean(computed.errors.vizSize))} inputMode="decimal" />
          </Field>

          <Field label="Resolution (--viz-res)" error={computed.errors.vizRes}>
            <input value={vizRes} onChange={(e) => setVizRes(e.target.value)} disabled={disabled || !vizEnabled || overrideActive} style={inputStyle(Boolean(computed.errors.vizRes))} inputMode="numeric" />
          </Field>
        </div>
      </Section>

      <Section title="Inline JSON config (optional)">
        <div style={{ fontSize: 12, color: "#444" }}>
          This inline JSON is validated client-side (FR-2) and is intended to be saved into the run manifest by the backend (FR-3). It is <em>not</em> passed
          directly to <code>electrodrive.cli solve</code> because the repo CLI does not accept a config flag for solve.
        </div>

        <ConfigEditor
          label="Config JSON (stored in manifest.json)"
          value={configJson}
          schema={OBJECT_SCHEMA}
          required={false}
          height={200}
          onChange={(val, meta) => {
            setConfigJson(val ?? {});
            setConfigMeta({
              isValid: meta.isValid,
              text: meta.text,
              parseError: meta.parseError,
              validationErrors: meta.validationErrors,
            });
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
            placeholder='python -m electrodrive.cli solve --problem spec.json --out ${RUN_DIR} --viz'
          />
        </Field>
      </Section>

      <Section title="Advanced CLI args (escape hatch)">
        <Field
          label="Advanced CLI args (appended to argv)"
          error={computed.errors.extraArgs}
          hint={overrideActive ? "Disabled because a full command override is active." : 'Shell-like splitting with quotes. Use for flags not exposed in the form (FR-2).'}
        >
          <input
            value={extraArgs}
            onChange={(e) => setExtraArgs(e.target.value)}
            disabled={disabled || overrideActive}
            style={inputStyle(Boolean(computed.errors.extraArgs))}
            placeholder='e.g. --viz-plane xy --viz-res 300'
          />
        </Field>
      </Section>

      <Section title="Explain / Command preview (reproducibility)">
        <div style={{ fontSize: 12, color: "#444" }}>
          Per FR-2, this is the exact argv that would be executed. The backend should replace <code>${"{RUN_DIR}"}</code> when creating the run directory and record
          both <code>command.txt</code> and <code>manifest.json</code> (FR-3).
        </div>

        <div style={{ display: "grid", gap: 10 }}>
          <CodeBlock title="argv[]">{safeJson(computed.argv)}</CodeBlock>
          <CodeBlock title="command">{computed.argvPretty}</CodeBlock>
          <CodeBlock title="env (preview)">{safeJson(computed.env)}</CodeBlock>
        </div>
      </Section>

      <div style={{ display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
        <button type="submit" disabled={!computed.canLaunch} style={primaryButtonStyle(computed.canLaunch)}>
          Launch Solve Run
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
      // ignore (clipboard may be unavailable)
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
