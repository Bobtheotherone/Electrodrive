import React, { useMemo, useState } from "react";
import ConfigEditor from "./ConfigEditor";

// Source notes:
// - Design Doc: FR-1 (launch workflows), FR-2 (validation + presets + Explain exact CLI command + config editor), FR-3 (run_dir contract + command record).
// - Repo: electrodrive/learn/cli.py (learn commands registered into electrodrive.cli; train subcommand uses required --config and required --out).
// - Repo: electrodrive/cli.py (registers learn commands when electrodrive/learn/cli.py exists; so command is `python -m electrodrive.cli train ...`).
// - Repo: electrodrive/live/controls.py (control protocol run_dir context; include EDE_RUN_DIR env placeholder).

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

export default function WorkflowFormLearnTrain(props: WorkflowFormProps) {
  const disabled = Boolean(props.disabled);

  const [runsRoot, setRunsRoot] = useState<string>(props.defaultRunsRoot ?? "./runs");
  const [runName, setRunName] = useState<string>("");

  const [presetId, setPresetId] = useState<string>("");
  const [presetName, setPresetName] = useState<string>("");

  // Repo: electrodrive/learn/cli.py uses --config (YAML) and --out (required).
  const [configPath, setConfigPath] = useState<string>("");
  const [useInlineConfig, setUseInlineConfig] = useState<boolean>(false);

  // Inline JSON can be written by backend to a config file path. YAML parsers commonly accept JSON as a subset,
  // but we keep this "advanced" and make it explicit in the Explain panel.
  const [configJson, setConfigJson] = useState<unknown>({});
  const [configMeta, setConfigMeta] = useState<{ isValid: boolean; text: string; parseError?: string; validationErrors?: string[] }>({
    isValid: true,
    text: "",
  });

  const [extraArgs, setExtraArgs] = useState<string>("");

  // Full override for argv (design-doc hardening: CLI flags may vary; this bypasses mismatches).
  const [fullCommandOverride, setFullCommandOverride] = useState<string>("");

  const presetOptions = (props.presets ?? []).filter((p) => p.workflow === "learn_train");
  const extraSplit = useMemo(() => shellSplit(extraArgs), [extraArgs]);

  const computed = useMemo(() => {
    const errors: Record<string, string> = {};
    const hasOverride = fullCommandOverride.trim().length > 0;

    if (!runsRoot.trim()) errors.runsRoot = "Runs root is required.";

    const env: Record<string, string> = { EDE_RUN_DIR: "${RUN_DIR}" };

    if (hasOverride) {
      const overrideSplit = shellSplit(fullCommandOverride);

      if (overrideSplit.error) errors.fullCommandOverride = overrideSplit.error;
      if (overrideSplit.args.length === 0) errors.fullCommandOverride = "Override command is empty.";

      const hasRunDirToken = overrideSplit.args.some((a) => a.includes("${RUN_DIR}"));
      if (!hasRunDirToken) errors.fullCommandOverride = "Override must include ${RUN_DIR} so the backend can substitute the run directory.";

      // If inline config is enabled, still block on invalid JSON (FR-2).
      if (useInlineConfig && !configMeta.isValid) errors.configJson = configMeta.parseError ?? "Inline config JSON is invalid.";

      const canLaunch = Object.keys(errors).length === 0 && !disabled;

      return { errors, argv: overrideSplit.args, env, canLaunch, argvPretty: argvToPretty(overrideSplit.args) };
    }

    // Normal structured validation/build
    if (useInlineConfig) {
      if (!configMeta.isValid) errors.configJson = configMeta.parseError ?? "Inline config JSON is invalid.";
    } else {
      if (!configPath.trim()) errors.configPath = "Config path is required (--config).";
    }

    if (extraSplit.error) errors.extraArgs = extraSplit.error;

    const argv: string[] = ["python", "-m", "electrodrive.cli", "train", "--config", useInlineConfig ? "${RUN_DIR}/train_config.json" : configPath.trim(), "--out", "${RUN_DIR}"];

    if (extraSplit.args.length > 0) argv.push(...extraSplit.args);

    const canLaunch = Object.keys(errors).length === 0 && !disabled;

    return {
      errors,
      argv,
      env,
      canLaunch,
      argvPretty: argvToPretty(argv),
    };
  }, [configMeta.isValid, configMeta.parseError, configPath, disabled, extraSplit.args, extraSplit.error, fullCommandOverride, runsRoot, useInlineConfig]);

  const onLoadPreset = async () => {
    if (!props.onLoadPreset || !presetId) return;
    const patch = await props.onLoadPreset(presetId);

    if (patch.runsRoot !== undefined) setRunsRoot(String(patch.runsRoot));
    if (patch.runName !== undefined) setRunName(String(patch.runName ?? ""));
    if (patch.configPath !== undefined) setConfigPath(String(patch.configPath ?? ""));
    if (patch.configJson !== undefined) {
      setUseInlineConfig(true);
      setConfigJson(patch.configJson);
    }
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
      workflow: "learn_train",
      runsRoot: runsRoot.trim(),
      runName: runName.trim() || undefined,
      configPath: !useInlineConfig ? configPath.trim() || undefined : undefined,
      configJson: useInlineConfig && configMeta.isValid ? configJson : undefined,
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
      workflow: "learn_train",
      runsRoot: runsRoot.trim(),
      runName: runName.trim() || undefined,
      configPath: !useInlineConfig ? configPath.trim() || undefined : undefined,
      configJson: useInlineConfig && configMeta.isValid ? configJson : undefined,
      argv: computed.argv,
      env: computed.env,
      extraArgs: extraArgs.trim() || undefined,
    };

    await props.onLaunch(req);
  };

  const overrideActive = fullCommandOverride.trim().length > 0;

  return (
    <form onSubmit={onSubmit} style={{ display: "grid", gap: 14, maxWidth: 980 }}>
      <h2 style={{ margin: 0 }}>Learn Train (electrodrive.cli train)</h2>

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
                <input value={presetName} onChange={(e) => setPresetName(e.target.value)} disabled={disabled || !props.onSavePreset} style={inputStyle(false)} placeholder="my-train-preset" />
              </Field>
              <button type="button" onClick={onSavePreset} disabled={disabled || !props.onSavePreset || presetName.trim().length === 0} style={buttonStyle()}>
                Save
              </button>
            </div>

            <div style={{ fontSize: 12, color: "#444" }}>
              Presets are workflow-scoped (FR-2). The train CLI requires <code>--config</code> and <code>--out</code> (repo: electrodrive/learn/cli.py).
            </div>
          </div>
        ) : null}
      </Section>

      <Section title="Training config (repo-aligned)">
        <Checkbox label="Use inline JSON config (advanced)" checked={useInlineConfig} onChange={setUseInlineConfig} disabled={disabled} />

        {!useInlineConfig ? (
          <Field label="Config path (--config)" error={computed.errors.configPath} hint="Repo train CLI expects a config file path (YAML). The backend should record this in manifest.json for reproducibility (FR-3).">
            <input value={configPath} onChange={(e) => setConfigPath(e.target.value)} disabled={disabled} style={inputStyle(Boolean(computed.errors.configPath))} placeholder="path/to/train.yaml" />
          </Field>
        ) : (
          <div style={{ display: "grid", gap: 10 }}>
            <div style={{ fontSize: 12, color: "#444" }}>
              Inline config is validated client-side (FR-2). The command preview will pass <code>--config ${"{RUN_DIR}"}/train_config.json</code>. The backend should write
              this JSON file before launching. Many YAML loaders accept JSON as a subset; if your backend requires strict YAML, it can convert JSON→YAML.
            </div>

            <ConfigEditor
              label="Inline config JSON"
              value={configJson}
              schema={OBJECT_SCHEMA}
              required={true}
              height={200}
              onChange={(val, meta) => {
                setConfigJson(val ?? {});
                setConfigMeta({ isValid: meta.isValid, text: meta.text, parseError: meta.parseError, validationErrors: meta.validationErrors });
              }}
            />

            {computed.errors.configJson ? <div style={{ color: "#b00020", fontSize: 12 }}>{computed.errors.configJson}</div> : null}
          </div>
        )}
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
            placeholder='python -m electrodrive.cli train --config my_train.yaml --out ${RUN_DIR}'
          />
        </Field>
      </Section>

      <Section title="Advanced CLI args (escape hatch)">
        <Field label="Advanced CLI args (appended to argv)" error={computed.errors.extraArgs} hint={overrideActive ? "Disabled because a full command override is active." : "Shell-like splitting with quotes (FR-2)."}>
          <input value={extraArgs} onChange={(e) => setExtraArgs(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.extraArgs))} placeholder='e.g. --some-train-flag "value"' />
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
          Launch Learn Train Run
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
