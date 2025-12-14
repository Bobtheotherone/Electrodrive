import React, { useMemo, useState } from "react";
import ConfigEditor from "./ConfigEditor";

// Source notes:
// - Design Doc: FR-1 (launch workflows), FR-2 (validation + presets + Explain exact CLI command; advanced args escape hatch), FR-3 (run_dir contract).
// - Repo: electrodrive/tools/images_discover.py (discover CLI flags: --spec (required), --out, --basis/--nmax/--reg-l1, --n-points/--ratio-boundary,
//         --solver, --operator-mode, --adaptive-collocation-rounds, --aug-boundary, --subtract-physical, --lambda-group, --restarts, --intensive,
//         --basis-generator/--geo-encoder/--basis-generator-mode, --model-checkpoint; outputs include discovered_system.json + discovery_manifest.json).
// - Repo: electrodrive/live/controls.py (control protocol run_dir context; we include EDE_RUN_DIR env placeholder for later integration).

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

export default function WorkflowFormImagesDiscover(props: WorkflowFormProps) {
  const disabled = Boolean(props.disabled);

  const [runsRoot, setRunsRoot] = useState<string>(props.defaultRunsRoot ?? "./runs");
  const [runName, setRunName] = useState<string>("");

  const [presetId, setPresetId] = useState<string>("");
  const [presetName, setPresetName] = useState<string>("");

  const [specPath, setSpecPath] = useState<string>("");

  // Repo CLI defaults (electrodrive/tools/images_discover.py)
  const [basis, setBasis] = useState<string>("point");
  const [nmax, setNmax] = useState<string>("16");
  const [regL1, setRegL1] = useState<string>("0.001");

  const [nPoints, setNPoints] = useState<string>("");
  const [ratioBoundary, setRatioBoundary] = useState<string>("");

  const [solver, setSolver] = useState<string>(""); // choices: ista, lista (empty = omit)
  const [operatorMode, setOperatorMode] = useState<boolean>(false); // --operator-mode (store_true; default None in repo)

  const [adaptiveRounds, setAdaptiveRounds] = useState<string>("");
  const [augBoundary, setAugBoundary] = useState<boolean>(false);
  const [subtractPhysical, setSubtractPhysical] = useState<boolean>(false);
  const [lambdaGroup, setLambdaGroup] = useState<string>("0.0");
  const [restarts, setRestarts] = useState<string>("");

  const [intensive, setIntensive] = useState<boolean>(false);

  const [basisGenerator, setBasisGenerator] = useState<string>("none");
  const [geoEncoder, setGeoEncoder] = useState<string>("egnn");
  const [basisGeneratorMode, setBasisGeneratorMode] = useState<string>("static_only");
  const [modelCheckpoint, setModelCheckpoint] = useState<string>("");

  const [extraArgs, setExtraArgs] = useState<string>("");

  // Full override for argv (design-doc hardening: CLI flags may vary; this bypasses mismatches).
  const [fullCommandOverride, setFullCommandOverride] = useState<string>("");

  // Optional inline JSON config (stored by backend; not passed to images_discover CLI by default)
  const [configJson, setConfigJson] = useState<unknown>({});
  const [configMeta, setConfigMeta] = useState<{ isValid: boolean; text: string; parseError?: string; validationErrors?: string[] }>({
    isValid: true,
    text: "",
  });

  const presetOptions = (props.presets ?? []).filter((p) => p.workflow === "images_discover");
  const extraSplit = useMemo(() => shellSplit(extraArgs), [extraArgs]);

  const computed = useMemo(() => {
    const errors: Record<string, string> = {};
    const hasOverride = fullCommandOverride.trim().length > 0;

    if (!runsRoot.trim()) errors.runsRoot = "Runs root is required.";
    if (!configMeta.isValid) errors.configJson = configMeta.parseError ?? "Config JSON is invalid.";

    const env: Record<string, string> = { EDE_RUN_DIR: "${RUN_DIR}" };

    if (hasOverride) {
      const overrideSplit = shellSplit(fullCommandOverride);

      if (overrideSplit.error) errors.fullCommandOverride = overrideSplit.error;
      if (overrideSplit.args.length === 0) errors.fullCommandOverride = "Override command is empty.";

      const hasRunDirToken = overrideSplit.args.some((a) => a.includes("${RUN_DIR}"));
      if (!hasRunDirToken) errors.fullCommandOverride = "Override must include ${RUN_DIR} so the backend can substitute the run directory.";

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
    if (!specPath.trim()) errors.specPath = "Spec path is required (--spec).";

    const nmaxNum = toIntOrNull(nmax);
    if (nmaxNum === null || nmaxNum <= 0) errors.nmax = "nmax must be a positive integer.";

    const reg = toNumOrNull(regL1);
    if (reg === null || reg < 0) errors.regL1 = "reg-l1 must be a number >= 0.";

    const nPointsNum = toIntOrNull(nPoints);
    if (nPoints.trim() && (nPointsNum === null || nPointsNum <= 0)) errors.nPoints = "n-points must be a positive integer.";

    const ratioNum = toNumOrNull(ratioBoundary);
    if (ratioBoundary.trim() && (ratioNum === null || ratioNum < 0 || ratioNum > 1)) errors.ratioBoundary = "ratio-boundary must be between 0 and 1.";

    const adaptiveNum = toIntOrNull(adaptiveRounds);
    if (adaptiveRounds.trim() && (adaptiveNum === null || adaptiveNum < 0)) errors.adaptiveRounds = "adaptive-collocation-rounds must be an integer >= 0.";

    const restartsNum = toIntOrNull(restarts);
    if (restarts.trim() && (restartsNum === null || restartsNum < 0)) errors.restarts = "restarts must be an integer >= 0.";

    const lambdaNum = toNumOrNull(lambdaGroup);
    if (lambdaNum === null || lambdaNum < 0) errors.lambdaGroup = "lambda-group must be a number >= 0.";

    if (extraSplit.error) errors.extraArgs = extraSplit.error;

    // Build argv aligned to electrodrive/tools/images_discover.py
    const argv: string[] = [
      "python",
      "-m",
      "electrodrive.tools.images_discover",
      "discover",
      "--spec",
      specPath.trim(),
      "--out",
      "${RUN_DIR}",
      "--basis",
      basis.trim() || "point",
      "--nmax",
      String(nmaxNum ?? 16),
      "--reg-l1",
      String(reg ?? 0.001),
    ];

    if (nPointsNum !== null) argv.push("--n-points", String(nPointsNum));
    if (ratioNum !== null) argv.push("--ratio-boundary", String(ratioNum));

    if (solver.trim()) argv.push("--solver", solver.trim());
    if (operatorMode) argv.push("--operator-mode");

    if (adaptiveNum !== null) argv.push("--adaptive-collocation-rounds", String(adaptiveNum));
    if (augBoundary) argv.push("--aug-boundary");
    if (subtractPhysical) argv.push("--subtract-physical");
    if (lambdaNum !== null && lambdaNum !== 0) argv.push("--lambda-group", String(lambdaNum));
    if (restartsNum !== null) argv.push("--restarts", String(restartsNum));

    if (intensive) argv.push("--intensive");

    if (basisGenerator.trim() && basisGenerator.trim() !== "none") argv.push("--basis-generator", basisGenerator.trim());
    if (geoEncoder.trim() && geoEncoder.trim() !== "egnn") argv.push("--geo-encoder", geoEncoder.trim());
    if (basisGeneratorMode.trim() && basisGeneratorMode.trim() !== "static_only") argv.push("--basis-generator-mode", basisGeneratorMode.trim());

    if (modelCheckpoint.trim()) argv.push("--model-checkpoint", modelCheckpoint.trim());

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
    adaptiveRounds,
    augBoundary,
    basis,
    basisGenerator,
    basisGeneratorMode,
    configMeta.isValid,
    configMeta.parseError,
    disabled,
    extraSplit.args,
    extraSplit.error,
    fullCommandOverride,
    geoEncoder,
    intensive,
    lambdaGroup,
    modelCheckpoint,
    nPoints,
    nmax,
    operatorMode,
    ratioBoundary,
    regL1,
    restarts,
    runsRoot,
    solver,
    specPath,
  ]);

  const onLoadPreset = async () => {
    if (!props.onLoadPreset || !presetId) return;
    const patch = await props.onLoadPreset(presetId);

    if (patch.runsRoot !== undefined) setRunsRoot(String(patch.runsRoot));
    if (patch.runName !== undefined) setRunName(String(patch.runName ?? ""));
    if (patch.specPath !== undefined) setSpecPath(String(patch.specPath ?? ""));
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
      workflow: "images_discover",
      runsRoot: runsRoot.trim(),
      runName: runName.trim() || undefined,
      specPath: specPath.trim() || undefined,
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
      workflow: "images_discover",
      runsRoot: runsRoot.trim(),
      runName: runName.trim() || undefined,
      specPath: specPath.trim() || undefined,
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
      <h2 style={{ margin: 0 }}>Images Discover (electrodrive.tools.images_discover discover)</h2>

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
                <input value={presetName} onChange={(e) => setPresetName(e.target.value)} disabled={disabled || !props.onSavePreset} style={inputStyle(false)} placeholder="my-discover-preset" />
              </Field>
              <button type="button" onClick={onSavePreset} disabled={disabled || !props.onSavePreset || presetName.trim().length === 0} style={buttonStyle()}>
                Save
              </button>
            </div>

            <div style={{ fontSize: 12, color: "#444" }}>
              Presets are workflow-scoped (FR-2). Images discovery runs write <code>discovered_system.json</code> and <code>discovery_manifest.json</code> in the run
              directory (repo: electrodrive/tools/images_discover.py; design doc FR-3).
            </div>
          </div>
        ) : null}
      </Section>

      <Section title="Required inputs">
        <Field label="Spec path (--spec)" error={computed.errors.specPath} hint="Repo CLI requires --spec for images_discover (electrodrive/tools/images_discover.py).">
          <input
            value={specPath}
            onChange={(e) => setSpecPath(e.target.value)}
            disabled={disabled}
            style={inputStyle(Boolean(computed.errors.specPath))}
            placeholder="path/to/spec.json"
          />
        </Field>
      </Section>

      <Section title="Discovery options (from repo CLI)">
        <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))" }}>
          <Field label="Basis (--basis)">
            <input value={basis} onChange={(e) => setBasis(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(false)} placeholder="point" />
          </Field>

          <Field label="nmax (--nmax)" error={computed.errors.nmax}>
            <input value={nmax} onChange={(e) => setNmax(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.nmax))} inputMode="numeric" />
          </Field>

          <Field label="L1 regularization (--reg-l1)" error={computed.errors.regL1}>
            <input value={regL1} onChange={(e) => setRegL1(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.regL1))} inputMode="decimal" />
          </Field>

          <Field label="Solver (--solver)">
            <select value={solver} onChange={(e) => setSolver(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(false)}>
              <option value="">(default/auto)</option>
              <option value="ista">ista</option>
              <option value="lista">lista</option>
            </select>
          </Field>
        </div>

        <div style={{ display: "grid", gap: 8, marginTop: 8 }}>
          <Checkbox label="Operator mode (--operator-mode)" checked={operatorMode} onChange={setOperatorMode} disabled={disabled || overrideActive} />
          <Checkbox label="Augmented boundary (--aug-boundary)" checked={augBoundary} onChange={setAugBoundary} disabled={disabled || overrideActive} />
          <Checkbox label="Subtract physical potential (--subtract-physical)" checked={subtractPhysical} onChange={setSubtractPhysical} disabled={disabled || overrideActive} />
          <Checkbox label="Intensive mode (--intensive)" checked={intensive} onChange={setIntensive} disabled={disabled || overrideActive} />
        </div>

        <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", marginTop: 8 }}>
          <Field label="n-points (--n-points)" error={computed.errors.nPoints} hint="Optional override; leave empty to use repo defaults (or intensive overrides).">
            <input value={nPoints} onChange={(e) => setNPoints(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.nPoints))} inputMode="numeric" placeholder="(auto)" />
          </Field>

          <Field label="ratio-boundary (--ratio-boundary)" error={computed.errors.ratioBoundary} hint="Optional override; [0,1].">
            <input value={ratioBoundary} onChange={(e) => setRatioBoundary(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.ratioBoundary))} inputMode="decimal" placeholder="(auto)" />
          </Field>

          <Field label="Adaptive collocation rounds (--adaptive-collocation-rounds)" error={computed.errors.adaptiveRounds} hint="Optional; leave empty for none/auto.">
            <input value={adaptiveRounds} onChange={(e) => setAdaptiveRounds(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.adaptiveRounds))} inputMode="numeric" placeholder="(none)" />
          </Field>

          <Field label="Restarts (--restarts)" error={computed.errors.restarts} hint="Optional; leave empty for none/auto.">
            <input value={restarts} onChange={(e) => setRestarts(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.restarts))} inputMode="numeric" placeholder="(none)" />
          </Field>

          <Field label="Lambda group (--lambda-group)" error={computed.errors.lambdaGroup} hint="Default 0.0 in repo CLI.">
            <input value={lambdaGroup} onChange={(e) => setLambdaGroup(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.lambdaGroup))} inputMode="decimal" />
          </Field>

          <Field label="Model checkpoint (--model-checkpoint)" hint="Optional; for generator models.">
            <input value={modelCheckpoint} onChange={(e) => setModelCheckpoint(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(false)} placeholder="path/to/checkpoint.pt" />
          </Field>
        </div>

        <div style={{ display: "grid", gap: 10, gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", marginTop: 8 }}>
          <Field label="Basis generator (--basis-generator)">
            <select value={basisGenerator} onChange={(e) => setBasisGenerator(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(false)}>
              <option value="none">none (default)</option>
              <option value="mlp">mlp</option>
              <option value="diffusion">diffusion</option>
              <option value="hybrid_diffusion">hybrid_diffusion</option>
            </select>
          </Field>

          <Field label="Geo encoder (--geo-encoder)">
            <select value={geoEncoder} onChange={(e) => setGeoEncoder(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(false)}>
              <option value="egnn">egnn (default)</option>
              <option value="simple">simple</option>
            </select>
          </Field>

          <Field label="Basis generator mode (--basis-generator-mode)">
            <select value={basisGeneratorMode} onChange={(e) => setBasisGeneratorMode(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(false)}>
              <option value="static_only">static_only (default)</option>
              <option value="generate_only">generate_only</option>
              <option value="static_then_generate">static_then_generate</option>
              <option value="generate_then_static">generate_then_static</option>
              <option value="hybrid">hybrid</option>
            </select>
          </Field>
        </div>
      </Section>

      <Section title="Inline JSON config (optional)">
        <div style={{ fontSize: 12, color: "#444" }}>
          This inline JSON is validated client-side (FR-2) and intended to be saved into the run manifest by the backend (FR-3). The repo discovery CLI primarily
          accepts flags, not a config file.
        </div>

        <ConfigEditor
          label="Config JSON (stored in manifest.json)"
          value={configJson}
          schema={OBJECT_SCHEMA}
          required={false}
          height={180}
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
            placeholder='python -m electrodrive.tools.images_discover discover --spec my_spec.json --out ${RUN_DIR} --intensive'
          />
        </Field>
      </Section>

      <Section title="Advanced CLI args (escape hatch)">
        <Field
          label="Advanced CLI args (appended to argv)"
          error={computed.errors.extraArgs}
          hint={overrideActive ? "Disabled because a full command override is active." : "Shell-like splitting with quotes (FR-2)."}
        >
          <input value={extraArgs} onChange={(e) => setExtraArgs(e.target.value)} disabled={disabled || overrideActive} style={inputStyle(Boolean(computed.errors.extraArgs))} placeholder='e.g. --basis-generator diffusion --model-checkpoint "ckpt.pt"' />
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
          Launch Images Discover Run
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
