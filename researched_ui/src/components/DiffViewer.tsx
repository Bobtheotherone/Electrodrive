import React, { useEffect, useId, useMemo, useRef, useState } from "react";

// Source notes:
// - Design Doc: FR-8 (Cross-run comparison + “What changed?”: argv diff + config/spec diff), §5.1 (manifest shape), §8 (day-one comparison view)
// - Design Doc: FR-9.5 (gate dashboards are stable artifacts) + FR-9.6 (log coverage used in gate dashboard), used here for “Gate” tab when gate_dashboard.json exists
// - Repo: electrodrive/researched/api.py (GET /runs/{run_id} returns {manifest: ...}; POST /compare returns unified diff lines when include_unified_diff=true; artifact fetch via /runs/{run_id}/artifact?path=...)
// - Repo: electrodrive/researched/plot_service.py (plots/gate_dashboard.json artifact path; gate dashboard JSON shape and log_coverage embedding)

type DiffViewerProps = {
  leftRunId: string;
  rightRunId: string;
  apiBase?: string; // default "/api"
  className?: string;
};

type TabId = "argv" | "manifest" | "spec_digest" | "gate";

type DiffLine = {
  kind: "ctx" | "add" | "del" | "skip";
  text: string;
  aLine?: number;
  bLine?: number;
  skipCount?: number;
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

function stripVolatile(value: unknown, path: string[] = []): unknown {
  // Design Doc FR-8: ignore volatile keys by default for manifest diff.
  // Volatile keys include timestamps and generated fields that change without semantic config changes.
  const volatileKeys = new Set([
    "started_at",
    "ended_at",
    "generated_at",
    "ts",
    "t",
    "pid",
    "ppid",
    "phase",
    "internal_status",
    "updated_at",
    "last_seen_at",
    "last_event_t",
  ]);

  if (Array.isArray(value)) return value.map((v, i) => stripVolatile(v, path.concat(String(i))));
  if (!isRecord(value)) return value;

  const out: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(value)) {
    if (volatileKeys.has(k)) continue;
    // Drop any nested “researched” internal blocks that are not part of stable manifest surface.
    if (path.length === 0 && k === "researched") continue;
    out[k] = stripVolatile(v, path.concat(k));
  }
  return out;
}

// Myers diff for line arrays (no external deps).
// IMPORTANT: we snapshot V BEFORE updating for each d (required for correct backtracking).
type Op = { type: "equal" | "insert" | "delete"; line: string };

function myersDiff(a: string[], b: string[]): Op[] {
  const n = a.length;
  const m = b.length;
  const max = n + m;

  // v maps diagonal k -> furthest x reached on that diagonal
  const v = new Map<number, number>();
  v.set(1, 0);

  // trace[d] is a snapshot of v BEFORE exploring paths of edit distance d
  const trace: Map<number, number>[] = [];

  for (let d = 0; d <= max; d++) {
    trace.push(new Map(v));

    for (let k = -d; k <= d; k += 2) {
      let x: number;
      if (k === -d || (k !== d && (v.get(k - 1) ?? 0) < (v.get(k + 1) ?? 0))) {
        // Down: insertion (advance in b)
        x = v.get(k + 1) ?? 0;
      } else {
        // Right: deletion (advance in a)
        x = (v.get(k - 1) ?? 0) + 1;
      }

      let y = x - k;

      // Follow snake (diagonal matches)
      while (x < n && y < m && a[x] === b[y]) {
        x++;
        y++;
      }

      v.set(k, x);

      if (x >= n && y >= m) {
        return backtrackMyers(a, b, trace);
      }
    }
  }

  return [];
}

function backtrackMyers(a: string[], b: string[], trace: Map<number, number>[]): Op[] {
  const ops: Op[] = [];
  let x = a.length;
  let y = b.length;

  for (let d = trace.length - 1; d >= 0; d--) {
    const v = trace[d];
    const k = x - y;

    let prevK: number;
    if (k === -d || (k !== d && (v.get(k - 1) ?? 0) < (v.get(k + 1) ?? 0))) {
      prevK = k + 1;
    } else {
      prevK = k - 1;
    }

    const prevX = v.get(prevK) ?? 0;
    const prevY = prevX - prevK;

    // Consume snake (matches)
    while (x > prevX && y > prevY) {
      ops.push({ type: "equal", line: a[x - 1] });
      x--;
      y--;
    }

    if (d === 0) break;

    // Determine whether this step was an insert or delete
    if (x === prevX) {
      if (y > 0) ops.push({ type: "insert", line: b[y - 1] });
      y = Math.max(0, y - 1);
    } else {
      if (x > 0) ops.push({ type: "delete", line: a[x - 1] });
      x = Math.max(0, x - 1);
    }
  }

  ops.reverse();
  return ops;
}

function opsToDiffLines(ops: Op[]): DiffLine[] {
  const out: DiffLine[] = [];
  let aLine = 1;
  let bLine = 1;
  for (const op of ops) {
    if (op.type === "equal") {
      out.push({ kind: "ctx", text: op.line, aLine, bLine });
      aLine++;
      bLine++;
    } else if (op.type === "delete") {
      out.push({ kind: "del", text: op.line, aLine, bLine: undefined });
      aLine++;
    } else {
      out.push({ kind: "add", text: op.line, aLine: undefined, bLine });
      bLine++;
    }
  }
  return out;
}

function collapseUnchanged(lines: DiffLine[], context = 3): DiffLine[] {
  // Collapse long runs of unchanged context into a single “skip” line.
  const isChange = (l: DiffLine) => l.kind === "add" || l.kind === "del";
  const keep = new Array(lines.length).fill(false);

  for (let i = 0; i < lines.length; i++) {
    if (isChange(lines[i])) {
      const start = Math.max(0, i - context);
      const end = Math.min(lines.length - 1, i + context);
      for (let j = start; j <= end; j++) keep[j] = true;
    }
  }

  const out: DiffLine[] = [];
  let i = 0;
  while (i < lines.length) {
    if (keep[i]) {
      out.push(lines[i]);
      i++;
      continue;
    }
    // Skip run
    let j = i;
    while (j < lines.length && !keep[j]) j++;
    const count = j - i;
    out.push({ kind: "skip", text: "", skipCount: count });
    i = j;
  }
  return out;
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

type RunLoaded = {
  runId: string;
  envelope?: Record<string, unknown>;
  manifest?: Record<string, unknown>;
  error?: string;
};

function pickManifestFromEnvelope(data: unknown): Record<string, unknown> | undefined {
  if (!isRecord(data)) return undefined;
  const m = (data as any).manifest;
  if (isRecord(m)) return m;
  // Fallback: sometimes the response might be the manifest itself.
  if ("workflow" in data || "inputs" in data || "run_id" in data) return data as Record<string, unknown>;
  return undefined;
}

function getWorkflow(man?: Record<string, unknown>, env?: Record<string, unknown>): string | undefined {
  if (env && isString((env as any).workflow)) return (env as any).workflow as string;
  if (man && isString((man as any).workflow)) return (man as any).workflow as string;
  return undefined;
}
function getGitSha(man?: Record<string, unknown>): string | undefined {
  if (!man) return undefined;
  const git = (man as any).git;
  if (isRecord(git) && isString((git as any).sha)) return String((git as any).sha).slice(0, 10);
  return undefined;
}

function normalizeCommand(man?: Record<string, unknown>): string[] {
  if (!man) return [];
  const inputs = (man as any).inputs;
  if (isRecord(inputs) && Array.isArray((inputs as any).command)) return (inputs as any).command.filter((x: unknown) => typeof x === "string") as string[];
  if (Array.isArray((man as any).command)) return ((man as any).command as unknown[]).filter((x) => typeof x === "string") as string[];
  return [];
}

function normalizeEnvOverrides(man?: Record<string, unknown>): Record<string, string> {
  if (!man) return {};
  const inputs = (man as any).inputs;
  const cand = isRecord(inputs) ? ((inputs as any).env_overrides ?? (inputs as any).env) : undefined;
  if (!isRecord(cand)) return {};
  const out: Record<string, string> = {};
  for (const [k, v] of Object.entries(cand)) {
    if (typeof v === "string") out[k] = v;
    else if (typeof v === "number" || typeof v === "boolean") out[k] = String(v);
  }
  return out;
}

function safeGateObject(man?: Record<string, unknown>): unknown {
  if (!man) return {};
  if (isRecord((man as any).gate)) return (man as any).gate;
  // Some manifests may have gate fields at root (per tolerance requirement).
  const keys = ["gate1_status", "gate2_status", "gate3_status", "structure_score", "novelty_score"];
  const out: Record<string, unknown> = {};
  let any = false;
  for (const k of keys) {
    if (k in man) {
      out[k] = (man as any)[k];
      any = true;
    }
  }
  return any ? out : {};
}

function textToLines(text: string): string[] {
  return text.replace(/\r\n/g, "\n").split("\n");
}

function buildDiff(leftText: string, rightText: string): DiffLine[] {
  const a = textToLines(leftText);
  const b = textToLines(rightText);
  const ops = myersDiff(a, b);
  return opsToDiffLines(ops);
}

function unifiedDiffToDiffLines(serverLines: string[]): DiffLine[] {
  return serverLines.map((l) => {
    // Preserve headers/hunks as context lines (don’t strip).
    if (l.startsWith("+++ ") || l.startsWith("--- ") || l.startsWith("@@")) return { kind: "ctx", text: l };

    // Strip the single-character prefix for real diff lines.
    if (l.startsWith("+")) return { kind: "add", text: l.slice(1) };
    if (l.startsWith("-")) return { kind: "del", text: l.slice(1) };
    if (l.startsWith(" ")) return { kind: "ctx", text: l.slice(1) };

    // Fallback.
    return { kind: "ctx", text: l };
  });
}

function useLatestRef<T>(value: T) {
  const ref = useRef(value);
  ref.current = value;
  return ref;
}

export default function DiffViewer(props: DiffViewerProps) {
  const apiBase = props.apiBase ?? "/api";
  const leftRunId = props.leftRunId;
  const rightRunId = props.rightRunId;

  const base = apiBase.replace(/\/+$/, "");
  const leftUrl = `${base}/runs/${encodeURIComponent(leftRunId)}`;
  const rightUrl = `${base}/runs/${encodeURIComponent(rightRunId)}`;

  const artifactUrl = (runId: string, relpath: string) =>
    `${base}/runs/${encodeURIComponent(runId)}/artifact?path=${encodeURIComponent(relpath)}`;

  const [left, setLeft] = useState<RunLoaded>({ runId: leftRunId });
  const [right, setRight] = useState<RunLoaded>({ runId: rightRunId });
  const [loading, setLoading] = useState(true);
  const [serverDiffLines, setServerDiffLines] = useState<string[] | null>(null);
  const [serverDiffError, setServerDiffError] = useState<string | null>(null);

  const [tab, setTab] = useState<TabId>("argv");
  const [search, setSearch] = useState<string>("");
  const [hideContext, setHideContext] = useState<boolean>(true);
  const [ignoreVolatile, setIgnoreVolatile] = useState<boolean>(true);

  const [gateLeftDash, setGateLeftDash] = useState<Record<string, unknown> | null>(null);
  const [gateRightDash, setGateRightDash] = useState<Record<string, unknown> | null>(null);
  const [gateDashError, setGateDashError] = useState<string | null>(null);

  const tabsId = useId();

  const tabOrder: TabId[] = ["argv", "manifest", "spec_digest", "gate"];

  useEffect(() => {
    setTab("argv");
  }, [leftRunId, rightRunId]);

  useEffect(() => {
    const ac = new AbortController();
    setLoading(true);
    setLeft({ runId: leftRunId });
    setRight({ runId: rightRunId });
    setServerDiffLines(null);
    setServerDiffError(null);
    setGateLeftDash(null);
    setGateRightDash(null);
    setGateDashError(null);

    (async () => {
      const [l, r] = await Promise.all([fetchJson(leftUrl, ac.signal), fetchJson(rightUrl, ac.signal)]);
      if (l.ok) {
        const man = pickManifestFromEnvelope(l.value);
        setLeft({ runId: leftRunId, envelope: isRecord(l.value) ? (l.value as Record<string, unknown>) : undefined, manifest: man });
      } else {
        setLeft({ runId: leftRunId, error: `Failed to load: ${l.message}` });
      }
      if (r.ok) {
        const man = pickManifestFromEnvelope(r.value);
        setRight({ runId: rightRunId, envelope: isRecord(r.value) ? (r.value as Record<string, unknown>) : undefined, manifest: man });
      } else {
        setRight({ runId: rightRunId, error: `Failed to load: ${r.message}` });
      }
      setLoading(false);
    })().catch((e) => {
      if ((e as any)?.name === "AbortError") return;
      setLeft({ runId: leftRunId, error: (e as Error)?.message || "Unknown error" });
      setRight({ runId: rightRunId, error: (e as Error)?.message || "Unknown error" });
      setLoading(false);
    });

    return () => ac.abort();
  }, [leftRunId, rightRunId, leftUrl, rightUrl]);

  useEffect(() => {
    // Design Doc FR-8: if backend provides a diff endpoint, use it for unified diff.
    // Repo: electrodrive/researched/api.py exposes POST /compare {run_ids, include_unified_diff}.
    const ac = new AbortController();
    setServerDiffLines(null);
    setServerDiffError(null);

    (async () => {
      try {
        const resp = await fetch(`${base}/compare`, {
          method: "POST",
          signal: ac.signal,
          headers: { "Content-Type": "application/json", Accept: "application/json" },
          body: JSON.stringify({ run_ids: [leftRunId, rightRunId], include_unified_diff: true }),
        });
        if (!resp.ok) return;
        const json = (await resp.json()) as unknown;
        if (!isRecord(json)) return;
        const dl = (json as any).diff_lines;
        if (Array.isArray(dl)) {
          setServerDiffLines(dl.filter((x: unknown) => typeof x === "string") as string[]);
        } else if (typeof dl === "string") {
          setServerDiffLines(textToLines(dl));
        }
      } catch (e) {
        if ((e as any)?.name === "AbortError") return;
        setServerDiffError((e as Error)?.message || "Failed to load server diff");
      }
    })();

    return () => ac.abort();
  }, [base, leftRunId, rightRunId]);

  useEffect(() => {
    // Gate dashboards: prefer plots/gate_dashboard.json per FR-9.5 stable artifact contract.
    if (tab !== "gate") return;

    const ac = new AbortController();
    setGateDashError(null);

    (async () => {
      const [l, r] = await Promise.all([
        fetchJson(artifactUrl(leftRunId, "plots/gate_dashboard.json"), ac.signal),
        fetchJson(artifactUrl(rightRunId, "plots/gate_dashboard.json"), ac.signal),
      ]);
      if (l.ok && isRecord(l.value)) setGateLeftDash(l.value);
      if (r.ok && isRecord(r.value)) setGateRightDash(r.value);

      if (!l.ok && !r.ok) {
        // Not fatal; we can fall back to manifest.gate.
        setGateDashError("Gate dashboard artifact not available; using manifest.gate where possible.");
      }
    })().catch((e) => {
      if ((e as any)?.name === "AbortError") return;
      setGateDashError((e as Error)?.message || "Failed to fetch gate dashboards");
    });

    return () => ac.abort();
  }, [tab, base, leftRunId, rightRunId]);

  const headerLeft = useMemo(() => {
    const wf = getWorkflow(left.manifest, left.envelope);
    const sha = getGitSha(left.manifest);
    return { wf, sha };
  }, [left.manifest, left.envelope]);

  const headerRight = useMemo(() => {
    const wf = getWorkflow(right.manifest, right.envelope);
    const sha = getGitSha(right.manifest);
    return { wf, sha };
  }, [right.manifest, right.envelope]);

  const diffBlock = useMemo(() => {
    if (!left.manifest || !right.manifest) return null;

    if (tab === "argv") {
      const leftCmd = normalizeCommand(left.manifest);
      const rightCmd = normalizeCommand(right.manifest);
      const leftEnv = normalizeEnvOverrides(left.manifest);
      const rightEnv = normalizeEnvOverrides(right.manifest);

      const cmdDiff = buildDiff(leftCmd.join("\n"), rightCmd.join("\n"));
      const envDiff = buildDiff(stableStringify(leftEnv, 2), stableStringify(rightEnv, 2));
      return { title: "Command (argv)", parts: [{ label: "argv", lines: cmdDiff }, { label: "env_overrides", lines: envDiff }] };
    }

    if (tab === "manifest") {
      if (serverDiffLines && serverDiffLines.length) {
        const lines = unifiedDiffToDiffLines(serverDiffLines);
        return {
          title: "Manifest (server unified diff)",
          parts: [{ label: "server", lines }],
          note: serverDiffError ? `Server diff note: ${serverDiffError}` : undefined,
        };
      }

      const leftM = ignoreVolatile ? (stripVolatile(left.manifest) as unknown) : left.manifest;
      const rightM = ignoreVolatile ? (stripVolatile(right.manifest) as unknown) : right.manifest;
      const lines = buildDiff(stableStringify(leftM, 2), stableStringify(rightM, 2));
      return { title: "Manifest", parts: [{ label: "manifest.json", lines }] };
    }

    if (tab === "spec_digest") {
      const l = isRecord((left.manifest as any).spec_digest) ? (left.manifest as any).spec_digest : {};
      const r = isRecord((right.manifest as any).spec_digest) ? (right.manifest as any).spec_digest : {};
      const lines = buildDiff(stableStringify(l, 2), stableStringify(r, 2));
      return { title: "Spec digest", parts: [{ label: "spec_digest", lines }] };
    }

    // gate tab
    const leftGate = gateLeftDash ?? (safeGateObject(left.manifest) as any);
    const rightGate = gateRightDash ?? (safeGateObject(right.manifest) as any);
    const lines = buildDiff(stableStringify(leftGate, 2), stableStringify(rightGate, 2));
    return {
      title: "Gate",
      parts: [{ label: gateLeftDash || gateRightDash ? "plots/gate_dashboard.json (preferred)" : "manifest.gate", lines }],
      note: gateDashError || undefined,
    };
  }, [
    left.manifest,
    right.manifest,
    tab,
    ignoreVolatile,
    serverDiffLines,
    serverDiffError,
    gateLeftDash,
    gateRightDash,
    gateDashError,
  ]);

  const filteredParts = useMemo(() => {
    if (!diffBlock) return null;
    const q = search.trim().toLowerCase();
    const parts = diffBlock.parts.map((p) => {
      let lines = p.lines;

      if (hideContext && q.length === 0) lines = collapseUnchanged(lines, 3);

      if (q.length > 0) {
        // Filter lines by substring (keep skip lines out).
        lines = lines.filter((l) => l.kind !== "skip" && l.text.toLowerCase().includes(q));
      }

      return { ...p, lines };
    });
    return { ...diffBlock, parts };
  }, [diffBlock, search, hideContext]);

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

  const tabBtn = (active: boolean): React.CSSProperties => ({
    padding: "8px 10px",
    border: "1px solid #ddd",
    borderBottom: active ? "2px solid #1b4f9c" : "1px solid #ddd",
    background: active ? "#f3f7ff" : "#fafafa",
    cursor: "pointer",
    borderRadius: 6,
  });

  const renderLine = (l: DiffLine, idx: number) => {
    if (l.kind === "skip") {
      return (
        <div key={`skip-${idx}`} style={{ ...styleMono, color: "#666", padding: "4px 8px", fontStyle: "italic" }}>
          … {l.skipCount ?? 0} unchanged lines hidden …
        </div>
      );
    }
    const prefix = l.kind === "add" ? "+" : l.kind === "del" ? "-" : " ";
    const color = l.kind === "add" ? "#0b5d1e" : l.kind === "del" ? "#8a1212" : "#333";
    const bg = l.kind === "add" ? "#e7f7ec" : l.kind === "del" ? "#fdecec" : "transparent";
    return (
      <div
        key={idx}
        style={{
          display: "grid",
          gridTemplateColumns: "64px 64px 1fr",
          gap: 8,
          padding: "2px 8px",
          background: bg,
          borderBottom: "1px solid #f3f3f3",
          alignItems: "baseline",
        }}
      >
        <div style={{ ...styleMono, color: "#666", textAlign: "right" }}>{l.aLine ?? ""}</div>
        <div style={{ ...styleMono, color: "#666", textAlign: "right" }}>{l.bLine ?? ""}</div>
        <div style={{ ...styleMono, color, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>
          <span aria-hidden="true" style={{ marginRight: 6 }}>
            {prefix}
          </span>
          {l.text}
        </div>
      </div>
    );
  };

  const tabsRef = useLatestRef(tab);

  const onTabKeyDown = (e: React.KeyboardEvent) => {
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    e.preventDefault();
    const idx = tabOrder.indexOf(tabsRef.current);
    const nextIdx = e.key === "ArrowRight" ? (idx + 1) % tabOrder.length : (idx - 1 + tabOrder.length) % tabOrder.length;
    setTab(tabOrder[nextIdx]);
  };

  return (
    <div className={props.className} style={styleRoot}>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12, marginBottom: 12 }}>
        <div style={styleCard}>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Left</div>
          <div>
            <span style={styleMono}>{leftRunId}</span>
          </div>
          <div style={{ color: "#555" }}>
            {headerLeft.wf ? <>workflow {headerLeft.wf}</> : <>workflow —</>} {headerLeft.sha ? <>· git {headerLeft.sha}</> : null}
          </div>
          {left.error ? (
            <div role="alert" style={{ marginTop: 8, color: "#8a1212" }}>
              {left.error}
            </div>
          ) : null}
        </div>

        <div style={styleCard}>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>Right</div>
          <div>
            <span style={styleMono}>{rightRunId}</span>
          </div>
          <div style={{ color: "#555" }}>
            {headerRight.wf ? <>workflow {headerRight.wf}</> : <>workflow —</>} {headerRight.sha ? <>· git {headerRight.sha}</> : null}
          </div>
          {right.error ? (
            <div role="alert" style={{ marginTop: 8, color: "#8a1212" }}>
              {right.error}
            </div>
          ) : null}
        </div>
      </div>

      <div style={{ ...styleCard, marginBottom: 12 }}>
        <div
          role="tablist"
          aria-labelledby={tabsId}
          onKeyDown={onTabKeyDown}
          style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center" }}
        >
          <span id={tabsId} style={{ fontWeight: 700, marginRight: 6 }}>
            Diff:
          </span>
          <button type="button" role="tab" aria-selected={tab === "argv"} style={tabBtn(tab === "argv")} onClick={() => setTab("argv")}>
            Command (argv)
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "manifest"}
            style={tabBtn(tab === "manifest")}
            onClick={() => setTab("manifest")}
          >
            Manifest
          </button>
          <button
            type="button"
            role="tab"
            aria-selected={tab === "spec_digest"}
            style={tabBtn(tab === "spec_digest")}
            onClick={() => setTab("spec_digest")}
          >
            Spec digest
          </button>
          <button type="button" role="tab" aria-selected={tab === "gate"} style={tabBtn(tab === "gate")} onClick={() => setTab("gate")}>
            Gate
          </button>

          <div style={{ flex: 1 }} />

          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <span style={{ color: "#555" }}>Search</span>
            <input
              type="text"
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              placeholder="filter diff lines…"
              style={{ padding: "6px 8px", border: "1px solid #ddd", borderRadius: 6, minWidth: 220 }}
            />
          </label>

          <label style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 10 }}>
            <input type="checkbox" checked={hideContext} onChange={(e) => setHideContext(e.target.checked)} />
            <span style={{ color: "#555" }}>Hide unchanged context</span>
          </label>

          {tab === "manifest" ? (
            <label style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: 10 }}>
              <input type="checkbox" checked={ignoreVolatile} onChange={(e) => setIgnoreVolatile(e.target.checked)} />
              <span style={{ color: "#555" }}>Ignore volatile keys</span>
            </label>
          ) : null}
        </div>
      </div>

      <div style={styleCard}>
        {loading ? (
          <div style={{ color: "#555" }}>Loading manifests…</div>
        ) : !left.manifest || !right.manifest ? (
          <div style={{ color: "#555" }}>Manifests unavailable for one or both runs.</div>
        ) : !filteredParts ? (
          <div style={{ color: "#555" }}>No diff available.</div>
        ) : (
          <>
            <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 12 }}>
              <div style={{ fontWeight: 700 }}>{filteredParts.title}</div>
              {filteredParts.note ? <div style={{ color: "#666", fontSize: 12 }}>{filteredParts.note}</div> : null}
            </div>

            {tab === "gate" ? (
              <div style={{ marginTop: 8, color: "#555", fontSize: 12 }}>
                Uses gate dashboard artifact <span style={styleMono}>plots/gate_dashboard.json</span> when present (Design Doc FR-9.5); otherwise falls back to{" "}
                <span style={styleMono}>manifest.gate</span>.
              </div>
            ) : null}

            <div style={{ marginTop: 10 }}>
              {filteredParts.parts.map((p, idx) => (
                <div key={idx} style={{ marginTop: idx === 0 ? 0 : 14 }}>
                  <div style={{ fontWeight: 700, marginBottom: 6 }}>{p.label}</div>
                  <div style={{ border: "1px solid #eee", borderRadius: 6, overflow: "hidden" }}>
                    <div
                      style={{
                        display: "grid",
                        gridTemplateColumns: "64px 64px 1fr",
                        gap: 8,
                        padding: "6px 8px",
                        borderBottom: "1px solid #eee",
                        background: "#fafafa",
                        color: "#666",
                      }}
                    >
                      <div style={{ ...styleMono, textAlign: "right" }}>Left</div>
                      <div style={{ ...styleMono, textAlign: "right" }}>Right</div>
                      <div style={{ ...styleMono }}>Line</div>
                    </div>
                    <div aria-label={`diff-${p.label}`} style={{ maxHeight: 520, overflow: "auto" }}>
                      {p.lines.length ? p.lines.map(renderLine) : <div style={{ padding: 10, color: "#666" }}>No diff lines.</div>}
                    </div>
                  </div>
                </div>
              ))}
            </div>

            {tab === "gate" ? (
              <details style={{ marginTop: 14 }}>
                <summary style={{ cursor: "pointer" }}>Open gate artifacts</summary>
                <div style={{ marginTop: 8, display: "flex", gap: 12, flexWrap: "wrap" }}>
                  <a href={artifactUrl(leftRunId, "plots/gate_dashboard.json")} target="_blank" rel="noreferrer">
                    Left: gate_dashboard.json
                  </a>
                  <a href={artifactUrl(rightRunId, "plots/gate_dashboard.json")} target="_blank" rel="noreferrer">
                    Right: gate_dashboard.json
                  </a>
                </div>
              </details>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}
