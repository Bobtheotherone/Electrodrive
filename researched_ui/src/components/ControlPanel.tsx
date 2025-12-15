import { useCallback, useEffect, useMemo, useRef, useState } from "react";

// Source notes:
// - Design Doc: FR-6 (control panel: pause/resume/terminate/write_every/snapshot-token; snapshot must be unique string token, not boolean), FR-5 (live monitor UI includes control panel).
// - Repo: electrodrive/live/controls.py (ControlState fields: terminate, pause, write_every, snapshot, ts, version, seq, ack_seq; extras preserved; writer increments seq and sets ts; atomic write) [schema: ~L26-39, write_controls: ~L439-533].
// - Repo: electrodrive/viz/live_console.py (existing live tooling uses the same control protocol; validates write_every/snapshot semantics) [context: early file ~L25-70].

export type ControlState = {
  pause?: boolean;
  terminate?: boolean;
  write_every?: number | null;
  snapshot?: string | null;
  ts?: number;
  version?: number;
  seq?: number;
  ack_seq?: number | null;
  [key: string]: unknown;
};

export interface ControlPanelProps {
  runId?: string;

  /** API base for control GET/POST (defaults to VITE_API_BASE or same-origin). */
  apiBase?: string;

  /** Poll interval to refresh current control state. */
  pollIntervalMs?: number;

  /** Optional externally supplied control state (presentational/controlled usage). */
  control?: ControlState;

  /** Optional external updater; if provided, component calls this instead of default fetch POST. */
  onUpdate?: (patch: Partial<ControlState>) => Promise<ControlState | void> | ControlState | void;

  /** Disable all control actions. */
  disabled?: boolean;

  /** Allow arbitrary extra props to avoid breaking unknown call-sites. */
  [key: string]: unknown;
}

function getEnvApiBase(): string | undefined {
  try {
    const maybe = (import.meta as unknown as { env?: Record<string, string | undefined> }).env;
    const v = maybe?.VITE_API_BASE;
    if (typeof v === "string" && v.trim()) return v.trim();
  } catch {
    // ignore
  }
  return undefined;
}

function resolveApiBase(apiBase?: string): string {
  const base = (apiBase ?? getEnvApiBase() ?? "").trim();
  if (base) return base.replace(/\/+$/, "");
  if (typeof window !== "undefined" && window.location?.origin) return window.location.origin;
  return "";
}

async function fetchJson(url: string, init: RequestInit = {}): Promise<unknown> {
  const res = await fetch(url, {
    ...init,
    headers: { Accept: "application/json", "Content-Type": "application/json", ...(init.headers ?? {}) },
  });
  if (!res.ok) throw new Error(`HTTP ${res.status} ${res.statusText}`);
  if (res.status === 204) return null;

  const ct = res.headers.get("content-type") ?? "";
  if (!ct.includes("application/json")) {
    const text = await res.text();
    try {
      return JSON.parse(text);
    } catch {
      return text;
    }
  }
  return await res.json();
}

function formatEpochSeconds(ts?: number): string {
  if (typeof ts !== "number" || !Number.isFinite(ts)) return "—";
  return new Date(ts * 1000).toISOString();
}

function getExtras(state: ControlState): Record<string, unknown> {
  const known = new Set(["pause", "terminate", "write_every", "snapshot", "ts", "version", "seq", "ack_seq"]);
  const extras: Record<string, unknown> = {};
  for (const [k, v] of Object.entries(state)) {
    if (!known.has(k)) extras[k] = v;
  }
  return extras;
}

function generateSnapshotToken(): string {
  // Must be a unique string token (FR-6 + repo controls schema uses snapshot: str|null).
  // Prefer crypto.randomUUID when available; else fallback to ISO timestamp + random.
  const g = (globalThis as unknown as { crypto?: { randomUUID?: () => string } }).crypto;
  if (g?.randomUUID) return g.randomUUID();
  return `${new Date().toISOString()}-${Math.random().toString(16).slice(2)}`;
}

export function ControlPanel(props: ControlPanelProps) {
  const { runId, apiBase, pollIntervalMs = 1500, control: controlProp, onUpdate, disabled } = props;

  const [control, setControl] = useState<ControlState>(() => controlProp ?? {});
  const [loading, setLoading] = useState<boolean>(!controlProp && !!runId);
  const [saving, setSaving] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [writeEveryText, setWriteEveryText] = useState<string>(""); // controlled input, empty => null
  const [confirmTerminate, setConfirmTerminate] = useState<boolean>(false);

  const lastFetchedAtRef = useRef<number>(0);

  useEffect(() => {
    if (controlProp) setControl(controlProp);
  }, [controlProp]);

  useEffect(() => {
    // Keep write_every input synced with fetched/prop state (but don't clobber while user is editing unless input empty).
    if (writeEveryText.trim() !== "") return;
    const we = control.write_every;
    if (typeof we === "number" && Number.isFinite(we)) setWriteEveryText(String(we));
    if (we == null) setWriteEveryText("");
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [control.write_every]);

  const base = useMemo(() => resolveApiBase(apiBase), [apiBase]);

  const fetchControl = useCallback(
    async (signal?: AbortSignal) => {
      if (!runId) return;
      if (controlProp) return; // controlled: don't poll over caller's data.
      setLoading(true);
      setError(null);

      const urls = [
        `${base}/api/runs/${encodeURIComponent(runId)}/controls`,
        `${base}/api/runs/${encodeURIComponent(runId)}/control`,
      ];

      let lastErr: unknown = null;
      for (const url of urls) {
        try {
          const data = await fetchJson(url, { method: "GET", signal });
          if (typeof data === "object" && data !== null) {
            setControl(data as ControlState);
            setLoading(false);
            setError(null);
            lastFetchedAtRef.current = Date.now();
            return;
          }
        } catch (e) {
          if (e instanceof DOMException && e.name === "AbortError") return;
          lastErr = e;
        }
      }

      // If request was aborted, do nothing (avoid false error state)
      if (signal?.aborted) return;

      setLoading(false);
      setError(lastErr instanceof Error ? lastErr.message : "Failed to load control state");
    },
    [runId, controlProp, base]
  );

  useEffect(() => {
    if (!runId) return;
    if (controlProp) return;
    const controller = new AbortController();
    void fetchControl(controller.signal);

    const t = window.setInterval(() => {
      void fetchControl(controller.signal);
    }, Math.max(750, pollIntervalMs));

    return () => {
      controller.abort();
      window.clearInterval(t);
    };
  }, [runId, controlProp, pollIntervalMs, fetchControl]);

  const postPatch = useCallback(
    async (patch: Partial<ControlState>) => {
      if (!runId) return;

      setSaving(true);
      setError(null);

      try {
        if (onUpdate) {
          const maybe = await onUpdate(patch);
          if (maybe && typeof maybe === "object") setControl(maybe as ControlState);
          else if (!controlProp) await fetchControl(); // reconcile
          setSaving(false);
          return;
        }

        // Default endpoint contract from earlier UI spec: POST /api/runs/{runId}/controls
        const urls = [
          `${base}/api/runs/${encodeURIComponent(runId)}/controls`,
          `${base}/api/runs/${encodeURIComponent(runId)}/control`,
        ];

        let lastErr: unknown = null;
        for (const url of urls) {
          try {
            const data = await fetchJson(url, { method: "POST", body: JSON.stringify(patch) });

            // If backend returns JSON object, use it.
            if (data && typeof data === "object") {
              setControl(data as ControlState);
              setSaving(false);

              // Best-effort logging of GUI control actions into run events stream (FR-6).
              // Use both msg and event for compatibility with differing parsers.
              try {
                await fetchJson(`${base}/api/runs/${encodeURIComponent(runId)}/log`, {
                  method: "POST",
                  body: JSON.stringify({ msg: "gui_control", event: "gui_control", fields: { patch } }),
                });
              } catch {
                // ignore
              }
              return;
            }

            // If backend returns 204/null/text, treat as success and refetch.
            if (!controlProp) {
              await fetchControl();
            }
            setSaving(false);

            // Best-effort logging (same as above; ignore failures).
            try {
              await fetchJson(`${base}/api/runs/${encodeURIComponent(runId)}/log`, {
                method: "POST",
                body: JSON.stringify({ msg: "gui_control", event: "gui_control", fields: { patch } }),
              });
            } catch {
              // ignore
            }

            return;
          } catch (e) {
            if (e instanceof DOMException && e.name === "AbortError") {
              setSaving(false);
              return;
            }
            lastErr = e;
          }
        }

        throw lastErr instanceof Error ? lastErr : new Error("Failed to update controls");
      } catch (e) {
        setError(e instanceof Error ? e.message : "Failed to update controls");
        setSaving(false);
      }
    },
    [runId, onUpdate, base, controlProp, fetchControl]
  );

  const acknowledged = useMemo(() => {
    const seq = typeof control.seq === "number" ? control.seq : undefined;
    const ack = typeof control.ack_seq === "number" ? control.ack_seq : undefined;
    if (typeof seq !== "number" || typeof ack !== "number") return null;
    return ack >= seq;
  }, [control.seq, control.ack_seq]);

  const extras = useMemo(() => getExtras(control), [control]);

  const validateWriteEvery = (text: string): { value: number | null; error?: string } => {
    const s = text.trim();
    if (!s) return { value: null };
    const n = Number(s);
    if (!Number.isFinite(n) || !Number.isInteger(n)) return { value: null, error: "write_every must be an integer (>= 1) or empty" };
    if (n < 1) return { value: null, error: "write_every must be >= 1 (or empty for null)" };
    return { value: n };
  };

  const weValidation = validateWriteEvery(writeEveryText);

  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "baseline" }}>
        <h3 style={{ margin: 0 }}>Run Controls</h3>
        <div style={{ fontSize: 12, opacity: 0.75 }}>
          {loading ? "Loading…" : "Ready"}
          {saving ? <span> · Saving…</span> : null}
          {runId ? <span> · runId {runId}</span> : null}
        </div>
      </div>

      {error ? (
        <div style={{ fontSize: 12, color: "#8a0000", background: "#ffecec", border: "1px solid #ffbcbc", padding: 10, borderRadius: 6 }}>
          {error}
        </div>
      ) : null}

      <div
        style={{
          border: "1px solid rgba(0,0,0,0.12)",
          borderRadius: 8,
          background: "#fff",
          padding: 10,
          display: "grid",
          gridTemplateColumns: "1fr",
          gap: 12,
        }}
      >
        {/* State summary */}
        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
          <div style={{ fontSize: 12, opacity: 0.85 }}>
            pause: <strong>{control.pause ? "true" : "false"}</strong>
          </div>
          <div style={{ fontSize: 12, opacity: 0.85 }}>
            terminate: <strong>{control.terminate ? "true" : "false"}</strong>
          </div>
          <div style={{ fontSize: 12, opacity: 0.85 }}>
            write_every: <strong>{control.write_every == null ? "null" : String(control.write_every)}</strong>
          </div>
          <div style={{ fontSize: 12, opacity: 0.85 }}>
            snapshot: <strong>{control.snapshot == null ? "null" : String(control.snapshot)}</strong>
          </div>
        </div>

        <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "center" }}>
          <div style={{ fontSize: 12, opacity: 0.85 }}>
            seq: <strong>{typeof control.seq === "number" ? control.seq : "—"}</strong>
          </div>
          <div style={{ fontSize: 12, opacity: 0.85 }}>
            ack_seq: <strong>{typeof control.ack_seq === "number" ? control.ack_seq : "—"}</strong>
          </div>
          <div style={{ fontSize: 12, opacity: 0.85 }}>
            ts: <strong>{formatEpochSeconds(control.ts)}</strong>
          </div>
          {acknowledged != null ? (
            <div
              style={{
                fontSize: 12,
                padding: "2px 8px",
                borderRadius: 999,
                border: "1px solid rgba(0,0,0,0.15)",
                background: acknowledged ? "#eef9f0" : "#fff7e6",
                color: acknowledged ? "#163" : "#5a3a00",
              }}
              title="Acknowledged when ack_seq >= seq (repo ControlState handshake field)"
            >
              {acknowledged ? "acknowledged" : "pending"}
            </div>
          ) : null}
        </div>

        {/* Controls */}
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center" }}>
          <button
            type="button"
            onClick={() => void postPatch({ pause: true })}
            disabled={!!disabled || !runId || saving || control.pause === true}
            style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid rgba(0,0,0,0.2)", background: "#fff" }}
          >
            Pause
          </button>

          <button
            type="button"
            onClick={() => void postPatch({ pause: false })}
            disabled={!!disabled || !runId || saving || control.pause === false}
            style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid rgba(0,0,0,0.2)", background: "#fff" }}
          >
            Resume
          </button>

          <button
            type="button"
            onClick={() => {
              // Confirm to avoid accidental kills.
              if (!confirmTerminate) {
                setConfirmTerminate(true);
                window.setTimeout(() => setConfirmTerminate(false), 2500);
                return;
              }
              setConfirmTerminate(false);
              void postPatch({ terminate: true });
            }}
            disabled={!!disabled || !runId || saving || control.terminate === true}
            style={{
              padding: "6px 10px",
              borderRadius: 6,
              border: "1px solid rgba(0,0,0,0.2)",
              background: confirmTerminate ? "#ffecec" : "#fff",
              color: confirmTerminate ? "#6a0000" : undefined,
            }}
            title="Terminate gracefully via control.json protocol (FR-6). Click twice to confirm."
          >
            {confirmTerminate ? "Confirm Terminate" : "Terminate"}
          </button>

          <button
            type="button"
            onClick={() => {
              const token = generateSnapshotToken();
              // FR-6 + repo schema: snapshot is string token; never boolean.
              void postPatch({ snapshot: token });
            }}
            disabled={!!disabled || !runId || saving}
            style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid rgba(0,0,0,0.2)", background: "#fff" }}
            title="Request a one-shot snapshot (writes snapshot=<unique string token>)"
          >
            Snapshot
          </button>
        </div>

        {/* write_every editor */}
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: 10, alignItems: "center" }}>
          <label style={{ display: "grid", gap: 6 }}>
            <span style={{ fontSize: 12, opacity: 0.8 }}>
              write_every (empty = null)
              <span style={{ marginLeft: 8, opacity: 0.7 }} title=">=1 or null per repo ControlState schema">
                controls frame/iter artifact cadence
              </span>
            </span>
            <input
              type="text"
              inputMode="numeric"
              value={writeEveryText}
              onChange={(e) => setWriteEveryText(e.target.value)}
              disabled={!!disabled || !runId || saving}
              placeholder="e.g., 10"
              style={{ padding: "6px 8px", borderRadius: 6, border: "1px solid rgba(0,0,0,0.18)" }}
              aria-invalid={!!weValidation.error}
            />
            {weValidation.error ? <span style={{ fontSize: 12, color: "#8a0000" }}>{weValidation.error}</span> : null}
          </label>

          <button
            type="button"
            onClick={() => void postPatch({ write_every: weValidation.value })}
            disabled={!!disabled || !runId || saving || !!weValidation.error}
            style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid rgba(0,0,0,0.2)", background: "#fff", height: 34 }}
            title="Apply write_every update via backend (browser does not write control.json directly)."
          >
            Apply
          </button>
        </div>

        {/* Advanced / extras */}
        <details>
          <summary style={{ cursor: "pointer", userSelect: "none" }}>Advanced</summary>
          <div style={{ marginTop: 8, display: "grid", gap: 8 }}>
            <div style={{ fontSize: 12, opacity: 0.8 }}>
              version: <strong>{typeof control.version === "number" ? control.version : "—"}</strong>
            </div>
            <div style={{ fontSize: 12, opacity: 0.75 }}>
              Extras are preserved by repo writers (ControlState allows unknown keys; write_controls merges and writes atomically).
            </div>
            <pre style={{ margin: 0, padding: 10, background: "rgba(0,0,0,0.04)", borderRadius: 6, overflowX: "auto", fontSize: 12 }}>
              {JSON.stringify(extras, null, 2)}
            </pre>
          </div>
        </details>

        {!controlProp && runId ? (
          <div style={{ display: "flex", justifyContent: "flex-end" }}>
            <button
              type="button"
              onClick={() => void fetchControl()}
              disabled={saving}
              style={{ padding: "6px 10px", borderRadius: 6, border: "1px solid rgba(0,0,0,0.2)", background: "#fff" }}
              title="Fetch current control state"
            >
              Refresh
            </button>
          </div>
        ) : null}

        <div style={{ fontSize: 12, opacity: 0.7 }}>
          Browser sends partial control updates to backend; backend must write <code>control.json</code> using repo protocol (atomic write + seq/ts management) per FR-6.
          Last fetched:{" "}
          <strong>{lastFetchedAtRef.current ? new Date(lastFetchedAtRef.current).toISOString() : "—"}</strong>
        </div>
      </div>
    </section>
  );
}

export default ControlPanel;
