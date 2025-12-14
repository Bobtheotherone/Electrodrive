import React, { useCallback, useEffect, useId, useMemo, useRef, useState } from "react";

// Source notes:
// - Design Doc: FR-4 (robust log ingestion/normalization), FR-5 (live log stream UI).
// - Repo: electrodrive/utils/logging.py (JsonlLogger writes JSONL with top-level keys: ts (ISO string), level, msg + arbitrary fields; default file is events.jsonl) [~L136-200].
// - Repo: electrodrive/viz/iter_viz.py (_event_name fallback uses event || msg || message; legacy logs may place structured JSON inside message string) [~L133-144].

export type AnyLogRecord = Record<string, unknown>;

type NormalizedLevel = "DEBUG" | "INFO" | "WARNING" | "ERROR" | "CRITICAL" | "TRACE" | "UNKNOWN";

type EventFieldUsed = "event" | "msg" | "message" | "parsed-json" | "none";

export interface NormalizedLogRecord {
  id: number;
  ts?: string;
  t?: number; // epoch seconds (best-effort)
  level: NormalizedLevel | string;
  eventName: string;
  message: string;
  raw: AnyLogRecord;
  fields: Record<string, unknown>;
  eventFieldUsed: EventFieldUsed;
}

export interface LogStreamProps {
  /** If provided, component is purely presentational and will not open its own WebSocket. */
  records?: ReadonlyArray<AnyLogRecord>;

  /** If provided (and `records` not provided), component will try to stream logs via WebSocket. */
  runId?: string;

  /** Override WebSocket URL to stream log records. If not set, derives from apiBase and tries common paths. */
  wsUrl?: string;

  /** Optional API base (e.g., http://localhost:8000). Used only for deriving wsUrl when runId is provided. */
  apiBase?: string;

  /** Render height in px for the scroll container. */
  height?: number;

  /** Keep only the last N records for rendering/windowing. */
  maxRows?: number;

  /** Initial level filter selection. Defaults to DEBUG/INFO/WARNING/ERROR enabled. */
  initialLevels?: ReadonlyArray<string>;

  /** Initial substring filter. */
  initialSubstring?: string;

  /** Disable interaction (filters still visible). */
  disabled?: boolean;

  /** Optional title shown above controls. */
  title?: string;

  /** Allow arbitrary extra props to avoid breaking unknown call-sites. */
  [key: string]: unknown;
}

const DEFAULT_LEVELS: ReadonlyArray<NormalizedLevel> = ["DEBUG", "INFO", "WARNING", "ERROR"];
const DEFAULT_MAX_ROWS = 2000;

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

function normalizeLevel(level: unknown): string {
  const raw = (typeof level === "string" ? level : "").trim();
  if (!raw) return "UNKNOWN";
  const up = raw.toUpperCase();
  if (up === "WARN") return "WARNING";
  if (up === "ERR") return "ERROR";
  if (up === "FATAL") return "CRITICAL";
  return up;
}

function parseEpochSecondsFromTs(ts: unknown): number | undefined {
  if (typeof ts === "number" && Number.isFinite(ts)) {
    // Could be seconds or ms; assume seconds if plausible, else ms->seconds.
    if (ts > 1e12) return ts / 1000;
    if (ts > 1e9) return ts;
    return ts;
  }
  if (typeof ts === "string") {
    const s = ts.trim();
    if (!s) return undefined;
    const asNum = Number(s);
    if (Number.isFinite(asNum)) {
      if (asNum > 1e12) return asNum / 1000;
      if (asNum > 1e9) return asNum;
    }
    const d = new Date(s);
    const ms = d.getTime();
    if (Number.isFinite(ms)) return ms / 1000;
  }
  return undefined;
}

function safeJsonParse(text: string): unknown | undefined {
  const s = text.trim();
  if (!(s.startsWith("{") && s.endsWith("}"))) return undefined;
  try {
    return JSON.parse(s);
  } catch {
    return undefined;
  }
}

function deriveEventNameAndFields(raw: AnyLogRecord): {
  eventName: string;
  message: string;
  eventFieldUsed: EventFieldUsed;
  fields: Record<string, unknown>;
} {
  // Canonical design doc format may include a nested "fields" object; repo JsonlLogger uses top-level kv.
  const nestedFields = isPlainObject(raw.fields) ? raw.fields : undefined;

  let eventFieldUsed: EventFieldUsed = "none";
  let eventName = "";
  if (typeof raw.event === "string" && raw.event.trim()) {
    eventName = raw.event.trim();
    eventFieldUsed = "event";
  } else if (typeof raw.msg === "string" && raw.msg.trim()) {
    eventName = raw.msg.trim();
    eventFieldUsed = "msg";
  } else if (typeof raw.message === "string" && raw.message.trim()) {
    eventName = raw.message.trim();
    eventFieldUsed = "message";
  }

  // FR-4: Handle legacy pattern where the message string itself is JSON containing {"event": "..."}.
  // (Design Doc FR-4 + repo learn/train embed JSON in message; iter_viz notes msg fallback.)
  let message = typeof raw.msg === "string" ? raw.msg : typeof raw.message === "string" ? raw.message : "";
  let parsedFromMessage: Record<string, unknown> | undefined;
  if (typeof message === "string" && message.trim().startsWith("{")) {
    const parsed = safeJsonParse(message);
    if (isPlainObject(parsed)) {
      parsedFromMessage = parsed;
      if ((!eventName || eventFieldUsed === "msg" || eventFieldUsed === "message") && typeof parsed.event === "string" && parsed.event.trim()) {
        eventName = parsed.event.trim();
        eventFieldUsed = "parsed-json";
      }
    }
  }

  // Combine fields:
  // - Start with nested "fields" if present (design doc canonical record)
  // - Then add parsed-from-message JSON fields (legacy stdlib logging embedding)
  // - Then add all top-level keys except the known ones.
  const fields: Record<string, unknown> = {};
  if (nestedFields) Object.assign(fields, nestedFields);
  if (parsedFromMessage) Object.assign(fields, parsedFromMessage);

  for (const [k, v] of Object.entries(raw)) {
    if (k === "ts" || k === "t" || k === "level" || k === "msg" || k === "message" || k === "event" || k === "fields") continue;
    fields[k] = v;
  }

  return { eventName: eventName || "", message, eventFieldUsed, fields };
}

function normalizeRecord(raw: AnyLogRecord, id: number): NormalizedLogRecord {
  const ts = typeof raw.ts === "string" ? raw.ts : undefined;
  const t = typeof raw.t === "number" ? raw.t : parseEpochSecondsFromTs(raw.ts);
  const level = normalizeLevel(raw.level);
  const { eventName, message, eventFieldUsed, fields } = deriveEventNameAndFields(raw);
  return {
    id,
    ts,
    t,
    level,
    eventName,
    message,
    raw,
    fields,
    eventFieldUsed,
  };
}

function formatTs(rec: NormalizedLogRecord): string {
  if (rec.ts) return rec.ts;
  if (typeof rec.t === "number" && Number.isFinite(rec.t)) {
    const d = new Date(rec.t * 1000);
    return d.toISOString();
  }
  return "";
}

function pickNumericSummary(fields: Record<string, unknown>): Array<[string, number]> {
  const preferredKeys = [
    "iter",
    "iters",
    "step",
    "k",
    "resid",
    "resid_precond",
    "resid_true",
    "resid_precond_l2",
    "resid_true_l2",
    "t_iter_ms",
    "dt_ms",
    "ms",
    "seconds",
  ];
  const out: Array<[string, number]> = [];
  const seen = new Set<string>();
  const tryPush = (k: string, v: unknown) => {
    if (seen.has(k)) return;
    const n = typeof v === "number" ? v : typeof v === "string" ? Number(v) : NaN;
    if (Number.isFinite(n)) {
      out.push([k, n]);
      seen.add(k);
    }
  };

  for (const k of preferredKeys) {
    tryPush(k, fields[k]);
    if (out.length >= 3) return out;
  }
  // Fallback: add up to 3 numeric keys in insertion order.
  for (const [k, v] of Object.entries(fields)) {
    tryPush(k, v);
    if (out.length >= 3) return out;
  }
  return out;
}

function getEnvApiBase(): string | undefined {
  // Avoid hard dependency on Vite typing; works in plain TS too.
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
  const env = getEnvApiBase();
  const base = (apiBase ?? env ?? "").trim();
  if (base) return base.replace(/\/+$/, "");
  if (typeof window !== "undefined" && window.location?.origin) return window.location.origin;
  return "";
}

function toWsBase(restBase: string): string {
  const b = restBase.trim();
  if (!b) {
    if (typeof window !== "undefined" && window.location) {
      const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
      return `${proto}//${window.location.host}`;
    }
    return "";
  }
  if (b.startsWith("https://")) return "wss://" + b.slice("https://".length);
  if (b.startsWith("http://")) return "ws://" + b.slice("http://".length);
  return b;
}

function joinWsUrl(wsBase: string, path: string): string {
  if (!wsBase) return path;
  if (path.startsWith("ws://") || path.startsWith("wss://")) return path;
  if (!path.startsWith("/")) return wsBase.replace(/\/+$/, "") + "/" + path;
  return wsBase.replace(/\/+$/, "") + path;
}

const badgeStyles: Record<string, React.CSSProperties> = {
  DEBUG: { background: "#eef", color: "#223", borderColor: "#ccd" },
  INFO: { background: "#eef9f0", color: "#163", borderColor: "#cfe9d5" },
  WARNING: { background: "#fff7e6", color: "#5a3a00", borderColor: "#ffe1aa" },
  ERROR: { background: "#ffecec", color: "#6a0000", borderColor: "#ffbcbc" },
  CRITICAL: { background: "#3b0000", color: "#fff", borderColor: "#3b0000" },
  UNKNOWN: { background: "#f3f3f3", color: "#333", borderColor: "#ddd" },
};

const LogRow = React.memo(function LogRow(props: { rec: NormalizedLogRecord; showJson: boolean }) {
  const { rec, showJson } = props;
  const levelUp = normalizeLevel(rec.level);
  const badge = badgeStyles[levelUp] ?? badgeStyles.UNKNOWN;
  const summary = pickNumericSummary(rec.fields);

  return (
    <div
      style={{
        padding: "6px 8px",
        borderBottom: "1px solid rgba(0,0,0,0.06)",
        fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, 'Liberation Mono', 'Courier New', monospace",
        fontSize: 12,
        lineHeight: 1.35,
      }}
    >
      <div style={{ display: "flex", gap: 8, alignItems: "baseline", flexWrap: "wrap" }}>
        <span style={{ opacity: 0.75, minWidth: 170 }}>{formatTs(rec)}</span>
        <span style={{ ...badge, border: "1px solid", borderRadius: 999, padding: "1px 7px", fontSize: 11 }}>
          {levelUp}
        </span>
        <span style={{ fontWeight: 600, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{rec.eventName || "(no event)"}</span>
        {summary.length > 0 ? (
          <span style={{ opacity: 0.85 }}>
            {summary.map(([k, v]) => (
              <span key={k} style={{ marginLeft: 10 }}>
                {k}={Number.isFinite(v) ? v.toPrecision(4) : String(v)}
              </span>
            ))}
          </span>
        ) : null}
      </div>

      {rec.message && rec.message !== rec.eventName ? (
        <div style={{ marginTop: 2, opacity: 0.9, whiteSpace: "pre-wrap", wordBreak: "break-word" }}>{rec.message}</div>
      ) : null}

      {showJson ? (
        <details style={{ marginTop: 6 }}>
          <summary style={{ cursor: "pointer", userSelect: "none" }}>Fields (event source: {rec.eventFieldUsed})</summary>
          <pre style={{ margin: "6px 0 0 0", padding: 8, background: "rgba(0,0,0,0.04)", overflowX: "auto" }}>
            {JSON.stringify(rec.fields, null, 2)}
          </pre>
        </details>
      ) : null}
    </div>
  );
});

export function LogStream(props: LogStreamProps) {
  const {
    records,
    runId,
    wsUrl,
    apiBase,
    height = 360,
    maxRows = DEFAULT_MAX_ROWS,
    initialLevels,
    initialSubstring,
    disabled,
    title = "Logs",
  } = props;

  const controlsId = useId();
  const listRef = useRef<HTMLDivElement | null>(null);

  const [levelEnabled, setLevelEnabled] = useState<Record<string, boolean>>(() => {
    const init = new Set((initialLevels ?? DEFAULT_LEVELS).map((l) => normalizeLevel(l)));
    const base: Record<string, boolean> = {};
    for (const l of DEFAULT_LEVELS) base[l] = init.has(l);
    // Default: include DEBUG/INFO/WARNING/ERROR enabled.
    return base;
  });
  const [substring, setSubstring] = useState<string>(initialSubstring ?? "");
  const [autoScroll, setAutoScroll] = useState<boolean>(true);
  const [paused, setPaused] = useState<boolean>(false);
  const [showJson, setShowJson] = useState<boolean>(false);

  // Internal streaming state (used only when `records` prop not provided).
  const [streamRecords, setStreamRecords] = useState<NormalizedLogRecord[]>([]);
  const [streamStatus, setStreamStatus] = useState<"idle" | "connecting" | "open" | "closed" | "error">("idle");
  const nextIdRef = useRef<number>(1);
  const pendingRawRef = useRef<AnyLogRecord[]>([]);
  const flushScheduledRef = useRef<boolean>(false);
  const pausedRef = useRef<boolean>(paused);
  const pendingWhilePausedRef = useRef<number>(0);
  const [pendingWhilePaused, setPendingWhilePaused] = useState<number>(0);

  useEffect(() => {
    pausedRef.current = paused;
    if (!paused) {
      // Flush any buffered records immediately when resuming.
      if (pendingRawRef.current.length > 0) {
        const raw = pendingRawRef.current.splice(0, pendingRawRef.current.length);
        const normalized = raw.map((r) => normalizeRecord(r, nextIdRef.current++));
        setStreamRecords((prev) => {
          const merged = prev.concat(normalized);
          const trimmed = merged.length > maxRows ? merged.slice(merged.length - maxRows) : merged;
          return trimmed;
        });
      }
      pendingWhilePausedRef.current = 0;
      setPendingWhilePaused(0);
    }
  }, [paused, maxRows]);

  // When paused, update the "pending while paused" counter at a low rate to avoid render storms.
  useEffect(() => {
    if (!paused) return;
    const t = window.setInterval(() => {
      setPendingWhilePaused(pendingWhilePausedRef.current);
    }, 250);
    return () => window.clearInterval(t);
  }, [paused]);

  const scheduleFlush = useCallback(
    (maxRowsLocal: number) => {
      if (flushScheduledRef.current) return;
      flushScheduledRef.current = true;

      // Batch updates to avoid re-rendering on every single WS message.
      requestAnimationFrame(() => {
        flushScheduledRef.current = false;
        if (pausedRef.current) return;

        const raw = pendingRawRef.current.splice(0, pendingRawRef.current.length);
        if (raw.length === 0) return;

        const normalized = raw.map((r) => normalizeRecord(r, nextIdRef.current++));
        setStreamRecords((prev) => {
          const merged = prev.concat(normalized);
          const trimmed = merged.length > maxRowsLocal ? merged.slice(merged.length - maxRowsLocal) : merged;
          return trimmed;
        });
      });
    },
    [setStreamRecords]
  );

  // Connect WebSocket only when no external records are provided.
  useEffect(() => {
    // Presentational mode: if records prop is provided (even empty), do not connect WS.
    if (records !== undefined) return;
    if (!runId && !wsUrl) return;

    let ws: WebSocket | null = null;
    let didFallback = false;
    let stopped = false;

    const base = resolveApiBase(apiBase);
    const wsBase = toWsBase(base);

    const candidates: string[] = [];
    if (wsUrl) {
      candidates.push(wsUrl);
    } else if (runId) {
      // Common endpoints used in earlier UI specs; try both, but avoid multiple concurrent sockets.
      candidates.push(joinWsUrl(wsBase, `/ws/runs/${encodeURIComponent(runId)}`));
      candidates.push(joinWsUrl(wsBase, `/api/ws/runs/${encodeURIComponent(runId)}/events`));
    }

    const connectAt = (idx: number) => {
      if (stopped) return;
      const url = candidates[idx];
      if (!url) return;

      setStreamStatus("connecting");
      try {
        ws = new WebSocket(url);
      } catch {
        setStreamStatus("error");
        return;
      }

      ws.onopen = () => {
        if (stopped) return;
        setStreamStatus("open");
      };

      ws.onerror = () => {
        if (stopped) return;
        setStreamStatus("error");
      };

      ws.onclose = () => {
        if (stopped) return;
        setStreamStatus("closed");
        // Fallback once to alternate endpoint if available.
        if (!didFallback && idx + 1 < candidates.length) {
          didFallback = true;
          connectAt(idx + 1);
        }
      };

      ws.onmessage = (ev: MessageEvent) => {
        if (stopped) return;
        let payload: unknown;
        try {
          payload = JSON.parse(String(ev.data));
        } catch {
          // Non-JSON lines: wrap as a message record.
          payload = { ts: new Date().toISOString(), level: "info", msg: String(ev.data) };
        }

        // Accept either:
        // - { type:"log", record: {...} }  (design doc WS suggestion)
        // - raw record {...} (ts/level/msg/etc)
        const rec: AnyLogRecord | undefined = (() => {
          if (isPlainObject(payload)) {
            if (payload.type === "log" && isPlainObject(payload.record)) return payload.record as AnyLogRecord;
            if (typeof payload.ts !== "undefined" || typeof payload.msg !== "undefined" || typeof payload.event !== "undefined") {
              return payload as AnyLogRecord;
            }
          }
          return undefined;
        })();

        if (!rec) return;

        if (pausedRef.current) {
          pendingRawRef.current.push(rec);
          // Keep pending bounded to avoid unbounded memory if paused for a long time.
          if (pendingRawRef.current.length > Math.max(5000, maxRows * 3)) {
            pendingRawRef.current.splice(0, pendingRawRef.current.length - Math.max(5000, maxRows * 3));
          }
          pendingWhilePausedRef.current += 1;
          return;
        }

        pendingRawRef.current.push(rec);
        // Bound pending buffer too.
        if (pendingRawRef.current.length > 2000) {
          pendingRawRef.current.splice(0, pendingRawRef.current.length - 2000);
        }
        scheduleFlush(maxRows);
      };
    };

    // Reset on new run/socket.
    setStreamRecords([]);
    nextIdRef.current = 1;
    pendingRawRef.current = [];
    pendingWhilePausedRef.current = 0;
    setPendingWhilePaused(0);
    connectAt(0);

    return () => {
      stopped = true;
      try {
        ws?.close();
      } catch {
        // ignore
      }
      ws = null;
      setStreamStatus("idle");
    };
  }, [records, runId, wsUrl, apiBase, scheduleFlush, maxRows]);

  // Normalize external records (presentational mode) or use streamed normalized records.
  const normalized: NormalizedLogRecord[] = useMemo(() => {
    const src = records ? Array.from(records) : undefined;
    if (src) {
      const sliced = src.length > maxRows ? src.slice(src.length - maxRows) : src;
      return sliced.map((r, i) => normalizeRecord(r as AnyLogRecord, i + 1));
    }
    return streamRecords;
  }, [records, streamRecords, maxRows]);

  const availableLevels = useMemo(() => {
    const set = new Set<string>(DEFAULT_LEVELS as ReadonlyArray<string>);
    for (const r of normalized) {
      set.add(normalizeLevel(r.level));
    }
    // Ensure stable order with defaults first.
    const ordered: string[] = [];
    for (const l of DEFAULT_LEVELS) ordered.push(l);
    for (const l of Array.from(set).sort()) {
      if (!ordered.includes(l)) ordered.push(l);
    }
    return ordered;
  }, [normalized]);

  // Ensure `levelEnabled` always has keys for any discovered levels, defaulting to true for non-default levels.
  useEffect(() => {
    setLevelEnabled((prev) => {
      let changed = false;
      const next: Record<string, boolean> = { ...prev };
      for (const l of availableLevels) {
        if (typeof next[l] !== "boolean") {
          next[l] = true;
          changed = true;
        }
      }
      return changed ? next : prev;
    });
  }, [availableLevels]);

  const filtered = useMemo(() => {
    const sub = substring.trim().toLowerCase();
    const enabled = levelEnabled;

    return normalized.filter((r) => {
      const lvl = normalizeLevel(r.level);
      if (enabled[lvl] === false) return false;
      if (!sub) return true;

      const hay = `${r.eventName} ${r.message}`.toLowerCase();
      if (hay.includes(sub)) return true;

      // Optional: also match against compact field string, but keep it cheap by only checking when substring is non-empty.
      // This helps with "iter=..." searches when iter is a structured field.
      for (const [k, v] of Object.entries(r.fields)) {
        if (String(k).toLowerCase().includes(sub)) return true;
        if (typeof v === "string" && v.toLowerCase().includes(sub)) return true;
        if (typeof v === "number" && String(v).includes(sub)) return true;
      }
      return false;
    });
  }, [normalized, substring, levelEnabled]);

  // Auto-scroll behavior: keep pinned to bottom when enabled.
  useEffect(() => {
    if (!autoScroll) return;
    const el = listRef.current;
    if (!el) return;
    // Only autoscroll when not paused; if paused, user is inspecting.
    if (paused) return;
    el.scrollTop = el.scrollHeight;
  }, [filtered.length, autoScroll, paused]);

  const toggleLevel = useCallback((lvl: string) => {
    setLevelEnabled((prev) => ({ ...prev, [lvl]: !prev[lvl] }));
  }, []);

  const headerHintId = `${controlsId}-hint`;

  return (
    <section aria-labelledby={`${controlsId}-title`} style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12, flexWrap: "wrap" }}>
        <h3 id={`${controlsId}-title`} style={{ margin: 0 }}>
          {title}
        </h3>
        {records === undefined && (runId || wsUrl) ? (
          <div style={{ fontSize: 12, opacity: 0.8 }}>
            Stream: <span style={{ fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace" }}>{streamStatus}</span>
            {paused && pendingWhilePaused > 0 ? <span> · {pendingWhilePaused} new while paused</span> : null}
          </div>
        ) : null}
      </div>

      <div
        id={headerHintId}
        style={{
          display: "grid",
          gridTemplateColumns: "1fr",
          gap: 10,
          padding: 10,
          border: "1px solid rgba(0,0,0,0.08)",
          borderRadius: 8,
          background: "rgba(0,0,0,0.02)",
        }}
      >
        <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          <fieldset style={{ border: "none", padding: 0, margin: 0, display: "flex", gap: 10, flexWrap: "wrap" }}>
            <legend style={{ fontSize: 12, opacity: 0.8, marginRight: 6 }}>Levels</legend>
            {availableLevels.map((lvl) => (
              <label key={lvl} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                <input
                  type="checkbox"
                  checked={levelEnabled[lvl] !== false}
                  onChange={() => toggleLevel(lvl)}
                  disabled={!!disabled}
                />
                {lvl}
              </label>
            ))}
          </fieldset>

          <label style={{ display: "inline-flex", alignItems: "center", gap: 8, fontSize: 12, flex: "1 1 280px" }}>
            <span style={{ opacity: 0.8 }}>Filter</span>
            <input
              type="text"
              value={substring}
              onChange={(e) => setSubstring(e.target.value)}
              placeholder="substring…"
              disabled={!!disabled}
              style={{ flex: 1, minWidth: 180, padding: "6px 8px", borderRadius: 6, border: "1px solid rgba(0,0,0,0.18)" }}
            />
          </label>

          <label style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12 }}>
            <input type="checkbox" checked={paused} onChange={(e) => setPaused(e.target.checked)} disabled={!!disabled} />
            Pause
          </label>

          <label style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12 }}>
            <input type="checkbox" checked={autoScroll} onChange={(e) => setAutoScroll(e.target.checked)} disabled={!!disabled} />
            Auto-scroll
          </label>

          <label style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12 }}>
            <input type="checkbox" checked={showJson} onChange={(e) => setShowJson(e.target.checked)} disabled={!!disabled} />
            Show JSON fields
          </label>
        </div>

        <div style={{ fontSize: 12, opacity: 0.75 }}>
          Showing <strong>{filtered.length}</strong> / {normalized.length} (kept last {maxRows}). Stream is tolerant of schema drift
          (event/msg/message + JSON-in-message) per FR-4.
        </div>
      </div>

      <div
        ref={listRef}
        role="log"
        aria-live="polite"
        aria-describedby={headerHintId}
        style={{
          height,
          overflow: "auto",
          border: "1px solid rgba(0,0,0,0.12)",
          borderRadius: 8,
          background: "#fff",
        }}
      >
        {filtered.length === 0 ? (
          <div style={{ padding: 12, fontSize: 12, opacity: 0.8 }}>No log records match current filters.</div>
        ) : (
          filtered.map((rec) => <LogRow key={rec.id} rec={rec} showJson={showJson} />)
        )}
      </div>
    </section>
  );
}

export default LogStream;
