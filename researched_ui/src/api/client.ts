import type {
  ApiEnvelope,
  ApiErrorDetails,
  ArtifactInfo,
  CompareResponse,
  ControlState,
  FrameEvent,
  LaunchRunRequest,
  LaunchRunResponse,
  PresetSummary,
  RunDetail,
  RunSummary,
} from "../types";

/**
 * ResearchED API client.
 *
 * Design Doc:
 * - §3.1: UI talks to Python backend via HTTP + WebSocket.
 * - FR-5: realtime event/frame streaming over WebSocket.
 * - FR-6: control panel posts ControlState updates (snapshot is a string token).
 *
 * Repo (Bobtheotherone/Electrodrive) verified integration points:
 * - ResearchED backend mounts REST under /api/v1 and WebSocket under /ws (electrodrive/researched/app.py).
 * - WS events merge events.jsonl + evidence_log.jsonl + train_log.jsonl + metrics.jsonl and emit canonical records (electrodrive/researched/ws.py).
 */

export class ApiError extends Error {
  public readonly status?: number;
  public readonly url?: string;
  public readonly details?: ApiErrorDetails;

  constructor(message: string, opts?: { status?: number; url?: string; details?: ApiErrorDetails }) {
    super(message);
    this.name = "ApiError";
    this.status = opts?.status;
    this.url = opts?.url;
    this.details = opts?.details;
  }
}

function normalizeBase(base: string): string {
  const b = (base ?? "").trim();
  if (!b) return "";
  return b.replace(/\/+$/, "");
}

function uniq(arr: string[]): string[] {
  const out: string[] = [];
  const seen = new Set<string>();
  for (const s of arr) {
    const v = s.trim();
    if (!v) continue;
    if (seen.has(v)) continue;
    seen.add(v);
    out.push(v);
  }
  return out;
}

function getConfiguredApiBase(): string {
  const env = (import.meta.env.VITE_API_BASE as string | undefined)?.trim();
  // Default to repo shape: /api/v1 (still same-origin). See electrodrive/researched/app.py.
  return normalizeBase(env && env.length ? env : "/api/v1");
}

let cachedApiBase: string | null = null;

function apiBaseCandidates(): string[] {
  const configured = normalizeBase(getConfiguredApiBase());
  // Backward-compat candidates for the "assumed endpoints" shape.
  return uniq([configured, "/api/v1", "/api"]);
}

function join(base: string, path: string): string {
  const b = normalizeBase(base);
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${b}${p}`;
}

async function readBody(resp: Response): Promise<unknown> {
  const ct = (resp.headers.get("content-type") ?? "").toLowerCase();
  if (resp.status === 204) return null;
  if (ct.includes("application/json")) {
    try {
      return await resp.json();
    } catch {
      return null;
    }
  }
  try {
    const text = await resp.text();
    return text || null;
  } catch {
    return null;
  }
}

function deriveMessage(status: number, body: unknown): string {
  if (typeof body === "string" && body.trim()) return body;
  if (body && typeof body === "object") {
    const b = body as Record<string, unknown>;
    const detail = b.detail ?? b.message ?? b.error;
    if (typeof detail === "string" && detail.trim()) return detail;
    try {
      return JSON.stringify(body);
    } catch {
      // ignore
    }
  }
  return `Request failed with status ${status}`;
}

/**
 * Low-level fetch helper.
 *
 * - Parses JSON when content-type is application/json.
 * - Throws ApiError on non-2xx.
 * - Supports AbortSignal.
 */
export async function fetchJson<T>(url: string, init: RequestInit & { signal?: AbortSignal } = {}): Promise<T> {
  const headers: HeadersInit = {
    Accept: "application/json",
    ...(init.headers ?? {}),
  };

  const resp = await fetch(url, {
    ...init,
    headers,
    credentials: init.credentials ?? "same-origin",
  });

  const body = await readBody(resp);

  if (!resp.ok) {
    const msg = deriveMessage(resp.status, body);
    const details: ApiErrorDetails =
      body && typeof body === "object" ? (body as ApiErrorDetails) : { raw: body };
    throw new ApiError(msg, { status: resp.status, url, details });
  }

  return body as T;
}

async function fetchFromApiBases<T>(
  path: string,
  init: RequestInit & { signal?: AbortSignal } = {},
): Promise<T> {
  const candidates = apiBaseCandidates();
  const first = cachedApiBase ? [cachedApiBase, ...candidates] : candidates;
  const bases = uniq(first.map(normalizeBase));

  let lastErr: unknown = null;

  for (const base of bases) {
    const url = join(base, path);
    try {
      const data = await fetchJson<T>(url, init);
      cachedApiBase = base;
      return data;
    } catch (e) {
      lastErr = e;
      // Only fall back on endpoint-shape errors.
      if (e instanceof ApiError && (e.status === 404 || e.status === 405)) {
        continue;
      }
      throw e;
    }
  }

  throw lastErr ?? new ApiError("All API base candidates failed");
}

function unwrapEnvelope<T>(val: unknown): T {
  // Some backends return { ok, data } or { data }. Keep tolerant.
  if (val && typeof val === "object") {
    const obj = val as ApiEnvelope<T>;
    if ("data" in obj) return obj.data as T;
  }
  return val as T;
}

export async function listRuns(opts: { signal?: AbortSignal } = {}): Promise<RunSummary[]> {
  const raw = await fetchFromApiBases<unknown>("/runs", { method: "GET", signal: opts.signal });
  const val = unwrapEnvelope<unknown>(raw);
  if (Array.isArray(val)) return val as RunSummary[];
  if (val && typeof val === "object" && Array.isArray((val as any).runs)) return (val as any).runs as RunSummary[];
  return [];
}

export async function getRun(runId: string, opts: { signal?: AbortSignal } = {}): Promise<RunDetail> {
  const rid = encodeURIComponent(String(runId));
  const raw = await fetchFromApiBases<unknown>(`/runs/${rid}`, { method: "GET", signal: opts.signal });
  return unwrapEnvelope<RunDetail>(raw);
}

export async function launchRun(req: LaunchRunRequest, opts: { signal?: AbortSignal } = {}): Promise<LaunchRunResponse> {
  const raw = await fetchFromApiBases<unknown>("/runs", {
    method: "POST",
    signal: opts.signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req ?? {}),
  });
  const val = unwrapEnvelope<any>(raw);
  // Back-compat: accept runId or run_id.
  const run_id = typeof val?.run_id === "string" ? val.run_id : typeof val?.runId === "string" ? val.runId : "";

  if (!run_id.trim()) {
    throw new ApiError("Launch failed: backend did not return run_id", { url: "/runs" });
  }

  return { run_id };
}

export async function postControl(
  runId: string,
  patch: Partial<ControlState>,
  opts: { signal?: AbortSignal } = {},
): Promise<ControlState> {
  const rid = encodeURIComponent(String(runId));
  const raw = await fetchFromApiBases<unknown>(`/runs/${rid}/control`, {
    method: "POST",
    signal: opts.signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch ?? {}),
  });
  return unwrapEnvelope<ControlState>(raw);
}

export async function listArtifacts(runId: string, opts: { signal?: AbortSignal } = {}): Promise<ArtifactInfo[]> {
  const rid = encodeURIComponent(String(runId));
  const raw = await fetchFromApiBases<unknown>(`/runs/${rid}/artifacts`, {
    method: "GET",
    signal: opts.signal,
  });
  const val = unwrapEnvelope<any>(raw);
  if (Array.isArray(val)) return val as ArtifactInfo[];
  if (val && typeof val === "object" && Array.isArray((val as any).artifacts)) return (val as any).artifacts as ArtifactInfo[];
  return [];
}

export async function compareRuns(
  runIds: string[],
  opts: { signal?: AbortSignal } = {},
): Promise<CompareResponse> {
  const params = new URLSearchParams();
  for (const r of runIds) params.append("r", r);
  const raw = await fetchFromApiBases<unknown>(`/compare?${params.toString()}`, {
    method: "GET",
    signal: opts.signal,
  });
  return unwrapEnvelope<CompareResponse>(raw);
}

export async function listPresets(opts: { signal?: AbortSignal } = {}): Promise<PresetSummary[]> {
  const raw = await fetchFromApiBases<unknown>("/presets", { method: "GET", signal: opts.signal });
  const val = unwrapEnvelope<any>(raw);
  if (Array.isArray(val)) return val as PresetSummary[];
  if (val && typeof val === "object" && Array.isArray((val as any).presets)) return (val as any).presets as PresetSummary[];
  return [];
}

export async function createPreset(
  preset: Omit<PresetSummary, "id"> & Partial<Pick<PresetSummary, "id">>,
  opts: { signal?: AbortSignal } = {},
): Promise<PresetSummary> {
  const raw = await fetchFromApiBases<unknown>("/presets", {
    method: "POST",
    signal: opts.signal,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(preset ?? {}),
  });
  return unwrapEnvelope<PresetSummary>(raw);
}

/* ----------------------------- WebSocket helpers ---------------------------- */

function restBaseForWsOrigin(): string {
  // Use configured API base to determine origin (http->ws, https->wss).
  const base = getConfiguredApiBase();
  // If API base is relative (/api/v1), resolve against window.origin.
  try {
    if (base.startsWith("http://") || base.startsWith("https://")) return base;
    return new URL(base, window.location.origin).toString();
  } catch {
    return window.location.origin;
  }
}

function wsOriginFromRest(): string {
  const rest = restBaseForWsOrigin();
  try {
    const u = new URL(rest);
    const proto = u.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${u.host}`;
  } catch {
    const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
    return `${proto}//${window.location.host}`;
  }
}

function makeWsUrl(path: string): string {
  // If already absolute ws(s) URL, return as-is.
  if (path.startsWith("ws://") || path.startsWith("wss://")) return path;

  const origin = wsOriginFromRest();
  const p = path.startsWith("/") ? path : `/${path}`;
  return `${origin}${p}`;
}

type WsBaseHandlers = {
  onOpen?: () => void;
  onClose?: (ev: CloseEvent) => void;
  onError?: (ev: Event) => void;
  maxRetries?: number;
};

type RunEventsHandlers = WsBaseHandlers & {
  onEvent?: (ev: any) => void;
};

type RunFramesHandlers = WsBaseHandlers & {
  onFrame?: (fe: FrameEvent) => void;
};

export type WsConnection = { close: () => void };

/**
 * Connect to run events stream.
 *
 * Default (assumed) endpoint: WS /api/ws/runs/{runId}/events
 * Repo endpoint: WS /ws/runs/{runId}/events (electrodrive/researched/ws.py, mounted under /ws in app.py).
 */
export function connectRunEvents(runId: string, handlers: RunEventsHandlers): WsConnection {
  const rid = encodeURIComponent(String(runId));
  const candidates = [
    `/api/ws/runs/${rid}/events`, // assumed shape
    `/ws/runs/${rid}/events`, // repo shape
    `/api/v1/ws/runs/${rid}/events`, // extra tolerance
  ];
  return connectWsWithFallback(candidates, handlers, (msg) => {
    if (!handlers.onEvent) return;
    if (!msg || typeof msg !== "object") return;

    const m: any = msg;

    // Unwrap common server envelopes: {type:"event", record:{...}} or {type:"event", data:{...}}
    const rec =
      m.type === "event"
        ? (m.record ?? m.data ?? m.payload ?? m.event_record ?? m)
        : m;

    if (!rec || typeof rec !== "object") return;

    // Canonical record shape: { event: string, fields: object }
    if (typeof rec.event === "string" && rec.fields && typeof rec.fields === "object") {
      handlers.onEvent(rec);
      return;
    }

    // Some servers might send canonical records without wrapper.
    if (typeof rec.event === "string" && (!rec.fields || typeof rec.fields !== "object")) {
      handlers.onEvent({ ...rec, fields: {} });
    }
  });
}

/**
 * Connect to run frames stream.
 *
 * Default (assumed) endpoint: WS /api/ws/runs/{runId}/frames
 * Repo endpoint: WS /ws/runs/{runId}/frames (electrodrive/researched/ws.py sends {type:"frame", bytes_b64,...}).
 */
export function connectRunFrames(runId: string, handlers: RunFramesHandlers): WsConnection {
  const rid = encodeURIComponent(String(runId));
  const candidates = [
    `/api/ws/runs/${rid}/frames`, // assumed shape
    `/ws/runs/${rid}/frames`, // repo shape
    `/api/v1/ws/runs/${rid}/frames`, // extra tolerance
  ];
  return connectWsWithFallback(candidates, handlers, (msg) => {
    if (!handlers.onFrame) return;
    if (!msg || typeof msg !== "object") return;

    const m = msg as any;

    // Pass through already-normalized shapes
    if ((m.type === "frame_added" || m.type === "frame_updated" || m.type === "frame_latest") && m.frame) {
      handlers.onFrame(m as FrameEvent);
      return;
    }
    if (m.type === "error") {
      handlers.onFrame({ type: "error", message: String(m.message ?? "frame error"), frame: m.frame } as any);
      return;
    }

    // Repo shape: {type:"frame", name, index, mtime, bytes_b64}
    if (m.type === "frame" && typeof m.name === "string") {
      const idx = typeof m.index === "number" ? m.index : typeof m.index === "string" ? Number(m.index) : null;
      const bytes = typeof m.bytes_b64 === "string" ? m.bytes_b64 : null;
      const url = bytes ? `data:image/png;base64,${bytes}` : undefined;

      handlers.onFrame({
        type: "frame_added",
        frame: {
          index: typeof idx === "number" && Number.isFinite(idx) ? idx : -1,
          path: m.name,
          name: m.name,
          mtime: typeof m.mtime === "number" ? m.mtime : undefined,
          bytes_b64: bytes ?? undefined,
          url,
        },
      });
      return;
    }

    // Fallback: if server sends {frame:{...}} without type.
    if (m.frame && typeof m.frame === "object") {
      handlers.onFrame({ type: "frame", frame: m.frame } as FrameEvent);
    }
  });
}

function connectWsWithFallback(
  pathCandidates: string[],
  handlers: WsBaseHandlers,
  onJson: (msg: unknown) => void,
): WsConnection {
  const maxRetries = typeof handlers.maxRetries === "number" ? Math.max(0, handlers.maxRetries) : 3;

  let manualClose = false;
  let opened = false;
  let activeSocket: WebSocket | null = null;

  let candidateIndex = 0;
  let lastGoodIndex: number | null = null;
  let reconnects = 0;

  const cleanup = () => {
    if (!activeSocket) return;
    try {
      activeSocket.onopen = null;
      activeSocket.onmessage = null;
      activeSocket.onclose = null;
      activeSocket.onerror = null;
      activeSocket.close();
    } catch {
      // ignore
    }
    activeSocket = null;
  };

  const scheduleReconnect = (delayMs: number) => {
    window.setTimeout(() => {
      if (manualClose) return;
      void connect();
    }, delayMs);
  };

  const connect = async (): Promise<void> => {
    cleanup();

    const len = pathCandidates.length;
    const idx = lastGoodIndex ?? candidateIndex;
    const path = pathCandidates[Math.min(idx, len - 1)];
    const url = makeWsUrl(path);

    try {
      activeSocket = new WebSocket(url);
    } catch {
      // If URL is invalid for this candidate, immediately try next.
      lastGoodIndex = null;
      if (candidateIndex < len - 1) {
        candidateIndex += 1;
        return connect();
      }
      return;
    }

    opened = false;

    activeSocket.onopen = () => {
      opened = true;
      lastGoodIndex = idx;
      candidateIndex = idx; // keep candidate aligned to last-good
      reconnects = 0;
      handlers.onOpen?.();
    };

    activeSocket.onerror = (ev) => {
      handlers.onError?.(ev);
      // Do not advance candidates here; onclose decides whether it never opened.
    };

    activeSocket.onclose = (ev) => {
      handlers.onClose?.(ev);

      if (manualClose) return;

      // If it never opened, try next candidate quickly (without skipping).
      if (!opened) {
        const next = idx + 1;
        lastGoodIndex = null;
        if (next < len) {
          candidateIndex = next;
          scheduleReconnect(50);
          return;
        }
        // No more candidates; limited retry with backoff.
        if (reconnects < maxRetries) {
          const delay = 250 * Math.pow(2, reconnects);
          reconnects += 1;
          scheduleReconnect(delay);
        }
        return;
      }

      // If it opened before, attempt limited reconnect on the last-successful candidate.
      if (reconnects < maxRetries) {
        const delay = 250 * Math.pow(2, reconnects);
        reconnects += 1;
        scheduleReconnect(delay);
      }
    };

    activeSocket.onmessage = (ev) => {
      try {
        const data = JSON.parse(String(ev.data ?? "null"));
        onJson(data);
      } catch {
        // ignore non-JSON
      }
    };
  };

  void connect();

  return {
    close: () => {
      manualClose = true;
      cleanup();
    },
  };
}
