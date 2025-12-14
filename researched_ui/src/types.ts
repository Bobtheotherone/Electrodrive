/**
 * Shared DTOs for ResearchED UI.
 *
 * Design Doc anchors:
 * - §5.1 Unified run manifest schema (RunManifest)
 * - §5.2 Canonical normalized event record schema (CanonicalEventRecord)
 * - §1.4 events.jsonl vs evidence_log.jsonl drift (LogCoverageSummary.files_ingested)
 * - FR-6 snapshot token semantics (ControlState.snapshot is a string token, not boolean)
 * - FR-9.6 log consumer audit panel (LogCoverageSummary diagnostics)
 *
 * Repo anchors (Bobtheotherone/Electrodrive):
 * - electrodrive/live/controls.py: ControlState fields + semantics:
 *   pause/terminate/write_every/snapshot + ts/version/seq/ack_seq and unknown-key preservation across merges.
 */

export type Workflow = "solve" | "images_discover" | "learn_train" | "fmm_suite" | (string & {});
export type RunStatus = "running" | "success" | "error" | "killed" | (string & {});
export type LogLevel = "debug" | "info" | "warning" | "error" | (string & {});
export type EventNameSource =
  | "event"
  | "msg"
  | "message"
  | "parsed_message_json"
  | "embedded_json"
  | "fallback"
  | "missing"
  | (string & {});

export type JsonObject = Record<string, unknown>;

export type SpecDigest = {
  // Produced by SpecInspector (Design Doc §3.2 + §5.1); kept extensible.
  [k: string]: unknown;
};

export interface RunManifest {
  // Design Doc §5.1 required fields:
  run_id: string;
  workflow: Workflow;
  started_at: string;
  ended_at?: string | null;
  status: RunStatus;

  git: {
    sha: string | null;
    branch?: string | null;
    dirty?: boolean | null;
    diff_summary?: string | null;
    [k: string]: unknown;
  };

  env: {
    python_version?: string;
    torch_version?: string;
    device?: string;
    dtype?: string;
    host?: string;
    [k: string]: unknown;
  };

  inputs: {
    spec_path?: string | null;
    config?: unknown;
    config_path?: string | null;
    command?: string[];
    [k: string]: unknown;
  };

  outputs: {
    metrics_json?: string | null;
    events_jsonl?: string | null;
    evidence_log_jsonl?: string | null;
    viz_dir?: string | null;
    plots_dir?: string | null;
    report_html?: string | null;
    [k: string]: unknown;
  };

  gate?: {
    gate1_status?: string | null;
    gate2_status?: string | null;
    gate3_status?: string | null;
    structure_score?: number | null;
    novelty_score?: number | null;
    [k: string]: unknown;
  };

  spec_digest?: SpecDigest;

  // Repo-compatible extras (electrodrive/researched/api.py writes schema_version, researched block, etc.)
  schema_version?: number;
  researched?: JsonObject;
  error?: string | null;

  [k: string]: unknown;
}

/**
 * Canonical normalized log record (Design Doc §5.2).
 * The backend may add provenance fields (event_source, source) as in electrodrive/researched/ws.py.
 */
export interface CanonicalEventRecord {
  ts?: string;
  t?: number;
  level?: LogLevel;
  event: string;
  fields: JsonObject;

  iter?: number | null;
  resid?: number | null;
  resid_precond?: number | null;
  resid_true?: number | null;

  // Provenance/diagnostics (FR-9.6; repo emits these via ws.py normalize_event)
  event_source?: EventNameSource;
  source?: string;

  [k: string]: unknown;
}

/**
 * Log coverage / ingestion diagnostics (FR-9.6) with explicit modeling of §1.4 filename drift.
 */
export interface LogCoverageSummary {
  files_ingested: {
    events_jsonl: boolean;
    evidence_log_jsonl: boolean;
    other?: string[];
  };

  event_name_source_counts: {
    event: number;
    msg: number;
    message: number;
    parsed_message_json: number;
    [k: string]: number;
  };

  residual_fields_detected: string[];
  total_records: number;
  parsed_records: number;
  warnings: string[];
}

/**
 * Control protocol (Repo: electrodrive/live/controls.py; Design Doc FR-6/§1.2):
 * - snapshot is a one-shot string token (or null), not a boolean.
 * - seq/ack_seq are sequencing/handshake fields.
 * - unknown keys may be preserved across merges; backend may return them at top-level.
 */
export type ControlState = {
  pause?: boolean;
  terminate?: boolean;
  write_every?: number | null;
  snapshot?: string | null;

  ts?: number;
  version?: number;
  seq?: number;
  ack_seq?: number | null;

  // Convenience bucket for unknown keys when callers want explicit access.
  extras?: JsonObject;
} & JsonObject;

export interface RunSummary {
  run_id: string;
  workflow: Workflow;
  started_at: string | null;
  status: RunStatus | null;

  display_name?: string;
  tags?: string[];

  // Repo run index extras (electrodrive/researched/api.py _run_summary)
  ended_at?: string | null;
  path?: string;
  manifest_file?: string;
  has_viz?: boolean;
  has_events?: boolean;
  has_evidence?: boolean;
  has_train_log?: boolean;
  has_metrics_jsonl?: boolean;
  has_metrics?: boolean;

  [k: string]: unknown;
}

export interface FrameInfo {
  index: number;
  path: string;

  // For convenience, UI can store a directly displayable URL (data: URL, or server URL).
  url?: string;

  created_at?: string;
  mtime?: number;
  name?: string;
  bytes_b64?: string;

  [k: string]: unknown;
}

export type FrameEvent =
  | { type: "frame_added" | "frame_updated" | "frame_latest"; frame: FrameInfo }
  | { type: "error"; message: string; frame?: FrameInfo }
  // Compatibility: some backends may pass raw "frame" event or omit the type wrapper.
  | { type: "frame"; frame: FrameInfo };

export interface RunDetail {
  manifest: RunManifest;

  coverage?: LogCoverageSummary;
  latest_frame?: FrameInfo;
  metrics?: Record<string, number>;

  // Extensible buckets for backend-derived fields
  frames?: FrameInfo[];
  process?: JsonObject;
  upgrades?: JsonObject;

  [k: string]: unknown;
}

export interface LaunchRunRequest {
  workflow: Workflow;
  spec_path?: string;
  config?: unknown;
  argv?: string[];
  preset_id?: string;
  out_dir?: string;
  [k: string]: unknown;
}

export interface LaunchRunResponse {
  run_id: string;
  [k: string]: unknown;
}

export interface ArtifactInfo {
  path: string;
  is_dir: boolean;
  size?: number;
  mtime?: number;
  url?: string;
  [k: string]: unknown;
}

export interface CompareResponse {
  // Design Doc FR-8 expects overlays + diffs; keep DTO flexible.
  run_ids?: string[];
  overlays?: unknown;
  deltas?: unknown;
  diff?: unknown;
  [k: string]: unknown;
}

export interface PresetSummary {
  id: string;
  name: string;
  workflow?: Workflow;
  created_at?: string;
  updated_at?: string;
  data?: unknown;
  [k: string]: unknown;
}

/** Small helper type for tolerant API responses (some servers wrap with {data}). */
export type ApiEnvelope<T> = { ok?: boolean; data?: T } & JsonObject;

/** Typed error details payload. */
export type ApiErrorDetails = JsonObject & { raw?: unknown };
