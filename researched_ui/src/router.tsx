import type { ReactNode } from "react";
import { useEffect, useMemo, useRef, useState } from "react";
import {
  Link,
  Navigate,
  createHashRouter,
  useNavigate,
  useParams,
  useRouteError,
  useSearchParams,
} from "react-router-dom";
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import App from "./App";
import {
  compareRuns,
  connectRunEvents,
  connectRunFrames,
  getRun,
  launchRun,
  listArtifacts,
  listPresets,
  listRuns,
  postControl,
} from "./api/client";
import type {
  ArtifactInfo,
  CanonicalEventRecord,
  CompareResponse,
  ControlState,
  FrameInfo,
  LaunchRunRequest,
  PresetSummary,
  RunDetail,
  RunSummary,
  Workflow,
} from "./types";

/**
 * ResearchED routes.
 *
 * Design Doc requirements:
 * - Day-one app sections: Run Library, Run Launcher, Live Monitor, Post-run Dashboards, Comparison, Upgrades.
 * - §3.1: local web UI ↔ Python backend over HTTP + WebSocket.
 * - FR-5: live monitor supports realtime streams + plots.
 * - FR-6: control panel semantics (pause/terminate/write_every/snapshot-token).
 *
 * Repo alignment (Bobtheotherone/Electrodrive):
 * - REST is mounted under /api/v1 and WS under /ws (electrodrive/researched/app.py).
 * - WS event stream merges events.jsonl + evidence_log.jsonl + train_log.jsonl + metrics.jsonl (electrodrive/researched/ws.py).
 */

function stringifyError(err: unknown): string {
  if (err instanceof Error) return err.message;
  try {
    return JSON.stringify(err);
  } catch {
    return String(err);
  }
}

function SectionTitle(props: { title: string; subtitle?: string; right?: ReactNode }) {
  return (
    <div style={{ display: "flex", alignItems: "baseline", justifyContent: "space-between", gap: 12 }}>
      <div>
        <h1 style={{ margin: 0, fontSize: 18 }}>{props.title}</h1>
        {props.subtitle ? (
          <div style={{ marginTop: 4, fontSize: 12, color: "#6b7280" }}>{props.subtitle}</div>
        ) : null}
      </div>
      {props.right ?? null}
    </div>
  );
}

function Card(props: { children: ReactNode; title?: string; subtitle?: string }) {
  return (
    <section
      style={{
        border: "1px solid #e5e7eb",
        borderRadius: 12,
        padding: 12,
        background: "#fff",
      }}
    >
      {props.title ? (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontWeight: 600 }}>{props.title}</div>
          {props.subtitle ? <div style={{ fontSize: 12, color: "#6b7280" }}>{props.subtitle}</div> : null}
        </div>
      ) : null}
      {props.children}
    </section>
  );
}

function SmallButton(props: {
  children: string;
  onClick?: () => void;
  disabled?: boolean;
  title?: string;
  kind?: "primary" | "danger" | "default";
}) {
  const kind = props.kind ?? "default";
  const bg = kind === "primary" ? "#111827" : kind === "danger" ? "#b91c1c" : "#ffffff";
  const fg = kind === "primary" || kind === "danger" ? "#ffffff" : "#111827";
  const border = kind === "default" ? "1px solid #e5e7eb" : "1px solid transparent";
  return (
    <button
      type="button"
      title={props.title}
      disabled={props.disabled}
      onClick={props.onClick}
      style={{
        padding: "6px 10px",
        borderRadius: 10,
        border,
        background: bg,
        color: fg,
        cursor: props.disabled ? "not-allowed" : "pointer",
        opacity: props.disabled ? 0.6 : 1,
      }}
    >
      {props.children}
    </button>
  );
}

function RouterErrorBoundary() {
  const err = useRouteError();
  return (
    <div>
      <SectionTitle title="Route error" subtitle="The router encountered an error rendering this page." />
      <pre
        style={{
          marginTop: 12,
          padding: 12,
          background: "#f9fafb",
          border: "1px solid #e5e7eb",
          borderRadius: 12,
          overflowX: "auto",
        }}
      >
        {stringifyError(err)}
      </pre>
      <div style={{ marginTop: 12 }}>
        <Link to="/runs">Go to Run Library</Link>
      </div>
    </div>
  );
}

function NotFoundPage() {
  return (
    <div>
      <SectionTitle title="Not found" subtitle="This page does not exist." />
      <div style={{ marginTop: 12 }}>
        <Link to="/runs">Go to Run Library</Link>
      </div>
    </div>
  );
}

function RunLibraryPage() {
  const [runs, setRuns] = useState<RunSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const inflight = useRef<AbortController | null>(null);

  const refresh = async () => {
    inflight.current?.abort();
    const ac = new AbortController();
    inflight.current = ac;

    setLoading(true);
    setError(null);

    try {
      const res = await listRuns({ signal: ac.signal });
      if (!ac.signal.aborted) setRuns(res);
    } catch (e) {
      if (!ac.signal.aborted) {
        setError(stringifyError(e));
        setRuns(null);
      }
    } finally {
      if (!ac.signal.aborted) setLoading(false);
    }
  };

  useEffect(() => {
    void refresh();
    return () => inflight.current?.abort();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <SectionTitle
        title="Run Library"
        subtitle="Browse local runs and jump to dashboards/monitor. (Design Doc day-one #1)"
        right={
          <SmallButton onClick={() => void refresh()} disabled={loading}>
            {loading ? "Loading…" : "Refresh"}
          </SmallButton>
        }
      />

      {error ? (
        <Card title="Error">
          <div style={{ color: "#b91c1c", fontSize: 13 }}>{error}</div>
        </Card>
      ) : null}

      <Card
        title="Runs"
        subtitle="Tip: click a run to open dashboards, or Monitor to view live streams (FR-5)."
      >
        {runs && runs.length === 0 ? <div style={{ color: "#6b7280" }}>No runs found.</div> : null}
        {!runs ? <div style={{ color: "#6b7280" }}>Loading…</div> : null}

        {runs && runs.length > 0 ? (
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 13 }}>
              <thead>
                <tr style={{ textAlign: "left" }}>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e5e7eb" }}>Run</th>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e5e7eb" }}>Workflow</th>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e5e7eb" }}>Status</th>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e5e7eb" }}>Started</th>
                  <th style={{ padding: "8px 6px", borderBottom: "1px solid #e5e7eb" }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => (
                  <tr key={r.run_id}>
                    <td style={{ padding: "8px 6px", borderBottom: "1px solid #f3f4f6" }}>
                      <Link to={`/runs/${encodeURIComponent(r.run_id)}`}>{r.run_id}</Link>
                    </td>
                    <td style={{ padding: "8px 6px", borderBottom: "1px solid #f3f4f6" }}>{String(r.workflow)}</td>
                    <td style={{ padding: "8px 6px", borderBottom: "1px solid #f3f4f6" }}>{String(r.status ?? "")}</td>
                    <td style={{ padding: "8px 6px", borderBottom: "1px solid #f3f4f6" }}>{String(r.started_at ?? "")}</td>
                    <td style={{ padding: "8px 6px", borderBottom: "1px solid #f3f4f6" }}>
                      <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
                        <Link to={`/runs/${encodeURIComponent(r.run_id)}`}>Dashboards</Link>
                        <Link to={`/runs/${encodeURIComponent(r.run_id)}/monitor`}>Monitor</Link>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : null}
      </Card>
    </div>
  );
}

function RunLauncherPage() {
  const nav = useNavigate();
  const [workflow, setWorkflow] = useState<Workflow>("solve");
  const [specPath, setSpecPath] = useState<string>("");
  const [argvText, setArgvText] = useState<string>("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const parseArgv = (): string[] | undefined => {
    const s = argvText.trim();
    if (!s) return undefined;
    if (s.startsWith("[")) {
      const parsed = JSON.parse(s);
      if (!Array.isArray(parsed)) throw new Error("argv JSON must be an array");
      return parsed.map((x) => String(x));
    }
    // Simple split (placeholder). In real UI, this becomes a proper command builder (Design Doc FR-2).
    return s.split(/\s+/).filter(Boolean);
  };

  const onLaunch = async () => {
    setBusy(true);
    setError(null);
    try {
      const req: LaunchRunRequest = {
        workflow,
        spec_path: specPath.trim() || undefined,
        argv: parseArgv(),
      };
      const res = await launchRun(req);
      nav(`/runs/${encodeURIComponent(res.run_id)}/monitor`);
    } catch (e) {
      setError(stringifyError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <SectionTitle
        title="Run Launcher"
        subtitle="Start experiments from the GUI (Design Doc FR-1, day-one #2)."
      />

      <Card title="Launch a run" subtitle="Minimal scaffold: workflow + spec path. Full validation/presets come from FR-2.">
        <div style={{ display: "grid", gap: 10, maxWidth: 720 }}>
          <label style={{ display: "grid", gap: 6 }}>
            <div style={{ fontSize: 12, color: "#6b7280" }}>Workflow</div>
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
            <div style={{ fontSize: 12, color: "#6b7280" }}>Spec path (optional)</div>
            <input
              value={specPath}
              onChange={(e) => setSpecPath(e.target.value)}
              placeholder="e.g. /path/to/spec.json"
              style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid #e5e7eb" }}
            />
          </label>

          <label style={{ display: "grid", gap: 6 }}>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              argv override (optional) — JSON array (preferred) or space-separated
            </div>
            <textarea
              value={argvText}
              onChange={(e) => setArgvText(e.target.value)}
              rows={3}
              placeholder='e.g. ["-m","electrodrive.cli","solve","--viz"]'
              style={{
                padding: "8px 10px",
                borderRadius: 10,
                border: "1px solid #e5e7eb",
                fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace",
                fontSize: 12,
              }}
            />
          </label>

          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <SmallButton kind="primary" disabled={busy} onClick={() => void onLaunch()}>
              {busy ? "Launching…" : "Launch"}
            </SmallButton>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              After launch, you’ll be redirected to the Live Monitor (FR-5).
            </div>
          </div>

          {error ? <div style={{ color: "#b91c1c", fontSize: 13 }}>{error}</div> : null}
        </div>
      </Card>
    </div>
  );
}

function RunDashboardPage() {
  const { runId } = useParams();
  const id = String(runId ?? "");
  const [detail, setDetail] = useState<RunDetail | null>(null);
  const [artifacts, setArtifacts] = useState<ArtifactInfo[] | null>(null);
  const [presets, setPresets] = useState<PresetSummary[] | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (!id) return;
    const ac = new AbortController();
    (async () => {
      try {
        setError(null);
        const d = await getRun(id, { signal: ac.signal });
        setDetail(d);
      } catch (e) {
        setError(stringifyError(e));
      }
    })();
    return () => ac.abort();
  }, [id]);

  const fetchArtifacts = async () => {
    if (!id) return;
    try {
      setError(null);
      const a = await listArtifacts(id);
      setArtifacts(a);
    } catch (e) {
      setError(stringifyError(e));
    }
  };

  const fetchPresets = async () => {
    try {
      setError(null);
      const p = await listPresets();
      setPresets(p);
    } catch (e) {
      setError(stringifyError(e));
    }
  };

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <SectionTitle
        title="Post-run Dashboards"
        subtitle="Per-run dashboards & reports (Design Doc day-one #4; FR-7/FR-10)."
        right={
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <Link to={`/runs/${encodeURIComponent(id)}/monitor`}>Open Monitor</Link>
            <SmallButton onClick={() => void fetchArtifacts()}>List Artifacts</SmallButton>
            <SmallButton onClick={() => void fetchPresets()}>List Presets</SmallButton>
          </div>
        }
      />

      {error ? (
        <Card title="Error">
          <div style={{ color: "#b91c1c", fontSize: 13 }}>{error}</div>
        </Card>
      ) : null}

      <Card title="Run detail" subtitle="Includes manifest (Design Doc §5.1) and optional coverage (FR-9.6).">
        {!detail ? (
          <div style={{ color: "#6b7280" }}>Loading…</div>
        ) : (
          <div style={{ display: "grid", gap: 10 }}>
            <div style={{ fontSize: 13 }}>
              <div>
                <b>Run:</b> {detail.manifest?.run_id ?? id}
              </div>
              <div>
                <b>Workflow:</b> {String(detail.manifest?.workflow ?? "")}
              </div>
              <div>
                <b>Status:</b> {String(detail.manifest?.status ?? "")}
              </div>
            </div>

            {detail.coverage ? (
              <Card title="Log coverage (FR-9.6)" subtitle="Filename drift & normalization diagnostics (Design Doc §1.4, §5.2).">
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
                  {JSON.stringify(detail.coverage, null, 2)}
                </pre>
              </Card>
            ) : (
              <div style={{ fontSize: 12, color: "#6b7280" }}>
                No coverage summary returned (backend may not populate yet; see Design Doc FR-9.6).
              </div>
            )}

            <div>
              <div style={{ fontWeight: 600, marginBottom: 6 }}>Manifest (Design Doc §5.1)</div>
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
                {JSON.stringify(detail.manifest, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </Card>

      {artifacts ? (
        <Card title="Artifacts" subtitle="Run directory contract (Design Doc FR-3).">
          {artifacts.length === 0 ? (
            <div style={{ color: "#6b7280" }}>No artifacts returned.</div>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 16 }}>
              {artifacts.slice(0, 200).map((a) => (
                <li key={a.path} style={{ fontSize: 13 }}>
                  {a.is_dir ? "📁" : "📄"} {a.path}
                </li>
              ))}
            </ul>
          )}
        </Card>
      ) : null}

      {presets ? (
        <Card title="Presets" subtitle="Presets are persisted to disk per FR-2 (backend: ~/.researched/presets/*.json).">
          {presets.length === 0 ? (
            <div style={{ color: "#6b7280" }}>No presets returned.</div>
          ) : (
            <ul style={{ margin: 0, paddingLeft: 16 }}>
              {presets.slice(0, 100).map((p) => (
                <li key={p.id} style={{ fontSize: 13 }}>
                  <b>{p.name}</b> <span style={{ color: "#6b7280" }}>({p.id})</span>
                </li>
              ))}
            </ul>
          )}
        </Card>
      ) : null}
    </div>
  );
}

type ResidPoint = { iter: number; resid: number };

function LiveMonitorPage() {
  const { runId } = useParams();
  const id = String(runId ?? "");

  const [events, setEvents] = useState<CanonicalEventRecord[]>([]);
  const [residPoints, setResidPoints] = useState<ResidPoint[]>([]);
  const [latestFrame, setLatestFrame] = useState<FrameInfo | null>(null);
  const [wsStatus, setWsStatus] = useState<string>("disconnected");
  const [err, setErr] = useState<string | null>(null);

  const latestFrameUrl = latestFrame?.url ?? null;

  const wsCloseRef = useRef<(() => void) | null>(null);
  const framesCloseRef = useRef<(() => void) | null>(null);

  const toNum = (v: unknown): number | null => {
    if (typeof v === "number" && Number.isFinite(v)) return v;
    if (typeof v === "string" && v.trim()) {
      const n = Number(v);
      return Number.isFinite(n) ? n : null;
    }
    return null;
  };

  const pickIter = (ev: CanonicalEventRecord): number | null => {
    return (
      toNum(ev.iter) ??
      toNum((ev.fields as any)?.iter) ??
      toNum((ev.fields as any)?.iters) ??
      toNum((ev.fields as any)?.k) ??
      toNum((ev.fields as any)?.step)
    );
  };

  const pickResid = (ev: CanonicalEventRecord): number | null => {
    return (
      toNum(ev.resid) ??
      toNum(ev.resid_true) ??
      toNum(ev.resid_precond) ??
      toNum((ev.fields as any)?.resid) ??
      toNum((ev.fields as any)?.resid_true) ??
      toNum((ev.fields as any)?.resid_precond)
    );
  };

  const appendEvent = (ev: CanonicalEventRecord) => {
    setEvents((prev) => {
      const next = [ev, ...prev];
      return next.slice(0, 400);
    });

    const iter = pickIter(ev);
    const resid = pickResid(ev);

    if (iter !== null && resid !== null) {
      setResidPoints((prev) => {
        const next = [...prev, { iter, resid }];
        // keep last ~2000 points
        return next.slice(Math.max(0, next.length - 2000));
      });
    }
  };

  useEffect(() => {
    if (!id) return;

    setErr(null);
    setWsStatus("connecting");

    // Close any previous connections (route param change)
    try {
      wsCloseRef.current?.();
      framesCloseRef.current?.();
    } catch {
      // ignore
    }

    const conn = connectRunEvents(id, {
      onOpen: () => setWsStatus("events:open"),
      onClose: () => setWsStatus("events:closed"),
      onError: () => setWsStatus("events:error"),
      onEvent: (ev) => appendEvent(ev),
    });
    wsCloseRef.current = () => conn.close();

    const frames = connectRunFrames(id, {
      onOpen: () => setWsStatus((s) => (s.includes("events:open") ? "events+frames:open" : "frames:open")),
      onClose: () => setWsStatus((s) => (s.includes("events:open") ? "events:open, frames:closed" : "frames:closed")),
      onError: () => setWsStatus((s) => (s.includes("events:open") ? "events:open, frames:error" : "frames:error")),
      onFrame: (fe) => {
        if (fe.type === "error") {
          setErr(fe.message || "Frame stream error");
          return;
        }
        if (fe.frame) setLatestFrame(fe.frame);
      },
    });
    framesCloseRef.current = () => frames.close();

    return () => {
      try {
        conn.close();
      } catch {
        // ignore
      }
      try {
        frames.close();
      } catch {
        // ignore
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [id]);

  const sendControl = async (patch: Partial<ControlState>) => {
    if (!id) return;
    setErr(null);
    try {
      await postControl(id, patch);
    } catch (e) {
      setErr(stringifyError(e));
    }
  };

  const requestSnapshotToken = () => {
    // FR-6 + repo electrodrive/live/controls.py: snapshot is a one-shot string token (NOT boolean).
    const token =
      (typeof crypto !== "undefined" && "randomUUID" in crypto && typeof crypto.randomUUID === "function"
        ? crypto.randomUUID()
        : `${Date.now()}-${Math.random().toString(16).slice(2)}`) + `@${new Date().toISOString()}`;
    void sendControl({ snapshot: token });
  };

  const latestResid = useMemo(() => {
    for (const p of [...residPoints].reverse()) return p.resid;
    return null;
  }, [residPoints]);

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <SectionTitle
        title="Live Monitor"
        subtitle="Live logs + plots + frames (Design Doc day-one #3; FR-5). Control panel (FR-6)."
        right={
          <div style={{ fontSize: 12, color: "#6b7280" }}>
            WS: <b>{wsStatus}</b>
          </div>
        }
      />

      {err ? (
        <Card title="Error">
          <div style={{ color: "#b91c1c", fontSize: 13 }}>{err}</div>
        </Card>
      ) : null}

      <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 12 }}>
        <Card title="Controls (FR-6)" subtitle="pause/resume/terminate/write_every/snapshot-token; aligns to electrodrive/live/controls.py.">
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <SmallButton onClick={() => void sendControl({ pause: true })}>Pause</SmallButton>
            <SmallButton onClick={() => void sendControl({ pause: false })}>Resume</SmallButton>
            <SmallButton kind="danger" onClick={() => void sendControl({ terminate: true })}>
              Terminate
            </SmallButton>
            <SmallButton onClick={() => void requestSnapshotToken()}>Snapshot (token)</SmallButton>
            <SmallButton onClick={() => void sendControl({ write_every: 1 })} title="Suggest more frequent outputs">
              write_every=1
            </SmallButton>
            <SmallButton onClick={() => void sendControl({ write_every: 10 })} title="Suggest less frequent outputs">
              write_every=10
            </SmallButton>
            <SmallButton onClick={() => void sendControl({ write_every: null })} title="Clear override">
              write_every=null
            </SmallButton>
          </div>
          <div style={{ marginTop: 8, fontSize: 12, color: "#6b7280" }}>
            Run: <b>{id}</b> • Latest resid: <b>{latestResid ?? "—"}</b>
          </div>
        </Card>

        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 12 }}>
          <Card title="Convergence plot (FR-5)" subtitle="GMRES residual vs iteration from normalized events (Design Doc §5.2).">
            <div style={{ height: 260 }}>
              {residPoints.length === 0 ? (
                <div style={{ color: "#6b7280", fontSize: 13 }}>No residual points yet.</div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={residPoints}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="iter" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="resid" dot={false} />
                  </LineChart>
                </ResponsiveContainer>
              )}
            </div>
          </Card>

          <Card title="Latest visualization frame (FR-5)" subtitle="Frames streamed from backend watcher (viz/*.png).">
            {latestFrameUrl ? (
              <div style={{ display: "grid", gap: 8 }}>
                <div style={{ fontSize: 12, color: "#6b7280" }}>
                  {latestFrame?.name ?? latestFrame?.path ?? "latest"}{" "}
                  {typeof latestFrame?.index === "number" ? `(index ${latestFrame.index})` : ""}
                </div>
                <img
                  alt="Latest frame"
                  src={latestFrameUrl}
                  style={{
                    maxWidth: "100%",
                    height: "auto",
                    borderRadius: 12,
                    border: "1px solid #e5e7eb",
                    background: "#f9fafb",
                  }}
                />
              </div>
            ) : (
              <div style={{ color: "#6b7280", fontSize: 13 }}>No frame received yet.</div>
            )}
          </Card>

          <Card title="Event stream (FR-5)" subtitle="Normalized event records; merges events.jsonl + evidence_log.jsonl when present (§1.4).">
            {events.length === 0 ? (
              <div style={{ color: "#6b7280", fontSize: 13 }}>Waiting for events…</div>
            ) : (
              <div style={{ maxHeight: 280, overflow: "auto", border: "1px solid #e5e7eb", borderRadius: 10 }}>
                <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
                  <thead>
                    <tr style={{ textAlign: "left" }}>
                      <th style={{ padding: "8px 10px", borderBottom: "1px solid #e5e7eb" }}>t</th>
                      <th style={{ padding: "8px 10px", borderBottom: "1px solid #e5e7eb" }}>level</th>
                      <th style={{ padding: "8px 10px", borderBottom: "1px solid #e5e7eb" }}>event</th>
                      <th style={{ padding: "8px 10px", borderBottom: "1px solid #e5e7eb" }}>iter</th>
                      <th style={{ padding: "8px 10px", borderBottom: "1px solid #e5e7eb" }}>resid</th>
                    </tr>
                  </thead>
                  <tbody>
                    {events.slice(0, 200).map((ev, i) => (
                      <tr key={`${ev.t ?? ev.ts ?? ""}-${i}`}>
                        <td style={{ padding: "6px 10px", borderBottom: "1px solid #f3f4f6", color: "#6b7280" }}>
                          {ev.ts ?? (typeof ev.t === "number" ? ev.t.toFixed(2) : "")}
                        </td>
                        <td style={{ padding: "6px 10px", borderBottom: "1px solid #f3f4f6" }}>{String(ev.level ?? "")}</td>
                        <td style={{ padding: "6px 10px", borderBottom: "1px solid #f3f4f6" }}>{String(ev.event)}</td>
                        <td style={{ padding: "6px 10px", borderBottom: "1px solid #f3f4f6" }}>
                          {typeof ev.iter === "number" ? ev.iter : ""}
                        </td>
                        <td style={{ padding: "6px 10px", borderBottom: "1px solid #f3f4f6" }}>
                          {typeof ev.resid === "number" ? ev.resid : ""}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </Card>
        </div>
      </div>

      <div style={{ fontSize: 12, color: "#6b7280" }}>
        Back to <Link to={`/runs/${encodeURIComponent(id)}`}>Dashboards</Link> or{" "}
        <Link to="/runs">Run Library</Link>.
      </div>
    </div>
  );
}

function ComparePage() {
  const [sp, setSp] = useSearchParams();
  const [input, setInput] = useState<string>(() => (sp.getAll("r") ?? []).join(", "));
  const [data, setData] = useState<CompareResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const runIds = useMemo(() => sp.getAll("r").map((x) => x.trim()).filter(Boolean), [sp]);

  useEffect(() => {
    if (runIds.length < 2) {
      setData(null);
      return;
    }
    const ac = new AbortController();
    (async () => {
      try {
        setError(null);
        const res = await compareRuns(runIds, { signal: ac.signal });
        setData(res);
      } catch (e) {
        setError(stringifyError(e));
        setData(null);
      }
    })();
    return () => ac.abort();
  }, [runIds]);

  const apply = () => {
    const ids = input
      .split(/[,\s]+/)
      .map((x) => x.trim())
      .filter(Boolean);
    const next = new URLSearchParams();
    ids.forEach((id) => next.append("r", id));
    setSp(next);
  };

  return (
    <div style={{ display: "grid", gap: 12 }}>
      <SectionTitle
        title="Compare Runs"
        subtitle="Cross-run comparison & trend view (Design Doc day-one #5; FR-8)."
      />

      <Card title="Select runs" subtitle="Enter at least two run IDs (comma/space separated).">
        <div style={{ display: "grid", gap: 8, maxWidth: 720 }}>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="runA, runB, runC"
            style={{ padding: "8px 10px", borderRadius: 10, border: "1px solid #e5e7eb" }}
          />
          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <SmallButton kind="primary" onClick={apply}>
              Compare
            </SmallButton>
            <div style={{ fontSize: 12, color: "#6b7280" }}>
              Query params: {runIds.length ? runIds.join(", ") : "—"}
            </div>
          </div>
        </div>
      </Card>

      {error ? (
        <Card title="Error">
          <div style={{ color: "#b91c1c", fontSize: 13 }}>{error}</div>
        </Card>
      ) : null}

      <Card title="Response" subtitle="Backend comparison DTO (will evolve with FR-8 overlays/diffs).">
        {!data ? (
          <div style={{ color: "#6b7280", fontSize: 13 }}>
            {runIds.length < 2 ? "Provide at least two run IDs." : "Loading…"}
          </div>
        ) : (
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
            {JSON.stringify(data, null, 2)}
          </pre>
        )}
      </Card>
    </div>
  );
}

function UpgradesPage() {
  return (
    <div style={{ display: "grid", gap: 12 }}>
      <SectionTitle
        title="Experimental Upgrades"
        subtitle="Dedicated upgrade-focused visuals mapped to ROI targets (Design Doc day-one #6; FR-9)."
      />

      <Card title="Scaffold" subtitle="This page will host FR-9 dashboards (basis expressivity, conditioning, collocation, reference stability, gates).">
        <div style={{ fontSize: 13, lineHeight: 1.5 }}>
          <ul style={{ marginTop: 0 }}>
            <li>FR-9.1 Basis expressivity: scatter + family mass bars.</li>
            <li>FR-9.2 Conditioning: histograms + trends (flags limited telemetry).</li>
            <li>FR-9.3 Collocation & oracle audit.</li>
            <li>FR-9.4 Analytic reference stability sweeps.</li>
            <li>FR-9.5 Gates/structure trends.</li>
            <li>FR-9.6 Log consumer audit panel (coverage & warnings).</li>
          </ul>
          <div style={{ color: "#6b7280", fontSize: 12 }}>
            Data contracts for these dashboards are modeled in <code>types.ts</code> (Design Doc §5.1–§5.2).
          </div>
        </div>
      </Card>
    </div>
  );
}

export const router = createHashRouter([
  {
    path: "/",
    element: <App />,
    errorElement: <RouterErrorBoundary />,
    children: [
      { index: true, element: <Navigate to="/runs" replace /> },

      // Day-one sections
      { path: "runs", element: <RunLibraryPage /> },
      { path: "launch", element: <RunLauncherPage /> },
      { path: "runs/:runId/monitor", element: <LiveMonitorPage /> },
      { path: "runs/:runId", element: <RunDashboardPage /> },
      { path: "compare", element: <ComparePage /> },
      { path: "upgrades", element: <UpgradesPage /> },

      // Not found
      { path: "*", element: <NotFoundPage /> },
    ],
  },
]);
