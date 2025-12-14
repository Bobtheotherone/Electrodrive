import React, { useMemo, useState } from "react";

// Source notes:
// - Design Doc: FR-4 (normalization rules; iter/resid variants), FR-5 (live convergence plot).
// - Repo: electrodrive/viz/iter_viz.py (_parse_iter_event: GMRES-ish if eventName contains "gmres"; iter keys iter|iters|step|k; resid keys resid|resid_true|resid_precond|resid_true_l2|resid_precond_l2; event name fallback event||msg||message) [~L146-193, ~L133-144].
// - Repo: electrodrive/utils/logging.py (JsonlLogger uses msg field; structured numeric fields can be top-level) [~L136-200].

export type AnyLogRecord = Record<string, unknown>;

export interface ResidualPoint {
  iter: number;
  resid?: number;
  resid_precond?: number;
  resid_true?: number;
  t?: number; // epoch seconds if present
}

export interface ResidualPlotProps {
  /** Supply raw/normalized log records; component will parse GMRES-ish telemetry defensively (FR-4). */
  records?: ReadonlyArray<AnyLogRecord>;

  /** Alternatively, provide pre-extracted points. */
  points?: ReadonlyArray<ResidualPoint>;

  /** Max records used for parsing (tail window). */
  maxRecords?: number;

  /** Max points rendered (downsampled if needed). */
  maxPoints?: number;

  /** Chart height in px. */
  height?: number;

  /** Optional title. */
  title?: string;

  /** Disable interactions. */
  disabled?: boolean;

  /** Allow arbitrary extra props to avoid breaking unknown call-sites. */
  [key: string]: unknown;
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
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

function messageJsonOf(rec: AnyLogRecord): Record<string, unknown> | null {
  // FR-4: Some workflows embed JSON in the message string (e.g., learn/train).
  const msg =
    (typeof rec.msg === "string" && rec.msg.trim()) ||
    (typeof rec.message === "string" && rec.message.trim()) ||
    "";
  if (!msg || !msg.trim().startsWith("{")) return null;
  const parsed = safeJsonParse(msg);
  return isPlainObject(parsed) ? (parsed as Record<string, unknown>) : null;
}

function mergedFieldsOf(rec: AnyLogRecord): Record<string, unknown> {
  // Support either top-level fields (repo JsonlLogger) or nested "fields" (canonical record),
  // plus JSON embedded in message string (legacy stdlib logging embedding).
  // Priority: nested fields -> parsed JSON-in-message -> top-level keys (top-level wins).
  const merged: Record<string, unknown> = {};
  if (isPlainObject(rec.fields)) Object.assign(merged, rec.fields);
  const msgJson = messageJsonOf(rec);
  if (msgJson) Object.assign(merged, msgJson);
  Object.assign(merged, rec);
  return merged;
}

function eventNameOf(rec: AnyLogRecord, merged?: Record<string, unknown>): string {
  // Repo & design doc: event fallback is event || msg || message.
  const m = merged ?? mergedFieldsOf(rec);
  const direct =
    (typeof m.event === "string" && m.event.trim()) ||
    (typeof m.msg === "string" && m.msg.trim()) ||
    (typeof m.message === "string" && m.message.trim()) ||
    "";
  return typeof direct === "string" ? direct : "";
}

function toNumber(v: unknown): number | undefined {
  if (typeof v === "number" && Number.isFinite(v)) return v;
  if (typeof v === "string") {
    const n = Number(v);
    if (Number.isFinite(n)) return n;
  }
  return undefined;
}

function parseEpochSecondsFromTs(ts: unknown): number | undefined {
  if (typeof ts === "number" && Number.isFinite(ts)) {
    if (ts > 1e12) return ts / 1000;
    if (ts > 1e9) return ts;
    return ts;
  }
  if (typeof ts === "string" && ts.trim()) {
    const s = ts.trim();
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

function parseResidualPoint(rec: AnyLogRecord): ResidualPoint | null {
  const merged = mergedFieldsOf(rec);

  const name = eventNameOf(rec, merged).toLowerCase();
  if (!name.includes("gmres")) return null; // matches repo iter_viz _parse_iter_event heuristic.

  // iter keys: iter|iters|step|k
  const iter = toNumber(merged.iter) ?? toNumber(merged.iters) ?? toNumber(merged.step) ?? toNumber(merged.k);
  if (typeof iter !== "number") return null;

  // resid variants: resid|resid_true|resid_precond|*_l2
  const resid_true = toNumber(merged.resid_true) ?? toNumber(merged.resid_true_l2);
  const resid_precond = toNumber(merged.resid_precond) ?? toNumber(merged.resid_precond_l2);
  const resid = toNumber(merged.resid) ?? resid_true ?? resid_precond;

  const t = toNumber(merged.t) ?? parseEpochSecondsFromTs(merged.ts);

  if (typeof resid !== "number" && typeof resid_true !== "number" && typeof resid_precond !== "number") return null;

  return { iter, resid, resid_precond, resid_true, t };
}

type SeriesName = "resid" | "resid_precond" | "resid_true";

function buildSeries(points: ReadonlyArray<ResidualPoint>): Record<SeriesName, Array<{ x: number; y: number }>> {
  const series: Record<SeriesName, Array<{ x: number; y: number }>> = { resid: [], resid_precond: [], resid_true: [] };
  for (const p of points) {
    if (typeof p.resid === "number" && Number.isFinite(p.resid)) series.resid.push({ x: p.iter, y: p.resid });
    if (typeof p.resid_precond === "number" && Number.isFinite(p.resid_precond))
      series.resid_precond.push({ x: p.iter, y: p.resid_precond });
    if (typeof p.resid_true === "number" && Number.isFinite(p.resid_true)) series.resid_true.push({ x: p.iter, y: p.resid_true });
  }
  // Ensure x-sorted.
  (Object.keys(series) as SeriesName[]).forEach((k) => series[k].sort((a, b) => a.x - b.x));
  return series;
}

function downsample<T>(arr: ReadonlyArray<T>, max: number): T[] {
  if (arr.length <= max) return Array.from(arr);
  const step = Math.ceil(arr.length / max);
  const out: T[] = [];
  for (let i = 0; i < arr.length; i += step) out.push(arr[i]);
  if (out[out.length - 1] !== arr[arr.length - 1]) out.push(arr[arr.length - 1]);
  return out;
}

function LinePath(props: {
  data: ReadonlyArray<{ x: number; y: number }>;
  xToPx: (x: number) => number;
  yToPx: (y: number) => number;
  stroke: string;
  strokeWidth?: number;
  title?: string;
}) {
  const { data, xToPx, yToPx, stroke, strokeWidth = 2, title } = props;
  if (data.length < 2) return null;
  const d = data
    .map((p, i) => `${i === 0 ? "M" : "L"} ${xToPx(p.x).toFixed(2)} ${yToPx(p.y).toFixed(2)}`)
    .join(" ");
  return (
    <path d={d} fill="none" stroke={stroke} strokeWidth={strokeWidth} opacity={0.9}>
      {title ? <title>{title}</title> : null}
    </path>
  );
}

export function ResidualPlot(props: ResidualPlotProps) {
  const {
    records,
    points,
    maxRecords = 5000,
    maxPoints = 1200,
    height = 240,
    title = "Convergence (GMRES residual vs iteration)",
    disabled,
  } = props;

  const [logScale, setLogScale] = useState<boolean>(true);

  const extractedPoints: ResidualPoint[] = useMemo(() => {
    if (points && points.length > 0) {
      // Trust caller but still sanitize.
      const clean = points
        .map((p) => ({
          iter: typeof p.iter === "number" ? p.iter : Number(p.iter),
          resid: typeof p.resid === "number" ? p.resid : p.resid != null ? Number(p.resid) : undefined,
          resid_precond: typeof p.resid_precond === "number" ? p.resid_precond : p.resid_precond != null ? Number(p.resid_precond) : undefined,
          resid_true: typeof p.resid_true === "number" ? p.resid_true : p.resid_true != null ? Number(p.resid_true) : undefined,
          t: typeof p.t === "number" ? p.t : p.t != null ? Number(p.t) : undefined,
        }))
        .filter((p) => Number.isFinite(p.iter));
      // Prefer monotonic sort.
      clean.sort((a, b) => a.iter - b.iter);
      return clean;
    }

    const src = records ? Array.from(records) : [];
    const tail = src.length > maxRecords ? src.slice(src.length - maxRecords) : src;
    const out: ResidualPoint[] = [];
    for (const r of tail) {
      const p = parseResidualPoint(r);
      if (p) out.push(p);
    }

    // De-dup by iter keeping the last seen per iter.
    const byIter = new Map<number, ResidualPoint>();
    for (const p of out) byIter.set(p.iter, p);
    const deduped = Array.from(byIter.values()).sort((a, b) => a.iter - b.iter);
    return deduped;
  }, [points, records, maxRecords]);

  const series = useMemo(() => buildSeries(extractedPoints), [extractedPoints]);

  const lastPoint = useMemo(() => {
    if (extractedPoints.length === 0) return null;
    return extractedPoints[extractedPoints.length - 1];
  }, [extractedPoints]);

  const chart = useMemo(() => {
    const width = 720; // responsive via viewBox
    const h = Math.max(160, height);
    const margin = { l: 54, r: 14, t: 16, b: 34 };
    const innerW = width - margin.l - margin.r;
    const innerH = h - margin.t - margin.b;

    const allX = extractedPoints.map((p) => p.iter);
    const xMin = allX.length ? Math.min(...allX) : 0;
    const xMax = allX.length ? Math.max(...allX) : 1;

    const allY: number[] = [];
    for (const s of Object.values(series)) for (const p of s) if (Number.isFinite(p.y)) allY.push(p.y);
    const positiveY = allY.filter((y) => y > 0);

    const yDomain = (() => {
      if (positiveY.length === 0) return { yMin: 1e-12, yMax: 1 };
      let yMin = Math.min(...positiveY);
      let yMax = Math.max(...positiveY);
      if (yMin === yMax) {
        yMin = yMin / 10;
        yMax = yMax * 10;
      }
      // pad
      return { yMin: yMin * 0.9, yMax: yMax * 1.1 };
    })();

    const xToPx = (x: number) => margin.l + ((x - xMin) / Math.max(1e-9, xMax - xMin)) * innerW;

    const yToDomain = (y: number) => (logScale ? Math.log10(Math.max(y, 1e-30)) : y);
    const yMinD = yToDomain(yDomain.yMin);
    const yMaxD = yToDomain(yDomain.yMax);

    const yToPx = (y: number) => {
      const yd = yToDomain(y);
      const t = (yd - yMinD) / Math.max(1e-9, yMaxD - yMinD);
      return margin.t + (1 - t) * innerH;
    };

    const xTicks = 4;
    const yTicks = 4;

    const xTickVals = Array.from({ length: xTicks + 1 }, (_, i) => xMin + (i / xTicks) * (xMax - xMin));
    const yTickVals = (() => {
      if (!logScale) {
        return Array.from({ length: yTicks + 1 }, (_, i) => yDomain.yMin + (i / yTicks) * (yDomain.yMax - yDomain.yMin));
      }
      // log ticks: integer powers within bounds
      const lo = Math.floor(Math.log10(Math.max(yDomain.yMin, 1e-30)));
      const hi = Math.ceil(Math.log10(Math.max(yDomain.yMax, 1e-30)));
      const ticks: number[] = [];
      for (let p = lo; p <= hi; p++) ticks.push(Math.pow(10, p));
      // keep reasonable count
      if (ticks.length > 7) {
        const step = Math.ceil(ticks.length / 7);
        return ticks.filter((_, i) => i % step === 0);
      }
      return ticks;
    })();

    const sDown: Record<SeriesName, Array<{ x: number; y: number }>> = {
      resid: downsample(series.resid, maxPoints),
      resid_precond: downsample(series.resid_precond, maxPoints),
      resid_true: downsample(series.resid_true, maxPoints),
    };

    return {
      width,
      height: h,
      margin,
      xToPx,
      yToPx,
      xTickVals,
      yTickVals,
      series: sDown,
      xMin,
      xMax,
      yMin: yDomain.yMin,
      yMax: yDomain.yMax,
    };
  }, [extractedPoints, series, logScale, height, maxPoints]);

  const hasAny =
    chart.series.resid.length > 0 || chart.series.resid_precond.length > 0 || chart.series.resid_true.length > 0;

  return (
    <section style={{ display: "flex", flexDirection: "column", gap: 10 }}>
      <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", alignItems: "baseline" }}>
        <h3 style={{ margin: 0 }}>{title}</h3>
        <label style={{ display: "inline-flex", alignItems: "center", gap: 8, fontSize: 12 }}>
          <input type="checkbox" checked={logScale} onChange={(e) => setLogScale(e.target.checked)} disabled={!!disabled} />
          Log scale (log10)
        </label>
      </div>

      {!hasAny ? (
        <div style={{ padding: 12, border: "1px solid rgba(0,0,0,0.12)", borderRadius: 8, background: "rgba(0,0,0,0.02)" }}>
          <div style={{ fontSize: 13, opacity: 0.85 }}>No residual telemetry yet…</div>
          <div style={{ fontSize: 12, opacity: 0.75, marginTop: 4 }}>
            Waiting for GMRES-ish log events (eventName contains <code>gmres</code>) and residual fields (resid/resid_true/resid_precond variants) per FR-4/FR-5.
          </div>
        </div>
      ) : (
        <div style={{ border: "1px solid rgba(0,0,0,0.12)", borderRadius: 8, background: "#fff", padding: 8 }}>
          <svg
            viewBox={`0 0 ${chart.width} ${chart.height}`}
            width="100%"
            height={chart.height}
            role="img"
            aria-label="Residual vs iteration plot"
          >
            {/* Grid + axes */}
            <rect x={0} y={0} width={chart.width} height={chart.height} fill="white" />
            {chart.xTickVals.map((x, i) => (
              <g key={`x-${i}`}>
                <line
                  x1={chart.xToPx(x)}
                  x2={chart.xToPx(x)}
                  y1={chart.margin.t}
                  y2={chart.height - chart.margin.b}
                  stroke="rgba(0,0,0,0.06)"
                />
                <text
                  x={chart.xToPx(x)}
                  y={chart.height - chart.margin.b + 18}
                  fontSize={11}
                  textAnchor="middle"
                  fill="rgba(0,0,0,0.7)"
                >
                  {Math.round(x)}
                </text>
              </g>
            ))}
            {chart.yTickVals.map((y, i) => (
              <g key={`y-${i}`}>
                <line
                  x1={chart.margin.l}
                  x2={chart.width - chart.margin.r}
                  y1={chart.yToPx(y)}
                  y2={chart.yToPx(y)}
                  stroke="rgba(0,0,0,0.06)"
                />
                <text
                  x={chart.margin.l - 8}
                  y={chart.yToPx(y) + 4}
                  fontSize={11}
                  textAnchor="end"
                  fill="rgba(0,0,0,0.7)"
                >
                  {logScale ? y.toExponential(0) : y.toPrecision(3)}
                </text>
              </g>
            ))}

            {/* Paths */}
            <LinePath data={chart.series.resid} xToPx={chart.xToPx} yToPx={chart.yToPx} stroke="#2563eb" title="resid" />
            <LinePath
              data={chart.series.resid_precond}
              xToPx={chart.xToPx}
              yToPx={chart.yToPx}
              stroke="#f59e0b"
              title="resid_precond"
            />
            <LinePath
              data={chart.series.resid_true}
              xToPx={chart.xToPx}
              yToPx={chart.yToPx}
              stroke="#16a34a"
              title="resid_true"
            />

            {/* Axis labels */}
            <text x={chart.width / 2} y={chart.height - 6} fontSize={12} textAnchor="middle" fill="rgba(0,0,0,0.75)">
              iteration
            </text>
            <text
              x={16}
              y={chart.height / 2}
              fontSize={12}
              textAnchor="middle"
              fill="rgba(0,0,0,0.75)"
              transform={`rotate(-90 16 ${chart.height / 2})`}
            >
              residual
            </text>
          </svg>

          <div style={{ display: "flex", justifyContent: "space-between", gap: 12, flexWrap: "wrap", marginTop: 8 }}>
            <div style={{ fontSize: 12, opacity: 0.8 }}>
              <strong>Legend:</strong>{" "}
              <span style={{ color: "#2563eb" }}>resid</span>,{" "}
              <span style={{ color: "#f59e0b" }}>resid_precond</span>,{" "}
              <span style={{ color: "#16a34a" }}>resid_true</span>
            </div>
            <div style={{ fontSize: 12, opacity: 0.9 }}>
              {lastPoint ? (
                <span>
                  Last: iter <strong>{lastPoint.iter}</strong>{" "}
                  {typeof lastPoint.resid === "number" ? (
                    <>
                      · resid <strong>{lastPoint.resid.toExponential(3)}</strong>
                    </>
                  ) : null}
                  {typeof lastPoint.resid_precond === "number" ? (
                    <>
                      {" "}
                      · precond <strong>{lastPoint.resid_precond.toExponential(3)}</strong>
                    </>
                  ) : null}
                  {typeof lastPoint.resid_true === "number" ? (
                    <>
                      {" "}
                      · true <strong>{lastPoint.resid_true.toExponential(3)}</strong>
                    </>
                  ) : null}
                </span>
              ) : (
                <span>—</span>
              )}
            </div>
          </div>
        </div>
      )}
    </section>
  );
}

export default ResidualPlot;
