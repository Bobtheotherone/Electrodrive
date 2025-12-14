from __future__ import annotations

"""
ResearchED compare_service: load run bundles, extract convergence overlays, and compare runs.

Design doc anchors (normative):
- FR-7 (Program/spec dashboards): provide SpecInspector summaries and spec_digest suitable for comparisons.
- FR-8 (Cross-run comparison and trend analysis): overlay convergence curves, compare scalar metrics, and “What changed?”
  (argv diff + config/spec diff).
- §5.1 (Unified run manifest schema v1): manifests contain spec_digest (dict).

Repo contract integration (normative):
- Manifest preference order: prefer manifest.researched.json over manifest.json
  (see electrodrive/researched/api.py:L86-L100).
- spec_digest must remain a dict:
  - placeholder set in contracts/manifest_schema.new_manifest (electrodrive/researched/contracts/manifest_schema.py:L295-L310)
  - placeholder set in RunManager manifest writes (electrodrive/researched/run_manager.py:L649-L651)
- CanonicalSpec parsing is handled by electrodrive/orchestration/parser.py (JSON + optional YAML), and this module
  keeps dependencies optional/stdlib-only.

Robustness:
- Never raises on missing/malformed files; returns warnings and empty structures.
- Keeps results bounded by line/point limits and downsampling.

Implementation note:
- We implement local FR-4 log normalization (msg/event/embedded JSON, iter/resid variants, ts parsing) to avoid
  importing any web-stack modules from electrodrive.researched.ws in non-GUI contexts.
"""

import dataclasses
import json
import math
import os
import time
from collections import deque
from datetime import datetime, timezone
from hashlib import sha1
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from electrodrive.researched.contracts.manifest_schema import MANIFEST_JSON_NAME, RESEARCHED_MANIFEST_NAME, json_sanitize
from electrodrive.researched.diff import DiffResult, diff_files, diff_manifests, diff_spec_digests
from electrodrive.researched.spec_inspector import maybe_digest_from_manifest, resolve_spec_path


def _safe_read_json(path: Path) -> Any:
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_jsonl_iter(path: Path, *, limit: int = 200_000) -> Iterable[Dict[str, Any]]:
    """
    Robust JSONL reader: ignores malformed lines; never raises.
    Mirrors ResearchED API’s robust JSONL reader behavior (see electrodrive/researched/api.py:L257-L286).
    """
    if not path.is_file():
        return
    n = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if n >= limit:
                    return
                s = (line or "").strip()
                if not s:
                    continue
                try:
                    obj = json.loads(s)
                except Exception:
                    n += 1
                    continue
                if isinstance(obj, dict):
                    yield obj
                n += 1
    except Exception:
        return


def _load_manifest_any(run_dir: Path) -> Tuple[Dict[str, Any], str, List[str]]:
    """
    Load manifest preferring ResearchED-owned manifest.researched.json (FR-8 policy).

    See electrodrive/researched/api.py:L86-L100 for the same preference order.
    """
    warnings: List[str] = []
    for name in (RESEARCHED_MANIFEST_NAME, MANIFEST_JSON_NAME):
        obj = _safe_read_json(run_dir / name)
        if isinstance(obj, dict):
            return obj, name, warnings
    warnings.append("manifest not found (manifest.researched.json or manifest.json)")
    return {}, "", warnings


def _metrics_from_payload(metrics_payload: Any) -> Dict[str, Any]:
    """
    Normalize metrics.json shapes into a flat dict of scalars.

    Solver writes metrics.json as {"metrics": {...}, "meta": {...}} (see electrodrive/cli.py.bak excerpt around metrics write).
    """
    if not isinstance(metrics_payload, dict):
        return {}
    if isinstance(metrics_payload.get("metrics"), dict):
        m = dict(metrics_payload.get("metrics") or {})
        meta = metrics_payload.get("meta")
        if isinstance(meta, dict):
            # Include a few useful scalar meta keys if present.
            for k in ("solve_time_sec", "mode", "run_status", "solver_mode_effective"):
                v = meta.get(k)
                if v is None:
                    continue
                if isinstance(v, (str, int, float, bool)):
                    m.setdefault(k, v)
        return m
    # Otherwise accept top-level as metrics.
    return dict(metrics_payload)


def _coerce_float(v: Any) -> Optional[float]:
    try:
        if v is None or isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            x = float(v)
        elif isinstance(v, str):
            s = v.strip()
            if not s or s in {"NaN", "Infinity", "-Infinity"}:
                return None
            x = float(s)
        else:
            return None
        if not math.isfinite(x):
            return None
        return x
    except Exception:
        return None


def _parse_ts_to_epoch(ts: Any, *, ingest_time: float) -> float:
    """
    FR-4 timestamp normalization:
    - accept ISO-8601 strings or numeric timestamps (seconds or ms)
    - fallback to ingest_time
    """
    if ts is None or isinstance(ts, bool):
        return float(ingest_time)

    if isinstance(ts, (int, float)):
        x = float(ts)
        # Heuristic: treat very large values as ms.
        if x > 1e12:
            x = x / 1000.0
        return x if math.isfinite(x) else float(ingest_time)

    if isinstance(ts, str):
        s = ts.strip()
        if not s:
            return float(ingest_time)
        # Handle trailing Z
        if s.endswith("Z"):
            s = s[:-1] + "+00:00"
        try:
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return float(dt.timestamp())
        except Exception:
            return float(ingest_time)

    return float(ingest_time)


def _try_parse_embedded_json_from_message(msg: Any) -> Optional[Dict[str, Any]]:
    """
    FR-4: learn/train sometimes emits a JSON dict into the message string (logger.info("%s", json.dumps(...))).
    """
    if not isinstance(msg, str):
        return None
    s = msg.strip()
    if not (s.startswith("{") and s.endswith("}")):
        return None
    try:
        obj = json.loads(s)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _first_present(d: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for k in keys:
        if k in d and d.get(k) is not None:
            return d.get(k)
    return None


def _normalize_record(rec: Mapping[str, Any], *, ingest_time: float, source: str) -> Dict[str, Any]:
    """
    Local FR-4 normalization into the canonical shape used by compare_service.

    Required normalization:
    - event: rec["event"] or rec["msg"] or rec["message"]; parse embedded JSON strings containing "event"
    - iter: iter / iters / step / k
    - residual: resid / resid_precond / resid_true (+ *_l2 variants)
    - ts: parse to epoch seconds
    """
    fields: Dict[str, Any] = dict(rec)

    # event name + embedded JSON in message
    msg = fields.get("event") or fields.get("msg") or fields.get("message")
    embedded = _try_parse_embedded_json_from_message(msg)
    if embedded:
        fields.update(embedded)
        msg = fields.get("event") or fields.get("msg") or fields.get("message") or msg

    event_name = str(msg or "")
    level = str(fields.get("level") or "")

    it = _first_present(fields, ("iter", "iters", "step", "k"))

    rp = _first_present(fields, ("resid_precond", "resid_precond_l2"))
    rt = _first_present(fields, ("resid_true", "resid_true_l2"))
    r = _first_present(fields, ("resid",)) or rp or rt

    t = _parse_ts_to_epoch(fields.get("ts"), ingest_time=ingest_time)

    return {
        "t": t,
        "ts": fields.get("ts"),
        "level": level,
        "event": event_name,
        "iter": it,
        "resid": r,
        "resid_precond": rp,
        "resid_true": rt,
        "fields": fields,
        "source": source,
    }


def _fingerprint(ev: Mapping[str, Any]) -> str:
    """
    Good-enough dedup hash for merged logs.
    """
    try:
        core = {
            "t": round(float(ev.get("t", 0.0)), 3),
            "level": str(ev.get("level", "")),
            "event": str(ev.get("event", "")),
            "iter": ev.get("iter"),
            "resid": ev.get("resid"),
            "resid_precond": ev.get("resid_precond"),
            "resid_true": ev.get("resid_true"),
            "fields": ev.get("fields", {}),
        }
        blob = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    except Exception:
        blob = str(ev)
    return sha1(blob.encode("utf-8", errors="ignore")).hexdigest()


def _merge_events_evidence(run_dir: Path, *, limit: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    """
    Read and merge events.jsonl + evidence_log.jsonl if present (FR-4 + §1.4), deduplicating conservatively.

    Important: uses a deque for dedup eviction to avoid O(n^2) behavior with large logs.
    """
    warnings: List[str] = []
    paths: List[Path] = []
    for name in ("events.jsonl", "evidence_log.jsonl", "researched_events.jsonl"):
        p = run_dir / name
        if p.is_file():
            paths.append(p)
    if not paths:
        return [], ["no events.jsonl or evidence_log.jsonl present"]

    out: List[Dict[str, Any]] = []
    seen_q: deque[str] = deque()
    seen_set: set[str] = set()
    max_seen = 50_000

    # Enforce a global record cap across both files.
    total = 0

    for p in paths:
        for rec in _safe_jsonl_iter(p, limit=limit):
            if total >= limit:
                break
            total += 1
            try:
                ev = _normalize_record(rec, ingest_time=time.time(), source=p.name)
                fp = _fingerprint(ev)
                if fp in seen_set:
                    continue
                seen_q.append(fp)
                seen_set.add(fp)
                if len(seen_q) > max_seen:
                    old = seen_q.popleft()
                    seen_set.discard(old)
                out.append(ev)
            except Exception:
                continue

    try:
        out.sort(key=lambda e: float(e.get("t", 0.0)))
    except Exception:
        pass
    return out, warnings


def extract_convergence_series(run_dir: Union[str, Path], *, limit: int = 200_000) -> dict[str, Any]:
    """
    Extract convergence series for a run (FR-8 overlays; FR-7 solve dashboard).

    Output per run:
      { "iters":[...], "resid":[...], "resid_precond":[...], "resid_true":[...], "t":[...] }

    Normalization rules follow FR-4:
      - event = rec.get("event") or rec.get("msg") or rec.get("message") (plus embedded JSON parsing)
      - iter = iter/iters/step/k
      - resid variants: resid/resid_precond/resid_true/*_l2
    """
    rd = Path(run_dir)
    merged, warnings = _merge_events_evidence(rd, limit=limit)

    iters: List[int] = []
    resid: List[Optional[float]] = []
    resid_pre: List[Optional[float]] = []
    resid_true: List[Optional[float]] = []
    ts: List[float] = []

    for i, ev in enumerate(merged):
        # Pull iter and residuals without assuming specific event names.
        it = ev.get("iter")
        try:
            it_i = int(it) if it is not None and not isinstance(it, bool) else None
        except Exception:
            it_i = None

        r = _coerce_float(ev.get("resid"))
        rp = _coerce_float(ev.get("resid_precond"))
        rt = _coerce_float(ev.get("resid_true"))
        if r is None and rp is None and rt is None:
            continue

        # If iter missing, fall back to sequence index for within-run plotting (still useful).
        if it_i is None:
            it_i = i

        iters.append(int(it_i))
        resid.append(r)
        resid_pre.append(rp)
        resid_true.append(rt)
        try:
            ts.append(float(ev.get("t", 0.0)))
        except Exception:
            ts.append(0.0)

    # Downsample to keep bounded.
    max_points = 20_000
    n = len(iters)
    if n > max_points and n > 0:
        step = int(math.ceil(n / max_points))
        iters = iters[::step]
        resid = resid[::step]
        resid_pre = resid_pre[::step]
        resid_true = resid_true[::step]
        ts = ts[::step]

    return {
        "iters": iters,
        "resid": resid,
        "resid_precond": resid_pre,
        "resid_true": resid_true,
        "t": ts,
        "warnings": warnings,
    }


def load_run_bundle(run_dir: Union[str, Path]) -> dict[str, Any]:
    """
    Load a run bundle for comparison/inspection (FR-7/FR-8).

    Returns:
      {
        "run_dir": str,
        "manifest": dict,
        "metrics": dict,
        "spec_digest": dict,
        "warnings": [...]
      }

    If manifest["spec_digest"] is missing/empty, computes it via SpecInspector best-effort
    and returns it (does not overwrite files).
    """
    rd = Path(run_dir)
    warnings: List[str] = []

    manifest, manifest_name, w0 = _load_manifest_any(rd)
    warnings.extend(w0)

    metrics_payload = _safe_read_json(rd / "metrics.json")
    metrics = _metrics_from_payload(metrics_payload)

    # Spec digest
    spec_digest: Dict[str, Any] = {}
    sd = manifest.get("spec_digest") if isinstance(manifest, dict) else None
    if isinstance(sd, dict) and sd:
        spec_digest = dict(sd)
    else:
        digest, w = maybe_digest_from_manifest(manifest, run_dir=rd)
        spec_digest = dict(digest or {})
        warnings.extend(w)

    # Ensure type-stable dict (Design Doc §5.1).
    if not isinstance(spec_digest, dict):
        spec_digest = {}

    return {
        "run_dir": str(rd),
        "manifest": manifest,
        "manifest_file": manifest_name,
        "metrics": metrics,
        "spec_digest": spec_digest,
        "warnings": warnings,
    }


def _run_id(bundle: Mapping[str, Any]) -> str:
    m = bundle.get("manifest") if isinstance(bundle.get("manifest"), dict) else {}
    rid = None
    if isinstance(m, dict):
        rid = m.get("run_id") or m.get("id") or m.get("uuid")
    if isinstance(rid, str) and rid.strip():
        return rid.strip()
    return Path(str(bundle.get("run_dir") or "")).name


def _workflow(bundle: Mapping[str, Any]) -> str:
    m = bundle.get("manifest") if isinstance(bundle.get("manifest"), dict) else {}
    if isinstance(m, dict):
        wf = m.get("workflow")
        if isinstance(wf, str) and wf.strip():
            return wf.strip()
    return "unknown"


def _status(bundle: Mapping[str, Any]) -> Optional[str]:
    m = bundle.get("manifest") if isinstance(bundle.get("manifest"), dict) else {}
    if isinstance(m, dict):
        v = m.get("status") or m.get("run_status")
        if isinstance(v, str) and v.strip():
            return v.strip()
        # Prefer ResearchED internal status when present (API does this too).
        r = m.get("researched")
        if isinstance(r, Mapping):
            s = r.get("internal_status") or r.get("phase")
            if isinstance(s, str) and s.strip():
                return s.strip()
    return None


def _ts(bundle: Mapping[str, Any], key: str) -> Optional[str]:
    m = bundle.get("manifest") if isinstance(bundle.get("manifest"), dict) else {}
    if isinstance(m, dict):
        v = m.get(key)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


def _git_sha(bundle: Mapping[str, Any]) -> Optional[str]:
    m = bundle.get("manifest") if isinstance(bundle.get("manifest"), dict) else {}
    if isinstance(m, dict):
        g = m.get("git")
        if isinstance(g, dict):
            s = g.get("sha")
            if isinstance(s, str) and s.strip():
                return s.strip()
    return None


def _gate_fields(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    m = bundle.get("manifest") if isinstance(bundle.get("manifest"), dict) else {}
    out: Dict[str, Any] = {}
    if isinstance(m, dict):
        g = m.get("gate")
        if isinstance(g, dict):
            for k in ("gate1_status", "gate2_status", "gate3_status", "structure_score", "novelty_score"):
                if k in g:
                    out[k] = g.get(k)
    return out


def _metric_value_is_number(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(float(v))


def compare_runs(run_dirs: List[Union[str, Path]], *, baseline: int = 0) -> dict[str, Any]:
    """
    Compare 2+ runs per FR-8.

    Returns a JSON-serializable payload:
      - runs: summaries (id, workflow, status, timestamps, git sha, spec_hash)
      - metrics_table: union of scalar metric keys (plus gate fields), per run values
      - deltas: per run vs baseline numeric deltas (abs, pct)
      - overlays: convergence series (iters/resid/resid_precond/resid_true/t)
      - what_changed: baseline-vs-run diffs (argv/config/spec pointers + spec_digest semantic diff)
    """
    warnings: List[str] = []
    if not run_dirs:
        return {"ok": False, "error": "no run_dirs provided", "warnings": []}
    if baseline < 0 or baseline >= len(run_dirs):
        baseline = 0

    bundles = [load_run_bundle(rd) for rd in run_dirs]
    for b in bundles:
        warnings.extend(list(b.get("warnings") or []))

    base = bundles[baseline]
    base_id = _run_id(base)

    runs_summary: List[Dict[str, Any]] = []
    overlays: Dict[str, Any] = {}
    metrics_by_run: Dict[str, Dict[str, Any]] = {}

    # Build per-run summaries and overlays.
    for b in bundles:
        rid = _run_id(b)
        sd = b.get("spec_digest") if isinstance(b.get("spec_digest"), dict) else {}
        spec_hash = sd.get("spec_hash") if isinstance(sd, dict) else None

        runs_summary.append(
            {
                "run_id": rid,
                "workflow": _workflow(b),
                "status": _status(b),
                "started_at": _ts(b, "started_at"),
                "ended_at": _ts(b, "ended_at"),
                "git_sha": _git_sha(b),
                "spec_hash": spec_hash,
                "run_dir": b.get("run_dir"),
            }
        )

        # Overlays: only include if residual telemetry exists; function itself is robust.
        series = extract_convergence_series(b.get("run_dir") or "", limit=200_000)
        if series.get("iters"):
            overlays[rid] = series

        # Metrics dict for table.
        m = b.get("metrics") if isinstance(b.get("metrics"), dict) else {}
        m2 = dict(m)
        m2.update(_gate_fields(b))
        # Include spec_hash as a pseudo-metric (useful in tables).
        if spec_hash is not None:
            m2.setdefault("spec_hash", spec_hash)
        metrics_by_run[rid] = m2

    # Union of metric keys.
    all_keys: List[str] = sorted({k for md in metrics_by_run.values() for k in md.keys()})

    # Build metrics_table (values per run).
    table_rows: List[Dict[str, Any]] = []
    for k in all_keys:
        row = {"key": k, "values": {}}
        for b in bundles:
            rid = _run_id(b)
            row["values"][rid] = metrics_by_run.get(rid, {}).get(k)
        table_rows.append(row)

    metrics_table = {"baseline": base_id, "rows": table_rows, "run_ids": [r["run_id"] for r in runs_summary]}

    # Deltas vs baseline.
    deltas: Dict[str, Dict[str, Any]] = {}
    base_metrics = metrics_by_run.get(base_id, {})
    for b in bundles:
        rid = _run_id(b)
        if rid == base_id:
            continue
        cur = metrics_by_run.get(rid, {})
        d: Dict[str, Any] = {}
        for k in all_keys:
            a = base_metrics.get(k)
            c = cur.get(k)
            if _metric_value_is_number(a) and _metric_value_is_number(c):
                av = float(a)
                cv = float(c)
                abs_d = cv - av
                pct = None
                if av != 0.0:
                    pct = abs_d / av * 100.0
                d[k] = {"abs": abs_d, "pct": pct}
        deltas[rid] = d

    # What changed (baseline vs each).
    what_changed: Dict[str, Any] = {}
    base_manifest = base.get("manifest") if isinstance(base.get("manifest"), dict) else {}
    base_sd = base.get("spec_digest") if isinstance(base.get("spec_digest"), dict) else {}

    for b in bundles:
        rid = _run_id(b)
        if rid == base_id:
            continue
        cur_manifest = b.get("manifest") if isinstance(b.get("manifest"), dict) else {}
        cur_sd = b.get("spec_digest") if isinstance(b.get("spec_digest"), dict) else {}

        man_pkg = diff_manifests(base_manifest, cur_manifest)
        sd_diff = diff_spec_digests(base_sd, cur_sd)

        # Also attempt file diffs for config/spec when paths exist (best-effort).
        file_diffs: Dict[str, Any] = {}
        try:
            a_in = base_manifest.get("inputs") if isinstance(base_manifest.get("inputs"), dict) else {}
            b_in = cur_manifest.get("inputs") if isinstance(cur_manifest.get("inputs"), dict) else {}
            a_spec = a_in.get("spec_path") or a_in.get("problem") or a_in.get("spec")
            b_spec = b_in.get("spec_path") or b_in.get("problem") or b_in.get("spec")
            a_cfg = a_in.get("config_path") or a_in.get("config")
            b_cfg = b_in.get("config_path") or b_in.get("config")

            # Resolve relative paths robustly (run_dir hint + repo root inference).
            a_dir = Path(base.get("run_dir") or "")
            b_dir = Path(b.get("run_dir") or "")

            def _resolve(p: Any, base_dir: Path) -> Optional[Path]:
                if not isinstance(p, str) or not p.strip():
                    return None
                try:
                    rp = resolve_spec_path(p.strip(), repo_root_hint=base_dir)
                except Exception:
                    rp = Path(p.strip()).expanduser()
                return rp if rp.exists() else None

            a_spec_p = _resolve(a_spec, a_dir)
            b_spec_p = _resolve(b_spec, b_dir)
            if a_spec_p and b_spec_p:
                file_diffs["spec_file_diff"] = dataclasses.asdict(diff_files(a_spec_p, b_spec_p))

            a_cfg_p = _resolve(a_cfg, a_dir)
            b_cfg_p = _resolve(b_cfg, b_dir)
            if a_cfg_p and b_cfg_p:
                file_diffs["config_file_diff"] = dataclasses.asdict(diff_files(a_cfg_p, b_cfg_p))
        except Exception:
            file_diffs = {}

        what_changed[rid] = {
            "baseline": base_id,
            "manifest": man_pkg,
            "spec_digest_diff": dataclasses.asdict(sd_diff),
            "file_diffs": file_diffs,
        }

    payload = {
        "ok": True,
        "baseline": base_id,
        "runs": runs_summary,
        "metrics_table": json_sanitize(metrics_table),
        "deltas": json_sanitize(deltas),
        "overlays": json_sanitize(overlays),
        "what_changed": json_sanitize(what_changed),
        "warnings": warnings,
    }
    return payload

