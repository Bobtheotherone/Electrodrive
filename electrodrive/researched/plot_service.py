# electrodrive/researched/plot_service.py
from __future__ import annotations

"""
ResearchED PlotService.

Design Doc requirements satisfied here:
- FR-3 (run-dir contract): plots go under run_dir/plots/ and must be robust to missing files.
- FR-4 (log normalization/merge): ingest events.jsonl and/or evidence_log.jsonl; normalize msg/event/embedded JSON;
  iter/resid variants; dedup merged streams.
- FR-7 (post-run dashboards): generate at least convergence (solve), basis_scatter/family_mass + gates (discover),
  loss curves (learn), accuracy/runtime (FMM).
- FR-9.1 (basis expressivity plots): compute from discovered_system.json alone when possible.
- FR-9.5 (gate dashboards): produce stable plots/gate_dashboard.json + plots/gate_dashboard.png per run.
- FR-9.6 (log consumer audit panel): compute log coverage stats (records parsed, event_source breakdown,
  residual variants, ingested files, malformed lines).
- FR-10 (report generation): report service consumes these plots; PlotService must be safe and robust.

Repo grounding (why normalization/compat exists):
- JsonlLogger writes events.jsonl and records include ts/level/msg (not "event"). :contentReference[oaicite:0]{index=0} :contentReference[oaicite:1]{index=1}
- ai_solve loads events.jsonl (fallback evidence_log.jsonl) and currently extracts traces by ev.get("event") and resid/iter,
  illustrating schema mismatch and filename drift. :contentReference[oaicite:2]{index=2} :contentReference[oaicite:3]{index=3}
- iter_viz documents evidence_log.jsonl + viz/*.png and parses rec.get("event")/rec.get("resid"). :contentReference[oaicite:4]{index=4} :contentReference[oaicite:5]{index=5}
- ResearchED ws.normalize_event already implements FR-4 normalization and exposes event_source for FR-9.6. :contentReference[oaicite:6]{index=6} :contentReference[oaicite:7]{index=7}
- Run-dir contract ensures plots/ exists and provides log-compat bridge helpers. :contentReference[oaicite:8]{index=8} :contentReference[oaicite:9]{index=9}
- discovered_system.json format uses {"images":[...], weight per element}, load/save in electrodrive/images/io.py. :contentReference[oaicite:10]{index=10}
- group_info serialization and _group_info deserialization lives in electrodrive/images/basis.py. :contentReference[oaicite:11]{index=11} :contentReference[oaicite:12]{index=12}
- Gate2 implementation is in tools/images_gate2.py (compute_structural_summary). :contentReference[oaicite:13]{index=13}
- Gate3 novelty scoring/status is in electrodrive/discovery/novelty.py. :contentReference[oaicite:14]{index=14}
- Learn metrics.jsonl writer appends json.dumps lines with step/lr/elapsed_time and metric keys. :contentReference[oaicite:15]{index=15}
- FMM sanity suite emits JSONL "fmm_test_result" with max_abs_err/rel_l2_err (+ extra, incl wall_time_s). :contentReference[oaicite:16]{index=16} :contentReference[oaicite:17]{index=17}

Dependency policy:
- No new hard deps. matplotlib is optional (plots degrade gracefully without it).
"""

import hashlib
import json
import logging
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

from electrodrive.researched.contracts.run_dir import ensure_log_compat, ensure_run_dir
from electrodrive.researched.contracts.manifest_schema import MANIFEST_JSON_NAME, RESEARCHED_MANIFEST_NAME

log = logging.getLogger(__name__)


# ----------------------------
# Small robust helpers
# ----------------------------

def _safe_json_load(path: Path) -> Any:
    try:
        if not path.is_file():
            return None
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_json_dump(obj: Any) -> str:
    try:
        return json.dumps(obj, indent=2, sort_keys=True, ensure_ascii=False, default=str)
    except Exception:
        try:
            return json.dumps(str(obj), ensure_ascii=False)
        except Exception:
            return "{}"


def _atomic_write_text(path: Path, text: str) -> None:
    """
    Atomic write via unique temp + os.replace (mirrors ResearchED contract style). :contentReference[oaicite:18]{index=18}
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, path)
    except Exception:
        try:
            path.write_text(text, encoding="utf-8")
        except Exception:
            pass
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        return
    tmp = path.with_suffix(path.suffix + f".tmp-{os.getpid()}-{time.time_ns()}")
    try:
        tmp.write_bytes(data)
        os.replace(tmp, path)
    except Exception:
        try:
            path.write_bytes(data)
        except Exception:
            pass
        try:
            if tmp.exists():
                tmp.unlink()
        except Exception:
            pass


# 1x1 transparent PNG (no external deps) as a placeholder when matplotlib is unavailable.
_PLACEHOLDER_PNG = bytes.fromhex(
    "89504E470D0A1A0A0000000D49484452000000010000000108060000001F15C489"
    "0000000A49444154789C6300010000050001"
    "0D0A2DB40000000049454E44AE426082"
)


def _write_placeholder_png(path: Path, note: str | None = None) -> None:
    """
    Write a minimal PNG so reports can still reference the file if plotting backends are missing.
    """
    # We cannot embed text without PIL/matplotlib; include note only via logs.
    if note:
        log.debug("Writing placeholder PNG for %s: %s", path, note)
    _atomic_write_bytes(path, _PLACEHOLDER_PNG)


def _load_manifest_any(run_dir: Path) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Load a manifest dict from a run directory, preferring ResearchED-owned manifest.

    This mirrors the repo’s preference order in researched/api.py. :contentReference[oaicite:19]{index=19}
    """
    for name in (RESEARCHED_MANIFEST_NAME, MANIFEST_JSON_NAME):
        obj = _safe_json_load(run_dir / name)
        if isinstance(obj, dict):
            return obj, name
    return None, ""


def _infer_workflow(run_dir: Path, manifest: Mapping[str, Any] | None, explicit: str | None) -> str:
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if isinstance(manifest, Mapping):
        wf = manifest.get("workflow")
        if isinstance(wf, str) and wf.strip():
            return wf.strip()
    # heuristic
    if (run_dir / "discovered_system.json").is_file() or (run_dir / "discovery_manifest.json").is_file():
        return "images_discover"
    if (run_dir / "metrics.jsonl").is_file() or (run_dir / "train_log.jsonl").is_file():
        return "learn_train"
    if (run_dir / "viz").is_dir():
        return "solve"
    return "unknown"


@dataclass
class _LogCoverage:
    """
    Minimal FR-9.6 coverage accumulator (JSON-serializable).
    """
    total_lines_seen: int = 0
    total_records_parsed: int = 0
    total_records_emitted: int = 0
    total_json_errors: int = 0
    total_non_dict_records: int = 0
    dropped_by_dedup: int = 0

    ingested_files: set[str] = None  # type: ignore[assignment]
    per_file: dict[str, dict[str, int]] = None  # type: ignore[assignment]
    event_source_counts: dict[str, int] = None  # type: ignore[assignment]
    residual_field_detection_counts: dict[str, int] = None  # type: ignore[assignment]
    last_event_t: float = 0.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "ingested_files", set())
        object.__setattr__(self, "per_file", {})
        object.__setattr__(self, "event_source_counts", {})
        object.__setattr__(self, "residual_field_detection_counts", {})

    def note_file(self, path: Path) -> None:
        self.ingested_files.add(path.name)
        self.per_file.setdefault(
            path.name,
            {"lines_seen": 0, "records_parsed": 0, "json_errors": 0, "non_dict": 0, "emitted": 0},
        )

    def note_line(self, path: Path) -> None:
        self.total_lines_seen += 1
        self.per_file[path.name]["lines_seen"] += 1

    def note_json_error(self, path: Path) -> None:
        self.total_json_errors += 1
        self.per_file[path.name]["json_errors"] += 1

    def note_non_dict(self, path: Path) -> None:
        self.total_non_dict_records += 1
        self.per_file[path.name]["non_dict"] += 1

    def note_parsed(self, path: Path) -> None:
        self.total_records_parsed += 1
        self.per_file[path.name]["records_parsed"] += 1

    def note_emitted(self, path: Path, t: float, event_source: str, raw: Mapping[str, Any]) -> None:
        self.total_records_emitted += 1
        self.per_file[path.name]["emitted"] += 1
        self.last_event_t = max(self.last_event_t, float(t))

        es = str(event_source or "unknown")
        self.event_source_counts[es] = self.event_source_counts.get(es, 0) + 1

        # Residual variant coverage (raw keys seen). FR-9.6.
        for k in ("resid", "resid_precond", "resid_precond_l2", "resid_true", "resid_true_l2"):
            if k in raw:
                self.residual_field_detection_counts[k] = self.residual_field_detection_counts.get(k, 0) + 1

    def note_dedup_drop(self) -> None:
        self.dropped_by_dedup += 1

    def snapshot(self) -> Dict[str, Any]:
        return {
            "total_lines_seen": int(self.total_lines_seen),
            "total_records_parsed": int(self.total_records_parsed),
            "total_records_emitted": int(self.total_records_emitted),
            "total_json_errors": int(self.total_json_errors),
            "total_non_dict_records": int(self.total_non_dict_records),
            "dropped_by_dedup": int(self.dropped_by_dedup),
            "event_source_counts": dict(sorted(self.event_source_counts.items(), key=lambda kv: kv[0])),
            "residual_field_detection_counts": dict(
                sorted(self.residual_field_detection_counts.items(), key=lambda kv: kv[0])
            ),
            "ingested_files": sorted(self.ingested_files),
            "per_file": self.per_file,
            "last_event_t": float(self.last_event_t) if self.last_event_t else None,
        }


class _DedupCache:
    """
    Bounded dedup cache (size-limited) to avoid unbounded memory use (engineering constraint).
    """
    def __init__(self, max_items: int = 50_000):
        self.max_items = max(1, int(max_items))
        self._order: List[str] = []
        self._set: set[str] = set()

    def seen(self, fp: str) -> bool:
        if fp in self._set:
            return True
        self._set.add(fp)
        self._order.append(fp)
        if len(self._order) > self.max_items:
            old = self._order.pop(0)
            self._set.discard(old)
        return False


def _fingerprint_norm(ev: Mapping[str, Any]) -> str:
    """
    Stable dedup fingerprint, similar to ws._fingerprint (t/level/event/iter/resid/fields). :contentReference[oaicite:20]{index=20}
    """
    try:
        core = {
            "t": round(float(ev.get("t", 0.0)), 3),
            "level": str(ev.get("level", "")),
            "event": str(ev.get("event", "")),
            "iter": ev.get("iter"),
            "resid": ev.get("resid"),
            "fields": ev.get("fields", {}),
        }
        blob = json.dumps(core, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str)
    except Exception:
        blob = str(ev)
    return hashlib.sha1(blob.encode("utf-8", errors="ignore")).hexdigest()


def _safe_jsonl_iter(path: Path, *, limit: int = 2_000_000) -> Iterable[Dict[str, Any]]:
    """
    Robust JSONL reader (best-effort, never raises), similar to researched/api.py’s reader. :contentReference[oaicite:21]{index=21}
    """
    if not path.is_file():
        return
    n = 0
    try:
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if n >= limit:
                    return
                line = (line or "").strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    n += 1
                    continue
                if isinstance(rec, dict):
                    yield rec
                n += 1
    except Exception:
        return


def _try_import_matplotlib():
    """
    Optional plotting backend; no hard dependency (repo constraint). :contentReference[oaicite:22]{index=22}
    """
    try:
        import matplotlib  # type: ignore

        # Headless-safe backend.
        try:
            matplotlib.use("Agg")  # type: ignore[attr-defined]
        except Exception:
            pass
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def _coerce_float(v: Any) -> Optional[float]:
    try:
        if isinstance(v, bool):
            return None
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str):
            return float(v.strip())
    except Exception:
        return None
    return None


def _coerce_int(v: Any) -> Optional[int]:
    f = _coerce_float(v)
    if f is None:
        return None
    try:
        return int(f)
    except Exception:
        return None


def _parse_ts_to_epoch(ts: Any, *, ingest_time: float) -> float:
    """
    Parse repo-style timestamps into epoch seconds.
    - numeric seconds or milliseconds
    - ISO-8601 strings, including 'Z'
    Falls back to ingest_time.
    """
    try:
        if isinstance(ts, (int, float)) and not isinstance(ts, bool):
            v = float(ts)
            # Heuristic: milliseconds if very large.
            return v / 1000.0 if v > 1.0e11 else v
        if isinstance(ts, str) and ts.strip():
            s = ts.strip()
            if s.endswith("Z"):
                s = s[:-1] + "+00:00"
            dt = datetime.fromisoformat(s)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return dt.timestamp()
    except Exception:
        pass
    return float(ingest_time)


def _minimal_normalize_event(rec: Mapping[str, Any], *, ingest_time: float, source: str) -> Dict[str, Any]:
    """
    Dependency-free implementation of the FR-4 normalization rules.
    Produces the canonical event shape used by PlotService.
    """
    raw_msg = rec.get("event") or rec.get("msg") or rec.get("message")
    if "event" in rec:
        event_source = "event"
    elif "msg" in rec:
        event_source = "msg"
    elif "message" in rec:
        event_source = "message"
    else:
        event_source = "unknown"
    event_name: str = str(raw_msg) if raw_msg is not None else ""

    # Parse JSON embedded in message string (learn/train pattern)
    parsed_msg: Dict[str, Any] | None = None
    if isinstance(raw_msg, str):
        s = raw_msg.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    parsed_msg = obj
                    if obj.get("event"):
                        event_name = str(obj.get("event"))
                        event_source = "parsed_from_message_json"
            except Exception:
                parsed_msg = None

    # iter variants
    it = rec.get("iter")
    if it is None:
        it = rec.get("iters")
    if it is None:
        it = rec.get("step")
    if it is None:
        it = rec.get("k")
    iter_val = _coerce_int(it)

    # residual variants
    resid_precond = rec.get("resid_precond")
    if resid_precond is None:
        resid_precond = rec.get("resid_precond_l2")
    resid_true = rec.get("resid_true")
    if resid_true is None:
        resid_true = rec.get("resid_true_l2")

    r_pre = _coerce_float(resid_precond)
    r_true = _coerce_float(resid_true)

    r = _coerce_float(rec.get("resid"))
    if r is None:
        r = r_pre if r_pre is not None else r_true

    # timestamp
    ts_raw = rec.get("ts")
    t = _parse_ts_to_epoch(ts_raw, ingest_time=float(ingest_time))

    # fields payload = everything else (plus parsed message extras)
    known = {
        "ts", "t", "time", "level", "msg", "message", "event",
        "iter", "iters", "step", "k",
        "resid", "resid_precond", "resid_precond_l2", "resid_true", "resid_true_l2",
    }
    fields: Dict[str, Any] = {k: v for k, v in rec.items() if k not in known}
    if isinstance(parsed_msg, dict):
        for k, v in parsed_msg.items():
            if k == "event" or k in known:
                continue
            if k not in fields:
                fields[k] = v

    level = rec.get("level", "info")
    level_str = str(level).lower() if level is not None else "info"

    return {
        "ts": ts_raw,
        "t": float(t),
        "level": level_str,
        "event": event_name,
        "event_source": event_source,
        "source": str(source),
        "iter": iter_val,
        "resid": r,
        "resid_precond": r_pre,
        "resid_true": r_true,
        "fields": fields,
    }


def _get_normalize_event():
    """
    Prefer a dependency-free normalizer module if present,
    otherwise fall back to electrodrive.researched.ws.normalize_event if available,
    otherwise use the minimal built-in normalizer.

    IMPORTANT: normalize_event implementations may vary in signature. We always adapt to:
        normalize_event(rec, ingest_time=..., source=...)
    """
    def _wrap(fn):
        def _call(rec, *, ingest_time: float, source: str):
            # Try the "new" signature first (recommended for ResearchED).
            try:
                ev = fn(rec, ingest_time=ingest_time, source=source)
            except TypeError:
                # Older signatures / partial kwargs.
                try:
                    ev = fn(rec, ingest_time=ingest_time)
                except TypeError:
                    try:
                        ev = fn(rec, source=source)
                    except TypeError:
                        ev = fn(rec)
            # Ensure we always return a dict-like canonical record.
            if not isinstance(ev, dict):
                return _minimal_normalize_event(rec, ingest_time=ingest_time, source=source)
            # Ensure required keys exist (defensive).
            if "event" not in ev:
                ev["event"] = str(rec.get("event") or rec.get("msg") or rec.get("message") or "")
            if "t" not in ev:
                ev["t"] = _parse_ts_to_epoch(rec.get("ts"), ingest_time=ingest_time)
            if "fields" not in ev or not isinstance(ev.get("fields"), dict):
                ev["fields"] = {}
            if "source" not in ev:
                ev["source"] = str(source)
            if "event_source" not in ev:
                ev["event_source"] = "unknown"
            return ev
        return _call

    for mod in ("electrodrive.researched.ingest.normalizer", "electrodrive.researched.ws"):
        try:
            m = __import__(mod, fromlist=["normalize_event"])
            fn = getattr(m, "normalize_event", None)
            if callable(fn):
                return lambda rec, ingest_time, source, _fn=fn: _wrap(_fn)(rec, ingest_time=float(ingest_time), source=str(source))
        except Exception:
            continue

    return lambda rec, ingest_time, source: _minimal_normalize_event(rec, ingest_time=float(ingest_time), source=str(source))



class PlotService:
    """
    Generate per-run plots and structured dashboards.

    Public API (required):
      - generate_all(...)
      - generate_gate_dashboard(...)
    """

    def generate_all(
        self,
        run_dir: Path,
        *,
        manifest: Mapping[str, Any] | None = None,
        workflow: str | None = None,
    ) -> Dict[str, Any]:
        run_dir = Path(run_dir)
        result: Dict[str, Any] = {
            "ok": True,
            "run_dir": str(run_dir),
            "plots_dir": str(run_dir / "plots"),
            "plots_generated": [],
            "missing_inputs": [],
            "warnings": [],
            "coverage": None,
        }

        # FR-3: ensure plots/ exists (contract helper stubs report + plots folder). :contentReference[oaicite:23]{index=23}
        try:
            ensure_run_dir(run_dir)
        except Exception as exc:
            result["ok"] = False
            result["warnings"].append(f"ensure_run_dir failed: {exc}")
            return result

        plots_dir = run_dir / "plots"
        try:
            plots_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        # Ensure log compatibility bridge exists (events.jsonl vs evidence_log.jsonl). :contentReference[oaicite:24]{index=24}
        try:
            ensure_log_compat(run_dir)
        except Exception:
            # Best-effort only.
            pass

        man = dict(manifest) if isinstance(manifest, Mapping) else None
        man_src = ""
        if man is None:
            man, man_src = _load_manifest_any(run_dir)
        if man is not None and man_src:
            result["manifest_file"] = man_src

        wf = _infer_workflow(run_dir, man, workflow)
        result["workflow"] = wf

        # Ingest logs for convergence/FMM + coverage panel (FR-4/FR-9.6).
        normalized_events, coverage = self._load_and_merge_logs(run_dir, extra_paths=None)
        result["coverage"] = coverage.snapshot()

        # Convergence plot for any run with residual telemetry (FR-7 Solve dashboard).
        conv_path = plots_dir / "convergence.png"
        conv_ok = self._plot_convergence(normalized_events, conv_path, warnings=result["warnings"])
        if conv_ok:
            result["plots_generated"].append("convergence.png")

        # Images discovery plots if discovered_system.json exists (FR-9.1).
        if (run_dir / "discovered_system.json").is_file():
            ok1 = self._plot_basis_scatter(run_dir, plots_dir / "basis_scatter.png", warnings=result["warnings"])
            if ok1:
                result["plots_generated"].append("basis_scatter.png")
            ok2 = self._plot_family_mass(run_dir, plots_dir / "family_mass.png", warnings=result["warnings"])
            if ok2:
                result["plots_generated"].append("family_mass.png")
        else:
            result["missing_inputs"].append("discovered_system.json")

        # Learn dashboard: loss curve from metrics.jsonl (FR-7 Learning dashboard).
        if (run_dir / "metrics.jsonl").is_file():
            ok3 = self._plot_learn_loss(run_dir / "metrics.jsonl", plots_dir / "loss_curve.png", warnings=result["warnings"])
            if ok3:
                result["plots_generated"].append("loss_curve.png")
        # train_log.jsonl exists in some learn variants; include in coverage but plot is metrics.jsonl-based.

        # FMM dashboard: accuracy/runtime plots (FR-7 FMM dashboard).
        fmm_ok = self._plot_fmm(normalized_events, plots_dir, warnings=result["warnings"])
        if fmm_ok.get("accuracy"):
            result["plots_generated"].append("fmm_accuracy.png")
        if fmm_ok.get("runtime"):
            result["plots_generated"].append("fmm_runtime.png")

        # Gate dashboard + log coverage PNG (FR-9.5 + FR-9.6).
        gate = self.generate_gate_dashboard(
            run_dir,
            manifest=man,
            workflow=wf,
            coverage=coverage.snapshot(),
        )
        result["gate_dashboard"] = gate
        if gate.get("gate_dashboard_png"):
            result["plots_generated"].append("gate_dashboard.png")
        if gate.get("log_coverage_png"):
            result["plots_generated"].append("log_coverage.png")
        if gate.get("gate_dashboard_json"):
            result["plots_generated"].append("gate_dashboard.json")

        return result

    def generate_gate_dashboard(
        self,
        run_dir: Path,
        *,
        manifest: Mapping[str, Any] | None = None,
        workflow: str | None = None,
        coverage: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        """
        Generate gate_dashboard.json and gate_dashboard.png (FR-9.5) and a small log_coverage.png (FR-9.6).

        Gate computation guidance:
        - Gate2: reuse tools/images_gate2.compute_structural_summary when possible. :contentReference[oaicite:25]{index=25}
        - Gate3: use electrodrive.discovery.novelty utilities (novelty_score + compute_gate3_status). :contentReference[oaicite:26]{index=26}
        - Avoid mutating user-owned manifests; compute and write into plots/gate_dashboard.json (design intent).
        """
        run_dir = Path(run_dir)
        plots_dir = run_dir / "plots"
        try:
            plots_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        man = dict(manifest) if isinstance(manifest, Mapping) else None
        if man is None:
            man, _ = _load_manifest_any(run_dir)
        wf = _infer_workflow(run_dir, man, workflow)

        warnings: List[str] = []
        dashboard: Dict[str, Any] = {
            "run_id": (man or {}).get("run_id") or (man or {}).get("id") or run_dir.name,
            "workflow": wf,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())),
            "gate2_status": None,
            "structure_score": None,
            "gate3_status": None,
            "novelty_score": None,
            "gate2_summary": None,
            "log_coverage": dict(coverage or {}),
            "warnings": warnings,
        }

        # Try to pull gate fields from manifest (v1 schema uses manifest["gate"]). :contentReference[oaicite:27]{index=27}
        gate_block = None
        if isinstance(man, Mapping):
            gb = man.get("gate")
            if isinstance(gb, Mapping):
                gate_block = dict(gb)
        if gate_block:
            dashboard["gate2_status"] = gate_block.get("gate2_status")
            dashboard["structure_score"] = gate_block.get("structure_score")
            dashboard["gate3_status"] = gate_block.get("gate3_status")
            dashboard["novelty_score"] = gate_block.get("novelty_score")

        # If this is an images_discover run, attempt computation from discovered_system.json/spec.
        discovered_path = run_dir / "discovered_system.json"
        disc_manifest_path = run_dir / "discovery_manifest.json"

        if discovered_path.is_file():
            try:
                gate2_summary, gate3 = self._compute_gates_from_discovery(
                    discovered_path=discovered_path,
                    discovery_manifest_path=disc_manifest_path if disc_manifest_path.is_file() else None,
                    run_dir=run_dir,
                )
                dashboard["gate2_summary"] = gate2_summary
                if gate2_summary:
                    dashboard["gate2_status"] = gate2_summary.get("gate2_status")
                    dashboard["structure_score"] = gate2_summary.get("structure_score")
                if gate3:
                    dashboard["gate3_status"] = gate3.get("gate3_status")
                    dashboard["novelty_score"] = gate3.get("novelty_score")
            except Exception as exc:
                warnings.append(f"gate computation failed: {exc!r}")
        else:
            warnings.append("discovered_system.json missing; gate2/gate3 may be n/a")

        # Guarantee stable keys for consumers.
        if dashboard["gate2_status"] is None:
            dashboard["gate2_status"] = "n/a"
        if dashboard["gate3_status"] is None:
            dashboard["gate3_status"] = "n/a"

        # Write JSON (always).
        json_path = plots_dir / "gate_dashboard.json"
        _atomic_write_text(json_path, _safe_json_dump(dashboard))
        out: Dict[str, Any] = {"gate_dashboard_json": str(json_path.relative_to(run_dir))}

        # Write PNGs (best-effort).
        plt = _try_import_matplotlib()
        gate_png = plots_dir / "gate_dashboard.png"
        cov_png = plots_dir / "log_coverage.png"

        if plt is None:
            warnings.append("matplotlib unavailable; wrote placeholder PNGs")
            _write_placeholder_png(gate_png, "matplotlib missing")
            _write_placeholder_png(cov_png, "matplotlib missing")
            out["gate_dashboard_png"] = str(gate_png.relative_to(run_dir))
            out["log_coverage_png"] = str(cov_png.relative_to(run_dir))
            out["warnings"] = warnings
            return out

        # gate_dashboard.png (text summary)
        try:
            fig = plt.figure(figsize=(8.5, 4.5))
            ax = fig.add_subplot(111)
            ax.axis("off")
            lines = [
                f"Run: {dashboard.get('run_id')}  Workflow: {dashboard.get('workflow')}",
                f"Gate2: {dashboard.get('gate2_status')}   structure_score={dashboard.get('structure_score')}",
                f"Gate3: {dashboard.get('gate3_status')}   novelty_score={dashboard.get('novelty_score')}",
                "",
                "Warnings:",
            ]
            if warnings:
                lines.extend([f" - {w}" for w in warnings[:12]])
            else:
                lines.append(" (none)")
            ax.text(0.01, 0.98, "\n".join(lines), va="top", ha="left")
            fig.tight_layout()
            fig.savefig(gate_png, dpi=150)
            plt.close(fig)
            out["gate_dashboard_png"] = str(gate_png.relative_to(run_dir))
        except Exception as exc:
            warnings.append(f"failed to render gate_dashboard.png: {exc!r}")
            _write_placeholder_png(gate_png, "gate plot failed")
            out["gate_dashboard_png"] = str(gate_png.relative_to(run_dir))

        # log_coverage.png (text summary)
        try:
            cov = dict(coverage or {})
            fig2 = plt.figure(figsize=(8.5, 5.5))
            ax2 = fig2.add_subplot(111)
            ax2.axis("off")
            lines2 = [
                "Log coverage (FR-9.6):",
                f"  total_lines_seen={cov.get('total_lines_seen')}  parsed={cov.get('total_records_parsed')}  emitted={cov.get('total_records_emitted')}",
                f"  json_errors={cov.get('total_json_errors')}  non_dict={cov.get('total_non_dict_records')}  dedup_drops={cov.get('dropped_by_dedup')}",
                "",
                "  ingested_files:",
            ]
            for fn in (cov.get("ingested_files") or [])[:20]:
                lines2.append(f"    - {fn}")
            lines2.append("")
            lines2.append("  event_source_counts:")
            esc = cov.get("event_source_counts") or {}
            for k in sorted(esc.keys()):
                lines2.append(f"    {k}: {esc[k]}")
            lines2.append("")
            lines2.append("  residual_field_detection_counts:")
            rfc = cov.get("residual_field_detection_counts") or {}
            for k in sorted(rfc.keys()):
                lines2.append(f"    {k}: {rfc[k]}")
            ax2.text(0.01, 0.98, "\n".join(lines2), va="top", ha="left", family="monospace")
            fig2.tight_layout()
            fig2.savefig(cov_png, dpi=150)
            plt.close(fig2)
            out["log_coverage_png"] = str(cov_png.relative_to(run_dir))
        except Exception as exc:
            warnings.append(f"failed to render log_coverage.png: {exc!r}")
            _write_placeholder_png(cov_png, "coverage plot failed")
            out["log_coverage_png"] = str(cov_png.relative_to(run_dir))

        out["warnings"] = warnings
        return out

    # ----------------------------
    # Log ingestion / normalization
    # ----------------------------

    def _load_and_merge_logs(
        self,
        run_dir: Path,
        extra_paths: Sequence[Path] | None,
    ) -> Tuple[List[Dict[str, Any]], _LogCoverage]:
        """
        Read and normalize both events.jsonl and evidence_log.jsonl if present (FR-4 + §1.4),
        deduplicating by a stable fingerprint (FR-4 multi-file merge).

        Normalization selection:
        - prefer electrodrive.researched.ingest.normalizer.normalize_event (dependency-free module)
        - fall back to electrodrive.researched.ws.normalize_event if available
        - otherwise use a minimal built-in normalizer (FR-4 compliant)
        """
        normalize_event = _get_normalize_event()

        run_dir = Path(run_dir)
        coverage = _LogCoverage()
        dedup = _DedupCache(max_items=75_000)

        paths: List[Path] = []
        for name in ("events.jsonl", "evidence_log.jsonl", "researched_events.jsonl"):
            p = run_dir / name
            if p.is_file():
                paths.append(p)

        # Include learn logs in coverage if present (FR-9.6).
        for name in ("metrics.jsonl", "train_log.jsonl", "stdout.log", "stderr.log"):
            p = run_dir / name
            if p.is_file():
                paths.append(p)

        if extra_paths:
            for p in extra_paths:
                if Path(p).is_file():
                    paths.append(Path(p))

        out: List[Dict[str, Any]] = []

        for p in paths:
            coverage.note_file(p)
            # Prefer line-by-line for JSONL; for stdout.log, we treat as raw text lines => synth records.
            if p.name.endswith(".log") and not p.name.endswith(".jsonl"):
                self._ingest_text_log(p, out, coverage, dedup, normalize_event)
                continue
            self._ingest_jsonl(p, out, coverage, dedup, normalize_event)

        # Sort by time ascending.
        try:
            out.sort(key=lambda ev: (float(ev.get("t", 0.0)), str(ev.get("source", "")), str(ev.get("event", ""))))
        except Exception:
            pass

        return out, coverage

    def _ingest_jsonl(
        self,
        path: Path,
        out: List[Dict[str, Any]],
        coverage: _LogCoverage,
        dedup: _DedupCache,
        normalize_event_fn,
    ) -> None:
        """
        Ingest JSONL file defensively. Never raises.
        """
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue
                    coverage.note_line(path)
                    try:
                        rec = json.loads(line)
                    except Exception:
                        coverage.note_json_error(path)
                        continue
                    if not isinstance(rec, dict):
                        coverage.note_non_dict(path)
                        continue
                    coverage.note_parsed(path)

                    ev = normalize_event_fn(rec, ingest_time=time.time(), source=path.name)
                    fp = _fingerprint_norm(ev)
                    if dedup.seen(fp):
                        coverage.note_dedup_drop()
                        continue
                    out.append(ev)
                    coverage.note_emitted(path, float(ev.get("t", 0.0)), str(ev.get("event_source", "")), rec)
        except Exception:
            return

    def _ingest_text_log(
        self,
        path: Path,
        out: List[Dict[str, Any]],
        coverage: _LogCoverage,
        dedup: _DedupCache,
        normalize_event_fn,
    ) -> None:
        """
        Ingest a raw text log by synthesizing JsonlLogger-like records {ts, level, msg} (FR-9.6 coverage).

        This matches the repo’s structured logger key choice {ts, level, msg}. :contentReference[oaicite:30]{index=30}
        """
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = line.rstrip("\n").rstrip("\r")
                    if not line:
                        continue
                    coverage.note_line(path)
                    rec = {"ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(time.time())), "level": "INFO", "msg": line}
                    coverage.note_parsed(path)
                    ev = normalize_event_fn(rec, ingest_time=time.time(), source=path.name)
                    fp = _fingerprint_norm(ev)
                    if dedup.seen(fp):
                        coverage.note_dedup_drop()
                        continue
                    out.append(ev)
                    coverage.note_emitted(path, float(ev.get("t", 0.0)), str(ev.get("event_source", "")), rec)
        except Exception:
            return

    # ----------------------------
    # Plot generation
    # ----------------------------

    def _plot_convergence(self, events: Sequence[Mapping[str, Any]], out_path: Path, *, warnings: List[str]) -> bool:
        """
        Convergence plot: iter vs residual(s). Must not assume a specific event name (FR-4/FR-7).
        """
        xs: List[int] = []
        y_resid: List[float] = []
        y_pre: List[float] = []
        y_true: List[float] = []

        for idx, ev in enumerate(events):
            it = ev.get("iter")
            xi = _coerce_int(it)
            if xi is None:
                # fallback: sequential index (still useful).
                xi = idx
            r = _coerce_float(ev.get("resid"))
            rp = _coerce_float(ev.get("resid_precond"))
            rt = _coerce_float(ev.get("resid_true"))
            if r is None and rp is None and rt is None:
                continue
            xs.append(int(xi))
            y_resid.append(float(r) if r is not None else float("nan"))
            y_pre.append(float(rp) if rp is not None else float("nan"))
            y_true.append(float(rt) if rt is not None else float("nan"))

        if not xs:
            warnings.append("no residual telemetry found; convergence.png not generated")
            return False

        plt = _try_import_matplotlib()
        if plt is None:
            warnings.append("matplotlib unavailable; writing placeholder convergence.png")
            _write_placeholder_png(out_path, "matplotlib missing")
            return True

        try:
            fig = plt.figure(figsize=(7.0, 4.0))
            ax = fig.add_subplot(111)

            def _plot_series(y: List[float], label: str) -> None:
                # Filter nan.
                pts = [(x, v) for x, v in zip(xs, y) if isinstance(v, float) and math.isfinite(v) and v > 0.0]
                if len(pts) < 2:
                    return
                x2 = [p[0] for p in pts]
                y2 = [p[1] for p in pts]
                ax.plot(x2, y2, label=label)

            # Prefer separate resid_true/resid_precond if present, else resid.
            _plot_series(y_pre, "resid_precond")
            _plot_series(y_true, "resid_true")
            if not any(math.isfinite(v) for v in y_pre) and not any(math.isfinite(v) for v in y_true):
                _plot_series(y_resid, "resid")

            ax.set_xlabel("iteration")
            ax.set_ylabel("residual")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.2)
            ax.legend(loc="best")
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return True
        except Exception as exc:
            warnings.append(f"failed to render convergence plot: {exc!r}")
            _write_placeholder_png(out_path, "plot failed")
            return True

    def _plot_basis_scatter(self, run_dir: Path, out_path: Path, *, warnings: List[str]) -> bool:
        """
        Basis expressivity scatter (FR-9.1) from discovered_system.json alone (preferred). :contentReference[oaicite:31]{index=31}
        """
        sys = self._load_image_system_safe(run_dir / "discovered_system.json", warnings=warnings)
        if sys is None:
            warnings.append("basis_scatter: failed to load discovered_system.json")
            return False

        elements, weights = sys
        pts: List[Tuple[float, float, str]] = []

        for elem, w in zip(elements, weights):
            z = self._z_from_elem(elem)
            if z is None:
                continue
            fam = self._family_name(elem)
            pts.append((float(z), float(abs(w)), fam))

        if not pts:
            warnings.append("basis_scatter: no usable positions/weights found")
            return False

        plt = _try_import_matplotlib()
        if plt is None:
            warnings.append("matplotlib unavailable; writing placeholder basis_scatter.png")
            _write_placeholder_png(out_path, "matplotlib missing")
            return True

        try:
            families = sorted({p[2] for p in pts})
            # Simple categorical color mapping via matplotlib default cycle.
            fig = plt.figure(figsize=(7.5, 4.5))
            ax = fig.add_subplot(111)

            for fam in families:
                xs = [p[0] for p in pts if p[2] == fam]
                ys = [p[1] for p in pts if p[2] == fam]
                ax.scatter(xs, ys, s=12, alpha=0.8, label=fam)

            ax.set_xlabel("z position")
            ax.set_ylabel("|weight|")
            ax.set_yscale("log")
            ax.grid(True, which="both", alpha=0.2)
            # Avoid huge legends.
            if len(families) <= 12:
                ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return True
        except Exception as exc:
            warnings.append(f"basis_scatter render failed: {exc!r}")
            _write_placeholder_png(out_path, "plot failed")
            return True

    def _plot_family_mass(self, run_dir: Path, out_path: Path, *, warnings: List[str]) -> bool:
        """
        Family mass bar chart (FR-9.1) from discovered_system.json alone. :contentReference[oaicite:32]{index=32}
        """
        sys = self._load_image_system_safe(run_dir / "discovered_system.json", warnings=warnings)
        if sys is None:
            warnings.append("family_mass: failed to load discovered_system.json")
            return False

        elements, weights = sys
        fam_count: Dict[str, int] = {}
        fam_mass: Dict[str, float] = {}
        for elem, w in zip(elements, weights):
            fam = self._family_name(elem)
            fam_count[fam] = fam_count.get(fam, 0) + 1
            fam_mass[fam] = fam_mass.get(fam, 0.0) + float(abs(w))

        if not fam_mass:
            warnings.append("family_mass: no usable weights found")
            return False

        plt = _try_import_matplotlib()
        if plt is None:
            warnings.append("matplotlib unavailable; writing placeholder family_mass.png")
            _write_placeholder_png(out_path, "matplotlib missing")
            return True

        try:
            items = sorted(fam_mass.items(), key=lambda kv: kv[1], reverse=True)
            labels = [k for k, _ in items]
            masses = [v for _, v in items]
            counts = [fam_count.get(k, 0) for k in labels]

            fig = plt.figure(figsize=(8.5, 4.5))
            ax = fig.add_subplot(111)
            ax.bar(range(len(labels)), masses)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("L1 mass (sum |w|)")
            ax.grid(True, axis="y", alpha=0.2)

            # annotate counts (small)
            for i, c in enumerate(counts):
                try:
                    ax.text(i, masses[i], f"n={c}", ha="center", va="bottom", fontsize=7, rotation=0)
                except Exception:
                    pass

            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return True
        except Exception as exc:
            warnings.append(f"family_mass render failed: {exc!r}")
            _write_placeholder_png(out_path, "plot failed")
            return True

    def _plot_learn_loss(self, metrics_jsonl: Path, out_path: Path, *, warnings: List[str]) -> bool:
        """
        Learning dashboard loss curve from metrics.jsonl (FR-7). Writer appends json.dumps(data) lines. :contentReference[oaicite:33]{index=33}
        """
        rows: List[Dict[str, Any]] = []
        try:
            with metrics_jsonl.open("r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    line = (line or "").strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if isinstance(rec, dict):
                        rows.append(rec)
        except Exception:
            warnings.append("loss_curve: failed to read metrics.jsonl")
            return False

        if not rows:
            warnings.append("loss_curve: metrics.jsonl empty or malformed")
            return False

        xs: List[int] = []
        series: Dict[str, List[float]] = {}

        # Pick up to 3 keys with "loss" (prefer non-val, then val) or fall back to first numeric key.
        key_candidates: List[str] = []
        try:
            all_keys = set().union(*(r.keys() for r in rows))
            loss_keys = [k for k in all_keys if "loss" in str(k).lower()]
            non_val = sorted([k for k in loss_keys if not str(k).lower().startswith("val_")])
            val = sorted([k for k in loss_keys if str(k).lower().startswith("val_")])
            key_candidates = (non_val + val)[:3]
            if not key_candidates:
                # fallback: any numeric field except step/lr/elapsed_time
                for k in sorted(all_keys):
                    if k in {"step", "lr", "elapsed_time"}:
                        continue
                    # check first row numeric-ish
                    v = rows[0].get(k)
                    if _coerce_float(v) is not None:
                        key_candidates = [str(k)]
                        break
        except Exception:
            key_candidates = []

        if not key_candidates:
            warnings.append("loss_curve: no numeric metrics fields found")
            return False

        for r in rows:
            step = _coerce_int(r.get("step"))
            if step is None:
                continue
            xs.append(step)
            for k in key_candidates:
                series.setdefault(k, []).append(_coerce_float(r.get(k)) or float("nan"))

        if len(xs) < 2:
            warnings.append("loss_curve: insufficient points")
            return False

        plt = _try_import_matplotlib()
        if plt is None:
            warnings.append("matplotlib unavailable; writing placeholder loss_curve.png")
            _write_placeholder_png(out_path, "matplotlib missing")
            return True

        try:
            fig = plt.figure(figsize=(7.0, 4.0))
            ax = fig.add_subplot(111)
            for k, ys in series.items():
                pts = [(x, y) for x, y in zip(xs, ys) if isinstance(y, float) and math.isfinite(y)]
                if len(pts) < 2:
                    continue
                ax.plot([p[0] for p in pts], [p[1] for p in pts], label=str(k))
            ax.set_xlabel("step")
            ax.set_ylabel("metric")
            ax.grid(True, alpha=0.2)
            if len(series) > 1:
                ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            return True
        except Exception as exc:
            warnings.append(f"loss_curve render failed: {exc!r}")
            _write_placeholder_png(out_path, "plot failed")
            return True

    def _plot_fmm(self, events: Sequence[Mapping[str, Any]], plots_dir: Path, *, warnings: List[str]) -> Dict[str, bool]:
        """
        FMM dashboard plots (FR-7): accuracy/runtime.

        FMM JSONL record uses event="fmm_test_result" with max_abs_err/rel_l2_err + extra (e.g., wall_time_s). :contentReference[oaicite:34]{index=34}
        """
        acc_path = plots_dir / "fmm_accuracy.png"
        rt_path = plots_dir / "fmm_runtime.png"

        tests: List[Mapping[str, Any]] = []
        for ev in events:
            if str(ev.get("event", "")).strip() == "fmm_test_result":
                tests.append(ev)

        if not tests:
            return {"accuracy": False, "runtime": False}

        xs = list(range(len(tests)))
        rel = []
        max_abs = []
        wall = []
        labels = []
        for ev in tests:
            fields = ev.get("fields") or {}
            if isinstance(fields, dict):
                labels.append(str(fields.get("name", ""))[:30])
                rel.append(_coerce_float(fields.get("rel_l2_err")) or _coerce_float(fields.get("rel_l2")) or float("nan"))
                max_abs.append(_coerce_float(fields.get("max_abs_err")) or float("nan"))
                wall.append(_coerce_float(fields.get("wall_time_s")) or float("nan"))
            else:
                labels.append("")
                rel.append(float("nan"))
                max_abs.append(float("nan"))
                wall.append(float("nan"))

        plt = _try_import_matplotlib()
        if plt is None:
            warnings.append("matplotlib unavailable; writing placeholder fmm plots")
            _write_placeholder_png(acc_path, "matplotlib missing")
            _write_placeholder_png(rt_path, "matplotlib missing")
            return {"accuracy": True, "runtime": True}

        ok_acc = False
        ok_rt = False

        try:
            fig = plt.figure(figsize=(8.5, 4.0))
            ax = fig.add_subplot(111)
            # filter finite positive
            pts_rel = [(i, v) for i, v in zip(xs, rel) if isinstance(v, float) and math.isfinite(v) and v > 0]
            pts_abs = [(i, v) for i, v in zip(xs, max_abs) if isinstance(v, float) and math.isfinite(v) and v > 0]
            if pts_rel:
                ax.plot([p[0] for p in pts_rel], [p[1] for p in pts_rel], label="rel_l2_err")
            if pts_abs:
                ax.plot([p[0] for p in pts_abs], [p[1] for p in pts_abs], label="max_abs_err")
            ax.set_yscale("log")
            ax.set_xlabel("test index")
            ax.set_ylabel("error")
            ax.grid(True, which="both", alpha=0.2)
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            fig.savefig(acc_path, dpi=150)
            plt.close(fig)
            ok_acc = True
        except Exception as exc:
            warnings.append(f"fmm_accuracy render failed: {exc!r}")
            _write_placeholder_png(acc_path, "plot failed")
            ok_acc = True

        try:
            pts_wall = [(i, v) for i, v in zip(xs, wall) if isinstance(v, float) and math.isfinite(v) and v >= 0]
            if pts_wall:
                fig2 = plt.figure(figsize=(8.5, 3.5))
                ax2 = fig2.add_subplot(111)
                ax2.plot([p[0] for p in pts_wall], [p[1] for p in pts_wall], label="wall_time_s")
                ax2.set_xlabel("test index")
                ax2.set_ylabel("seconds")
                ax2.grid(True, alpha=0.2)
                fig2.tight_layout()
                fig2.savefig(rt_path, dpi=150)
                plt.close(fig2)
                ok_rt = True
        except Exception as exc:
            warnings.append(f"fmm_runtime render failed: {exc!r}")
            _write_placeholder_png(rt_path, "plot failed")
            ok_rt = True

        return {"accuracy": ok_acc, "runtime": ok_rt}

    # ----------------------------
    # Gate computation helpers (best-effort)
    # ----------------------------

    def _compute_gates_from_discovery(
        self,
        *,
        discovered_path: Path,
        discovery_manifest_path: Path | None,
        run_dir: Path,
    ) -> Tuple[Dict[str, Any] | None, Dict[str, Any] | None]:
        """
        Compute Gate2 structural summary and Gate3 novelty status without mutating manifests.

        Gate2 implementation is in tools/images_gate2.compute_structural_summary. :contentReference[oaicite:35]{index=35}
        Gate3 implementation is in electrodrive.discovery.novelty (compute_gate3_status). :contentReference[oaicite:36]{index=36}
        """
        # Load discovery manifest for numeric/conditioning/gate1 context if present.
        disc_man: Dict[str, Any] = {}
        if discovery_manifest_path is not None and discovery_manifest_path.is_file():
            obj = _safe_json_load(discovery_manifest_path)
            if isinstance(obj, dict):
                disc_man = obj

        # Resolve spec path (best-effort: manifest spec_path may be relative to repo root).
        spec_path = None
        for cand in (
            (disc_man.get("spec_path") if isinstance(disc_man, dict) else None),
            (disc_man.get("inputs", {}).get("spec_path") if isinstance(disc_man.get("inputs", {}), dict) else None),
        ):
            if isinstance(cand, str) and cand.strip():
                spec_path = cand.strip()
                break

        # Fall back to ResearchED manifest inputs.spec_path if present.
        man, _ = _load_manifest_any(run_dir)
        if spec_path is None and isinstance(man, dict):
            inp = man.get("inputs")
            if isinstance(inp, dict):
                sp = inp.get("spec_path") or inp.get("problem") or inp.get("spec")
                if isinstance(sp, str) and sp.strip():
                    spec_path = sp.strip()

        # Load ImageSystem via repo IO helper (avoids constructor mismatches).
        system_obj = None
        try:
            import torch  # type: ignore
            from electrodrive.images.io import load_image_system  # type: ignore

            system_obj = load_image_system(discovered_path, device="cpu", dtype=torch.float32)
        except Exception:
            system_obj = None

        if system_obj is None:
            return None, None

        # Compute Gate2 if we can load spec and import gate utilities.
        gate2_summary: Dict[str, Any] | None = None
        gate3: Dict[str, Any] | None = None

        spec_obj = None
        if spec_path is not None:
            spec_obj = self._load_spec_best_effort(spec_path, run_dir=run_dir)

        if spec_obj is not None:
            # Gate2
            try:
                try:
                    from electrodrive.tools.images_gate2 import compute_structural_summary  # type: ignore
                except Exception:
                    # Fallback for alternate layouts / dev paths.
                    from tools.images_gate2 import compute_structural_summary  # type: ignore

                # numeric_status/condition_status are defined in discovery_manifest.json in tools/images_gate2 main. :contentReference[oaicite:37]{index=37}
                numeric_status = disc_man.get("numeric_status", "ok") if isinstance(disc_man, dict) else "ok"
                condition_status = disc_man.get("condition_status", None) if isinstance(disc_man, dict) else None

                gate2_summary = compute_structural_summary(
                    spec_obj,
                    system_obj,
                    numeric_status=numeric_status,
                    condition_status=condition_status,
                )
            except Exception:
                gate2_summary = None

            # Gate3 novelty
            try:
                from electrodrive.discovery import novelty as nov  # type: ignore

                fp = nov.structural_fingerprint(system_obj, spec_obj)
                novelty_val = None
                g2 = (gate2_summary or {}).get("gate2_status")
                if g2 in {"pass", "borderline"}:
                    novelty_val = float(nov.novelty_score(fp))
                gate3_status, novelty_score_val = nov.compute_gate3_status(
                    disc_man if isinstance(disc_man, dict) else {},
                    novelty_val,
                )
                gate3 = {"gate3_status": gate3_status, "novelty_score": novelty_score_val}
            except Exception:
                gate3 = None

        return gate2_summary, gate3

    def _load_spec_best_effort(self, spec_path: str, *, run_dir: Path) -> Any | None:
        """
        Best-effort spec loader for Gate2/Gate3.
        tools/images_gate2 uses CanonicalSpec.from_json on JSON file. :contentReference[oaicite:38]{index=38}
        """
        try:
            from electrodrive.orchestration.parser import CanonicalSpec  # type: ignore
        except Exception:
            return None

        cand = Path(spec_path).expanduser()
        if cand.is_absolute() and cand.exists():
            p = cand
        else:
            # Try relative to run_dir, then repo root (find .git upward from this file).
            p = run_dir / cand
            if not p.exists():
                repo_root = self._find_repo_root(Path(__file__).resolve())
                p2 = (repo_root / cand) if repo_root is not None else p
                p = p2 if p2.exists() else p

        if not p.exists():
            return None
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return CanonicalSpec.from_json(data)
        except Exception:
            return None

    def _find_repo_root(self, start: Path) -> Path | None:
        try:
            start = start.resolve()
        except Exception:
            pass
        for p in [start, *start.parents]:
            try:
                if (p / ".git").exists():
                    return p
            except Exception:
                continue
        return None

    # ----------------------------
    # Discovered system helpers
    # ----------------------------

    def _load_image_system_safe(self, path: Path, *, warnings: List[str]) -> Tuple[List[Any], List[float]] | None:
        """
        Load discovered_system.json via electrodrive.images.io.load_image_system (preferred). :contentReference[oaicite:39]{index=39}

        Returns (elements, weights_list) for simple downstream plotting without forcing torch usage elsewhere.
        """
        try:
            import torch  # type: ignore
            from electrodrive.images.io import load_image_system  # type: ignore

            system = load_image_system(path, device="cpu", dtype=torch.float32)
            elements = list(getattr(system, "elements", []))
            w = getattr(system, "weights", None)
            weights: List[float] = []
            if w is not None:
                try:
                    # torch tensor -> list
                    weights = [float(x) for x in w.detach().cpu().view(-1).tolist()]
                except Exception:
                    weights = []
            if not elements or not weights or len(elements) != len(weights):
                # Fall back to raw JSON parsing for weights mismatch.
                raise ValueError("ImageSystem elements/weights mismatch")
            return elements, weights
        except Exception:
            # Fallback: parse JSON directly (format: data["images"][i]["weight"]). :contentReference[oaicite:40]{index=40}
            try:
                data = json.loads(Path(path).read_text(encoding="utf-8"))
                imgs = data.get("images", []) if isinstance(data, dict) else []
                elements = [img for img in imgs if isinstance(img, dict)]
                weights = [float(img.get("weight", 1.0)) for img in elements]
                # Elements are raw dicts here; plotting helpers handle both dict and ImageBasisElement.
                return elements, weights
            except Exception as exc:
                warnings.append(f"failed to load discovered_system.json: {exc!r}")
                return None

    def _family_name(self, elem: Any) -> str:
        """
        Prefer group_info.family_name when present; it is serialized as "group_info" and becomes _group_info on deserialize. :contentReference[oaicite:41]{index=41} :contentReference[oaicite:42]{index=42}
        Mirrors tools/images_gate2._family_name behavior. :contentReference[oaicite:43]{index=43}
        """
        try:
            info = getattr(elem, "_group_info", None)
            if isinstance(info, dict) and info.get("family_name"):
                return str(info.get("family_name"))
        except Exception:
            pass
        # Raw dict fallback (when we loaded JSON directly).
        if isinstance(elem, dict):
            gi = elem.get("group_info") or elem.get("_group_info") or {}
            if isinstance(gi, dict) and gi.get("family_name"):
                return str(gi.get("family_name"))
            t = elem.get("type")
            return str(t) if t is not None else "unknown"
        try:
            return str(getattr(elem, "type", "unknown"))
        except Exception:
            return "unknown"

    def _z_from_elem(self, elem: Any) -> float | None:
        """
        Best-effort z position extractor for basis_scatter:
        - For ImageBasisElement, look in elem.params["position"] or elem.params["center"].
          PointChargeBasis requires params["position"]. :contentReference[oaicite:44]{index=44}
        - For raw dict element, look in ["params"]["position"] etc.
        """
        try:
            if hasattr(elem, "params"):
                params = getattr(elem, "params")
                if isinstance(params, dict):
                    for k in ("position", "center"):
                        if k in params:
                            v = params.get(k)
                            # torch tensor -> list
                            try:
                                import torch  # type: ignore

                                if isinstance(v, torch.Tensor):
                                    v = v.detach().cpu().view(-1).tolist()
                            except Exception:
                                pass
                            if isinstance(v, (list, tuple)) and len(v) >= 3:
                                return float(v[2])
        except Exception:
            pass

        if isinstance(elem, dict):
            params = elem.get("params", {})
            if isinstance(params, dict):
                for k in ("position", "center"):
                    v = params.get(k)
                    if isinstance(v, (list, tuple)) and len(v) >= 3:
                        try:
                            return float(v[2])
                        except Exception:
                            continue
        return None

