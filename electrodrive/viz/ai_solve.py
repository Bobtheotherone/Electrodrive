"""
Post-hoc AI-solve overlay utility (PR-1A).

Behavior:
    - Reads OUT_DIR/metrics.json and OUT_DIR/events.jsonl (or evidence_log.jsonl).
    - Extracts residual vs iteration and tile/autotiling decisions.
    - Overlays a small textual summary on top of each PNG produced by
      _render_visualizations (viz/*.png).

Constraints:
    - Torch-only / stdlib-only; no new hard dependencies.
    - Must NOT re-run the solver.
    - Safe to run even if some files are missing.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from PIL import Image, ImageDraw, ImageFont  # type: ignore

    _HAVE_PIL = True
except Exception:  # pragma: no cover
    _HAVE_PIL = False


def _first_present(rec: Dict[str, Any], keys) -> Any:
    for k in keys:
        if k in rec:
            v = rec.get(k)
            if v is not None:
                return v
    return None


def _safe_float(v: Any) -> Optional[float]:
    try:
        if isinstance(v, (int, float)):
            return float(v)
        if isinstance(v, str) and v.strip():
            return float(v.strip())
    except Exception:
        return None
    return None


def _event_name(rec: Dict[str, Any]) -> str:
    return str(rec.get("event") or rec.get("msg") or rec.get("message") or "")


def _iter_value(rec: Dict[str, Any]) -> Optional[int]:
    raw = _first_present(rec, ("iter", "iters", "step", "k"))
    val = _safe_float(raw)
    if val is None:
        return None
    try:
        return int(val)
    except Exception:
        return None


def _resid_value(rec: Dict[str, Any]) -> Optional[float]:
    raw = _first_present(
        rec,
        ("resid", "resid_true", "resid_precond", "resid_true_l2", "resid_precond_l2"),
    )
    val = _safe_float(raw)
    if val is None:
        return None
    return float(val)


def _load_metrics(out_dir: Path) -> Dict[str, Any]:
    mpath = out_dir / "metrics.json"
    if not mpath.is_file():
        return {}
    try:
        with mpath.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data.get("metrics", data)
    except Exception:
        return {}
    return {}


def _load_events(out_dir: Path) -> List[Dict[str, Any]]:
    # Prefer events.jsonl (PR-1A); fall back to legacy evidence_log.jsonl.
    ev_path = out_dir / "events.jsonl"
    if not ev_path.is_file():
        alt = out_dir / "evidence_log.jsonl"
        if alt.is_file():
            ev_path = alt
        else:
            return []
    events: List[Dict[str, Any]] = []
    try:
        with ev_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict):
                        events.append(rec)
                except Exception:
                    continue
    except Exception:
        return []
    return events


def _extract_solver_trace(
    events: List[Dict[str, Any]]
) -> Tuple[List[int], List[float], Dict[str, Any]]:
    iters: List[int] = []
    res: List[float] = []
    tile_info: Dict[str, Any] = {}
    for ev in events:
        msg = _event_name(ev).lower()
        if "gmres" in msg and ("iter" in msg or "progress" in msg):
            i = _iter_value(ev)
            r = _resid_value(ev)
            if i is not None and r is not None:
                iters.append(i)
                res.append(r)
                # Capture tiling context (last seen wins).
                for key in ("tile_size", "pass_index", "cycle", "restart"):
                    if key in ev:
                        tile_info[key] = ev[key]
    return iters, res, tile_info


def _annotate_png(path: Path, text: str) -> None:
    if not _HAVE_PIL:
        # Without Pillow, we can't draw; just skip.
        return
    try:
        im = Image.open(path).convert("RGBA")
    except Exception:
        return

    # Create overlay band at top-left.
    draw = ImageDraw.Draw(im)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None  # type: ignore

    margin = 6
    lines = text.split("\n")
    # Rough text box size.
    max_w = 0
    total_h = 0
    for line in lines:
        if not line:
            continue
        sz = draw.textbbox((0, 0), line, font=font)
        w = sz[2] - sz[0]
        h = sz[3] - sz[1]
        max_w = max(max_w, w)
        total_h += h + 2
    box_w = min(im.width, max_w + 2 * margin)
    box_h = min(im.height, total_h + 2 * margin)

    # Semi-transparent dark box.
    box_color = (0, 0, 0, 160)
    draw.rectangle([0, 0, box_w, box_h], fill=box_color)

    # Draw text.
    y = margin
    for line in lines:
        if not line:
            continue
        draw.text((margin, y), line, fill=(255, 255, 255, 255), font=font)
        sz = draw.textbbox((0, 0), line, font=font)
        h = sz[3] - sz[1]
        y += h + 2

    # Save annotated sibling as *_ai.png
    out_path = path.with_name(f"{path.stem}_ai{path.suffix}")
    try:
        im.save(out_path)
    except Exception:
        return


def apply_ai_overlay(out_dir: str) -> None:
    """
    Apply AI-solve overlays in OUT_DIR.

    Safe no-op if metrics/events or PNGs are missing.
    """
    base = Path(out_dir)
    metrics = _load_metrics(base)
    events = _load_events(base)
    iters, res, tile_info = _extract_solver_trace(events)

    if not iters or not res:
        # Nothing to overlay.
        return

    # Build compact summary string.
    try:
        it_min = min(iters)
        it_max = max(iters)
        r0 = res[0]
        r_last = res[-1]
        r_min = min(res)
    except Exception:
        return

    parts = [
        f"AI-solve GMRES iters: {it_min}..{it_max}",
        f"residual: start={r0:.2e}, min={r_min:.2e}, last={r_last:.2e}",
    ]
    if tile_info:
        # Deterministic ordering of a few known keys.
        keys = ["tile_size", "pass_index", "cycle", "restart"]
        kv = []
        for k in keys:
            if k in tile_info:
                kv.append(f"{k}={tile_info[k]}")
        if kv:
            parts.append("tiles: " + ", ".join(kv))
    summary = "\n".join(parts)

    viz_dir = base / "viz"
    if not viz_dir.is_dir():
        return

    for png in sorted(viz_dir.glob("*.png")):
        _annotate_png(png, summary)


if __name__ == "__main__":
    import sys as _sys

    if len(_sys.argv) != 2:
        print("Usage: python -m electrodrive.viz.ai_solve OUT_DIR", file=_sys.stderr)
        raise SystemExit(1)
    apply_ai_overlay(_sys.argv[1])
