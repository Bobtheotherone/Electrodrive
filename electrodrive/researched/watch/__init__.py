"""
ResearchED viz-watching utilities.

Design Doc alignment:
- §3.2 “VizWatcher”: watch `viz/` for new PNGs; provide “latest frame” + timeline/slider
- FR-5: live monitor frames appear as written (detect within ~1–2s locally)
- FR-3: run directory contract includes `run_dir/viz/*.png`

Repo alignment:
- Keep dependencies optional (stdlib-only by default)
- Defensive filesystem handling: never raise to callers due to I/O issues
- Prefer `viz_*.png` frames; fall back to `viz.png` when that is the only output
"""

from __future__ import annotations

from .viz_watcher import FrameInfo, VizEvent, VizWatcher

__all__ = ["VizWatcher", "FrameInfo", "VizEvent"]
