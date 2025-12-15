from __future__ import annotations

from pathlib import Path

import pytest


def test_viz_watcher_scan_once(tmp_path: Path):
    from electrodrive.researched.watch.viz_watcher import VizWatcher

    run_dir = tmp_path / "run"
    viz_dir = run_dir / "viz"
    viz_dir.mkdir(parents=True)

    # minimal PNG signature bytes; enough for file existence, not for rendering
    png = viz_dir / "viz_0001.png"
    png.write_bytes(b"\x89PNG\r\n\x1a\n" + b"0" * 64)

    w = VizWatcher(run_dir=run_dir)
    w.scan_once()
    latest = w.latest_frame_path()
    assert latest is not None
    assert Path(latest).name == "viz_0001.png"
