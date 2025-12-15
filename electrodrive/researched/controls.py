from __future__ import annotations

"""
ResearchED control helpers.

Design Doc alignment (FR-6 / §1.2):
- Snapshot is a string token (not boolean).
- Writes go through electrodrive.live.controls.write_controls to preserve seq/ack
  semantics and atomicity.
"""

import inspect
from pathlib import Path
from typing import Any, Dict, Mapping


def _write_controls_compat(run_dir: Path, updates: Mapping[str, Any] | None = None, *, merge: bool = True) -> Any:
    """
    Call electrodrive.live.controls.write_controls with signature detection.
    """
    from electrodrive.live.controls import write_controls  # type: ignore

    u: Dict[str, Any] = dict(updates or {})

    try:
        sig = inspect.signature(write_controls)
        params = sig.parameters
        args = [run_dir]
        kwargs: Dict[str, Any] = {}

        if "updates" in params:
            kwargs["updates"] = u
        else:
            args.append(u)

        if "merge" in params:
            kwargs["merge"] = merge

        if "seq_increment" in params:
            kwargs["seq_increment"] = True

        return write_controls(*args, **kwargs)
    except TypeError:
        pass
    except Exception:
        pass

    # Fallback attempts.
    attempts = [
        ((run_dir,), {"updates": u, "merge": merge, "seq_increment": True}),
        ((run_dir,), {"updates": u, "merge": merge}),
        ((run_dir,), {"updates": u}),
        ((run_dir, u), {"merge": merge, "seq_increment": True}),
        ((run_dir, u), {"merge": merge}),
        ((run_dir, u), {}),
        ((run_dir,), {}),
    ]
    last_exc: Exception | None = None
    for args, kwargs in attempts:
        try:
            return write_controls(*args, **kwargs)
        except TypeError as exc:
            last_exc = exc
            continue
        except Exception as exc:
            last_exc = exc
            break
    if last_exc is not None:
        raise last_exc
    return write_controls(run_dir, u)


def set_controls(control_path: Path | str, **updates: Any) -> Any:
    """
    Write control.json updates, forcing snapshot to be a string token when provided.
    """
    path = Path(control_path).expanduser()
    run_dir = path if path.name != "control.json" else path.parent

    if "snapshot" in updates and updates["snapshot"] is not None:
        updates["snapshot"] = str(updates["snapshot"])

    return _write_controls_compat(run_dir, updates, merge=True)


def request_snapshot(*, control_path: Path | str, token: str) -> Any:
    """
    Convenience wrapper used by QC tests; writes snapshot string token.
    """
    return set_controls(control_path=control_path, snapshot=str(token))
