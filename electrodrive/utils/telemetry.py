import csv
import logging
import math
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional


class TelemetryWriter:
    """
    Thread-safe writer for structured telemetry (e.g., solver traces).

    Features:
    - Atomic initialization with header.
    - Row cap via EDE_SOLVER_TRACE_MAX_ROWS (data rows, excludes header).
    - Best-effort semantics: never raises to caller.
    - NaN/Inf floats are serialized as strings to maintain CSV validity.
    """

    def __init__(self, path: Path, headers: List[str]):
        self.path = path
        self.headers = headers
        self._lock = threading.Lock()
        self._initialized = False
        self._rows = 0
        self.logger = logging.getLogger("ede.telemetry")
        max_rows_env = os.environ.get("EDE_SOLVER_TRACE_MAX_ROWS")
        try:
            self._max_rows = int(max_rows_env) if max_rows_env else None
        except Exception:  # pragma: no cover
            self._max_rows = None

    def _initialize(self) -> None:
        if self._initialized:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(self.headers)
                try:
                    f.flush()
                    os.fsync(f.fileno())
                except Exception:
                    pass
            self._initialized = True
            self._rows = 0
            self.logger.info("Telemetry file initialized: %s", self.path)
        except Exception as exc:  # pragma: no cover
            self.logger.error(
                "Failed to initialize telemetry file %s: %s",
                self.path,
                exc,
            )
            self._initialized = True

    def append_row(self, row: Dict[str, Any]) -> None:
        with self._lock:
            if not self._initialized:
                self._initialize()
            if not self.path.exists():
                return
            if self._max_rows and self._rows >= self._max_rows:
                return
            try:
                data: List[Any] = []
                for header in self.headers:
                    val = row.get(header)
                    if isinstance(val, float) and not math.isfinite(val):
                        val = str(val)
                    data.append(val)
                exists = self.path.exists()
                with self.path.open("a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    if not exists:
                        writer.writerow(self.headers)
                    writer.writerow(data)
                self._rows += 1
            except Exception:
                return
