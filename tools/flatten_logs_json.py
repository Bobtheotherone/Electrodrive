#!/usr/bin/env python3
"""
Flatten logs & JSON into a single text file.

- Targets: *.json, *.jsonl, *.ndjson, *.log  (extend with --ext)
- Safety: excludes venv/node_modules/.git by default; per-file & total caps; atomic writes
- Reports: TOC in the flat file + optional TSV of sizes
"""

from __future__ import annotations
import argparse, fnmatch, io, json, os, sys
from pathlib import Path
from datetime import datetime
from typing import Iterable, List, Optional, Tuple

# --- defaults (note: unlike the .py flattener, we DO NOT exclude runs/logs) ---
DEFAULT_EXCLUDED_DIRS = {
    "__pycache__", ".git", ".hg", ".svn",
    ".mypy_cache", ".pytest_cache", ".ruff_cache",
    ".vscode", ".idea", ".tox",
    "build", "dist",
    ".venv", "venv", "env",
    "site-packages", "node_modules",
}
DEFAULT_EXCLUDED_GLOBS = {
    "*/__pycache__/*",
    "*/.mypy_cache/*",
    "*/.pytest_cache/*",
    "*/.ruff_cache/*",
    "*.pyc", "*.pyo", "*.pyd",
}
DEFAULT_TARGET_EXTS = {".json", ".jsonl", ".ndjson", ".log"}

# --- helpers ---

def _is_ignored_dir(path: Path) -> bool:
    return any(part in DEFAULT_EXCLUDED_DIRS for part in path.parts)

def _match_any(path_str: str, patterns: Iterable[str]) -> bool:
    return any(fnmatch.fnmatch(path_str, pat) for pat in patterns)

def _read_text_safely(p: Path, max_bytes: Optional[int]) -> str:
    try:
        size = p.stat().st_size
        if max_bytes is not None and size > max_bytes:
            return f"<<TRUNCATED: {p.name} {size} bytes exceeds {max_bytes} cap>>\n"
    except Exception:
        pass
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            s = p.read_text(encoding=enc, errors="strict")
            return s.replace("\r\n", "\n").replace("\r", "\n")
        except Exception:
            continue
    try:
        raw = p.read_bytes()
        return raw.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
    except Exception:
        return f"<<ERROR: failed to read {p.name}>>\n"

def _render_file(p: Path, per_file_cap: Optional[int],
                 pretty_json: bool, prettify_jsonl: bool) -> str:
    ext = p.suffix.lower()
    # JSON file → parse & re-dump (stable, pretty) unless user disabled
    if ext == ".json" and pretty_json:
        try:
            obj = json.loads(p.read_text(encoding="utf-8-sig"))
            return json.dumps(obj, indent=2, sort_keys=True) + "\n"
        except Exception:
            return _read_text_safely(p, per_file_cap)
    # JSONL/NDJSON → optionally pretty-print item-per-line
    if ext in (".jsonl", ".ndjson") and prettify_jsonl:
        out_lines: List[str] = []
        try:
            with p.open("r", encoding="utf-8-sig", errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        out_lines.append("")
                        continue
                    try:
                        obj = json.loads(line)
                        out_lines.append(json.dumps(obj, sort_keys=True))
                    except Exception:
                        out_lines.append(line)
            return "\n".join(out_lines) + "\n"
        except Exception:
            return _read_text_safely(p, per_file_cap)
    # Logs or raw text
    return _read_text_safely(p, per_file_cap)

def _atomic_write(path: Path, content: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    os.replace(tmp, path)

def _collect_files(root: Path, include_exts: set[str],
                   extra_excludes: List[str], out_path: Optional[Path]) -> List[Path]:
    result: List[Path] = []
    out_real = None
    if out_path:
        try: out_real = out_path.resolve()
        except Exception: out_real = None

    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        dirnames[:] = [d for d in dirnames if not _is_ignored_dir(dp / d)]
        for name in filenames:
            p = dp / name
            if out_real:
                try:
                    if p.resolve() == out_real:
                        continue
                except Exception:
                    pass
            rel = p.relative_to(root).as_posix()
            if extra_excludes and _match_any(rel, extra_excludes):
                continue
            if p.suffix.lower() in include_exts:
                result.append(p)
    result.sort(key=lambda x: x.relative_to(root).as_posix().lower())
    return result

def _format_toc(rows: List[Tuple[int, str]]) -> str:
    lines = ["TOC", "---"]
    for sz, rel in rows:
        lines.append(f"- {rel} ({sz} bytes)")
    return "\n".join(lines) + "\n"

def _write_tsv(rows: List[Tuple[int, str]], tsv_path: Path) -> None:
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("size_bytes\tpath\n")
        for sz, rel in sorted(rows, reverse=True):
            f.write(f"{sz}\t{rel}\n")

# --- main flatten ---

def make_flat_document(root: Path, files: List[Path], per_file_cap: Optional[int],
                       total_cap: Optional[int], pretty_json: bool, prettify_jsonl: bool) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rows: List[Tuple[int, str]] = []
    total = 0
    parts: List[str] = []
    header = []
    header.append("REPO LOG+JSON FLATTEN")
    header.append(f"Root: {root}")
    header.append(f"Generated: {now}")
    header.append(f"Count: {len(files)}")
    header.append("=" * 72)
    parts.append("\n".join(header) + "\n\n")

    # TOC
    for p in files:
        try: size = p.stat().st_size
        except Exception: size = 0
        rows.append((size, p.relative_to(root).as_posix()))
    parts.append(_format_toc(rows) + "\n" + "=" * 72 + "\n\n")

    # Body
    for p in files:
        try: size = p.stat().st_size
        except Exception: size = 0
        rel = p.relative_to(root).as_posix()
        parts.append(f"===== FILE: {rel} ({size} bytes) =====\n")
        chunk = _render_file(p, per_file_cap, pretty_json, prettify_jsonl)
        total += len(chunk.encode("utf-8", errors="ignore"))
        if total_cap is not None and total > total_cap:
            parts.append("<<TOTAL CAP REACHED; STOPPING HERE>>\n")
            break
        parts.append(chunk)
        if not chunk.endswith("\n"):
            parts.append("\n")
    return "".join(parts)

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Flatten logs & JSON into a single text file.")
    ap.add_argument("--root", type=str, default=".", help="Repo root to scan.")
    ap.add_argument("--out", type=str, default="repo_logs_flatten.txt", help="Output flat file.")
    ap.add_argument("--report-tsv", type=str, default="", help="Optional TSV path for sizes.")
    ap.add_argument("--top", type=int, default=30, help="Print top-N largest files to stdout.")
    ap.add_argument("--exclude", action="append", default=[], help="Extra glob(s) to exclude.")
    ap.add_argument("--ext", action="append", default=[], help="Extra file extensions to include (e.g., .txt).")
    ap.add_argument("--max-file-bytes", type=int, default=5_000_000, help="Cap per file; if exceeded, mark as TRUNCATED.")
    ap.add_argument("--max-total-bytes", type=int, default=50_000_000, help="Overall cap for the flat document.")
    ap.add_argument("--pretty-json", action="store_true", help="Pretty-print *.json with sorted keys.")
    ap.add_argument("--prettify-jsonl", action="store_true", help="Parse each JSONL line and re-dump; falls back on raw lines if parse fails.")
    args = ap.parse_args(argv)

    root = Path(args.root).expanduser().resolve()
    out = Path(args.out).expanduser().resolve()
    include_exts = set(DEFAULT_TARGET_EXTS)
    if args.ext:
        include_exts |= {e if e.startswith(".") else "." + e for e in args.ext}

    files = _collect_files(root, include_exts, args.exclude, out)
    # size report
    rows = []
    for p in files:
        try: sz = p.stat().st_size
        except Exception: sz = 0
        rows.append((sz, p.relative_to(root).as_posix()))
    rows_sorted = sorted(rows, reverse=True)
    print(f"Top {min(args.top, len(rows_sorted))} largest log/json files under {root}:")
    for sz, rel in rows_sorted[: args.top]:
        print(f"  {sz/1024/1024:8.1f}MB  {rel}")

    # build doc and write atomically
    doc = make_flat_document(
        root, files,
        per_file_cap=args.max_file_bytes,
        total_cap=args.max_total_bytes,
        pretty_json=args.pretty_json,
        prettify_jsonl=args.prettify_jsonl,
    )
    _atomic_write(out, doc)

    if args.report_tsv:
        _write_tsv(rows, Path(args.report_tsv).expanduser().resolve())

    return 0

if __name__ == "__main__":
    raise SystemExit(main())
