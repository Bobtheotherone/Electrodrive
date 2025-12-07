#!/usr/bin/env python3
# Flatten a repo's Python sources into one text file, safely.
#
# Safety features:
# - Excludes its OWN output path
# - Per-file and total byte caps (configurable)
# - Atomic writes (temp -> os.replace) to avoid half-written or redirected outputs
# - Conservative default excludes (venv, caches, logs, run_out, etc.)
# - Case-insensitive directory filtering
# - Optional extra glob excludes (--exclude)
# - Top-N largest file report + TSV size report
# - Math Focus Mode (-m/--math) to isolate numerical kernels
# - Core-only default (packages/tests/configs); use --mode all for full crawl

from __future__ import annotations
import argparse
import ast
import fnmatch
import io
import os
import tempfile
import tokenize
from datetime import datetime
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Directories to skip anywhere in the tree (case-insensitive match on any part)
EXCLUDED_DIRS_ANYWHERE = {
    # Python / Environment
    "__pycache__", ".venv", "venv", "env", "virtualenv",
    "site-packages", "develop-eggs", "eggs", ".eggs",
    "parts", "sdist", "wheels", "pip-wheel-metadata",

    # Version Control / IDE
    ".git", ".hg", ".svn", ".gitattributes", ".gitignore",
    ".vscode", ".idea", ".vs", ".settings", ".project",

    # Testing / Quality / Caches
    ".mypy_cache", ".pytest_cache", ".ruff_cache", ".tox", ".hypothesis",
    "htmlcov",

    # Node / JS
    "node_modules",
}

# Directories to skip when they appear at the repository root in core mode.
EXCLUDED_DIRS_ROOT_ONLY = {
    # Non-core / ops / artifacts
    "tools", "scripts", "experiments", "runs", "run_out",
    "outputs", "output", "staging", "upload", "notebooks",
    "docs", "benchmarks", "dist", "build",
    # Data-ish
    "data", "datasets", "downloads", "models", "checkpoints",
    "media", "images", "static", "public", "assets",
    # Logs / scratch
    "log", "logs", "logging", "tmp", "temp", "local", "scratch",
    # Env mirrors (kept here to catch root-only names)
    ".venv", "venv", "env", "virtualenv", "node_modules",
}

# File patterns to skip even if they are inside a valid directory
DEFAULT_EXCLUDED_GLOBS = {
    # Compiled / Binary
    "*.pyc", "*.pyo", "*.pyd", "*.so", "*.dll", "*.dylib", "*.exe",
    
    # Package metadata
    "*.egg-info",
    
    # Cache files
    ".coverage", ".toc",
    
    # Obvious large generated blobs (can be toggled separately)
    "*_pb2.py", "*_pb2.pyi", "*_pb2_grpc.py",
    
    # Misc
    "*.log", "*.tmp", "*.bak",
}

# Globs that are very likely to be huge or generated; used when --skip-huge-generated is on.
DEFAULT_HUGE_GLOBS = {
    "*_pb2.py", "*_pb2.pyi", "*_pb2_grpc.py",
    "*schema.generated.py",
    "*_autogen.py",
}

# Globs that define the "Math/Physics Core" of the repo.
# Used when --math is passed to focus debugging on numerical logic.
MATH_FOCUS_GLOBS = {
    # FMM implementation (Multipoles, Tree, Interaction Lists)
    "*electrodrive/fmm3d/*.py",
    # Core BEM / Physics / Certify (Kernels, Solvers, Quadrature)
    "*electrodrive/core/*.py",
    # Analytic solutions for validation
    "*electrodrive/images/*.py",
    # Learning models (PINN physics)
    "*electrodrive/learn/models/*.py",
    # Physical constants (K_E, EPS_0)
    "*electrodrive/utils/config.py",
    # Tests (to see assertions/tolerances)
    "*tests/*.py",
    "*electrodrive/fmm3d/tests/*.py"
}

# ---------------------------------------------------------------------------
# Repo-specific extra excludes
# ---------------------------------------------------------------------------

# Directories that are snapshots / sandboxes / generated outputs and
# aren't part of the active development surface.
PROJECT_EXCLUDED_DIRS_ROOT_ONLY = {
    # Archived snapshots of real code (duplicates)
    "staging",
    # FMM sandbox experiments (not part of production surface)
    "_agent_sandbox",
    # Log / agent output trees we never want to flatten
    ".logs_fmm",
    "_agent_outputs",
}

# Path globs for repo-specific excludes (used at file-level)
PROJECT_EXCLUDED_GLOBS = {
    # Snapshot / sandbox trees
    "staging/*",
    "staging/**",
    "*/_agent_sandbox/*",
    "*/_agent_sandbox/**",

    # Generated agent/BEM intercept outputs
    "experiments/_agent_outputs/*",
    "experiments/_agent_outputs/**",

    # Log folders at root
    ".logs_fmm/*",
    ".logs_fmm/**",
}

# Core root-level files that developers need even though they are not .py.
ESSENTIAL_ROOT_GLOBS = {
    "pyproject.toml", "setup.cfg", "setup.py",
    "requirements*.txt", "pipfile", "poetry.lock",
    "readme*", "license*", "makefile", "dockerfile",
    ".pre-commit-config.yaml",
    "ruff.toml", "mypy.ini", "pytest.ini", "tox.ini", ".editorconfig",
}

# JSON configs allowed at root in core mode (for specs/configs only).
JSON_CONFIG_ROOT_DIRS = {"config", "configs", "configurations", "spec", "specs", "specifications"}
JSON_CONFIG_FILE_KEYWORDS = ("config", "spec")


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def read_text_safely(p: Path, max_bytes: int | None) -> str:
    """Read text with sensible fallbacks; normalize newlines; enforce per-file cap."""
    # Enforce per-file cap
    try:
        size = p.stat().st_size
        if max_bytes is not None and size > max_bytes:
            return f"<<TRUNCATED: {p.name} {size} bytes exceeds {max_bytes} cap>>\n"
    except Exception:
        pass

    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            txt = p.read_text(encoding=enc, errors="strict")
            return txt.replace("\r\n", "\n").replace("\r", "\n")
        except Exception:
            continue
    # Last resort: binary read then decode ignoring errors
    try:
        raw = p.read_bytes()
        return raw.decode("utf-8", errors="ignore").replace("\r\n", "\n").replace("\r", "\n")
    except Exception:
        return f"<<ERROR: failed to read {p.name}>>\n"


def strip_docstrings_and_comments(src: str) -> str:
    """
    Very basic "strip comments" that:
      - Parses the file with `ast` to get docstring ranges,
      - Removes them from the source,
      - Then strips hash comments using tokenize.
    Falls back to returning the original src if parsing fails.
    """
    try:
        # 1) Use tokenize to strip hash comments
        src_no_comments = _strip_hash_comments(src)
        # 2) Use AST to strip docstrings
        return _strip_docstrings(src_no_comments)
    except Exception:
        # If anything goes wrong, just return the original text
        return src


def _strip_hash_comments(src: str) -> str:
    """Strip comments that start with #, preserving code layout as best we can."""
    out = io.StringIO()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except tokenize.TokenError:
        # If tokenization fails, just return the original
        return src

    for tok_type, tok_string, start, end, line in tokens:
        if tok_type == tokenize.COMMENT:
            # Replace comments with nothing but keep the newline
            continue
        out.write(tok_string)
    return out.getvalue()


def _strip_docstrings(src: str) -> str:
    """
    Remove top-level, class-level, and function-level docstrings from the source
    while leaving the structure intact. This is a best-effort approach and not
    a perfect code formatter.
    """
    try:
        mod = ast.parse(src)
    except SyntaxError:
        return src

    # We will blank out (not reflow) docstrings by line/column
    # then reconstruct.
    lines = src.splitlines(keepends=True)
    to_blank: list[tuple[int, int, int, int]] = []  # (start_line, start_col, end_line, end_col)

    def visit_node(node: ast.AST):
        doc = ast.get_docstring(node, clean=False)
        if not doc:
            return
        # The docstring is usually the first statement of the body
        if not getattr(node, "body", None):
            return
        first_stmt = node.body[0]
        if not isinstance(first_stmt, ast.Expr) or not isinstance(getattr(first_stmt, "value", None), ast.Constant):
            return
        val = first_stmt.value
        if not isinstance(val, ast.Constant) or not isinstance(val.value, str):
            return
        # Record the location of this docstring
        sl, sc = first_stmt.lineno - 1, first_stmt.col_offset
        el, ec = getattr(first_stmt, "end_lineno", sl + 1) - 1, getattr(first_stmt, "end_col_offset", sc)
        to_blank.append((sl, sc, el, ec))

    # Visit module, classes, and functions
    for node in ast.walk(mod):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            visit_node(node)

    # Blank out each docstring region
    for sl, sc, el, ec in sorted(to_blank):
        if sl == el:
            # Single-line docstring
            line = lines[sl]
            lines[sl] = line[:sc] + (" " * (ec - sc)) + line[ec:]
        else:
            # Multi-line docstring
            first = lines[sl]
            lines[sl] = first[:sc] + (" " * (len(first) - sc))
            for i in range(sl + 1, el):
                lines[i] = " " * len(lines[i])
            last = lines[el]
            lines[el] = (" " * ec) + last[ec:]

    src_no_docstrings = "".join(lines)
    # Finally, strip hash comments from the docstring-free source
    return _strip_hash_comments(src_no_docstrings)


def path_matches_any_glob(path_str: str, patterns: set[str]) -> bool:
    return any(fnmatch.fnmatch(path_str, pat) for pat in patterns)


def _is_relative_to(path: Path, ancestor: Path) -> bool:
    """Python 3.8 compatible Path.is_relative_to."""
    try:
        path.relative_to(ancestor)
        return True
    except ValueError:
        return False


def discover_package_roots(root: Path) -> set[Path]:
    """Find first-party package roots at repo root or inside src/ (contains __init__.py)."""
    package_roots: set[Path] = set()
    for child in root.iterdir():
        if child.is_dir() and (child / "__init__.py").is_file():
            package_roots.add(child)
    src = root / "src"
    if src.is_dir():
        for child in src.iterdir():
            if child.is_dir() and (child / "__init__.py").is_file():
                package_roots.add(child)
    return package_roots


def is_test_path(path: Path, root: Path) -> bool:
    return any(part.lower() == "tests" for part in path.relative_to(root).parts)


def is_in_package_root(path: Path, package_roots: set[Path]) -> bool:
    return any(_is_relative_to(path, pkg) for pkg in package_roots)


def is_on_allowed_path(path: Path, allowed_prefixes: set[Path], root: Path) -> bool:
    """Allow both ancestors and descendants to keep traversal around package roots."""
    return any(
        (allowed != root and (_is_relative_to(path, allowed) or _is_relative_to(allowed, path)))
        or (allowed == root and path == root)
        for allowed in allowed_prefixes
    )


def build_allowed_prefixes(
    root: Path,
    package_roots: set[Path],
    include_tests: bool,
    include_json: bool,
) -> set[Path]:
    raw: set[Path] = set(package_roots)
    raw.add(root)
    if include_tests:
        raw.add(root / "tests")
    workflows = root / ".github" / "workflows"
    if workflows.exists() or (root / ".github").exists():
        raw.add(workflows)
    if include_json:
        for name in JSON_CONFIG_ROOT_DIRS:
            raw.add(root / name)

    expanded: set[Path] = set()
    for p in raw:
        cur = p
        while True:
            expanded.add(cur)
            if cur == root:
                break
            cur = cur.parent
    return expanded


def should_prune_dir(
    path: Path,
    root: Path,
    allowed_prefixes: set[Path],
    mode: str,
    include_tests: bool,
    package_roots: set[Path],
) -> bool:
    rel_parts = path.relative_to(root).parts
    lower_parts = [p.lower() for p in rel_parts]
    root_name = lower_parts[0] if lower_parts else ""
    is_pkg_root = path in package_roots

    if any(part in EXCLUDED_DIRS_ANYWHERE for part in lower_parts):
        return True
    if root_name in PROJECT_EXCLUDED_DIRS_ROOT_ONLY and not is_pkg_root:
        return True

    if mode == "core":
        if root_name in EXCLUDED_DIRS_ROOT_ONLY and not is_pkg_root:
            return True
        if not include_tests and path.name.lower() == "tests":
            return True
        if not is_on_allowed_path(path, allowed_prefixes, root):
            return True
    else:
        if not include_tests and path.name.lower() == "tests":
            return True
    return False


def should_exclude_by_glob(rel_posix: str, extra_excludes: list[str]) -> bool:
    # Apply built-in and project-specific glob excludes first.
    builtin_globs = DEFAULT_EXCLUDED_GLOBS | PROJECT_EXCLUDED_GLOBS
    if path_matches_any_glob(rel_posix, builtin_globs):
        return True
    if extra_excludes:
        return path_matches_any_glob(rel_posix, set(extra_excludes))
    return False


def is_core_root_file(p: Path, root: Path) -> bool:
    if p.parent != root:
        return False
    lower = p.name.lower()
    return any(fnmatch.fnmatch(lower, pat) for pat in ESSENTIAL_ROOT_GLOBS)


def is_github_workflow_file(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root / ".github" / "workflows")
    except ValueError:
        return False
    return p.suffix.lower() in {".yml", ".yaml"}


def should_include_json_core(
    path: Path,
    root: Path,
    package_roots: set[Path],
    include_tests: bool,
    json_config_dirs: list[Path],
) -> bool:
    if is_in_package_root(path, package_roots):
        return True
    if include_tests and is_test_path(path, root):
        return True
    if path.parent == root:
        name_lower = path.name.lower()
        return any(key in name_lower for key in JSON_CONFIG_FILE_KEYWORDS)
    return any(_is_relative_to(path, cfg_dir) for cfg_dir in json_config_dirs)


def collect_files(
    root: Path,
    extra_excludes: list[str],
    output_path_to_exclude: Optional[Path],
    skip_huge_globs: bool,
    mode: str,
    include_tests: bool,
    include_json: bool,
    math_focus: bool = False,
) -> list[Path]:
    """Select files to flatten based on mode (core/all), test toggle, and JSON inclusion."""
    files: list[Path] = []
    exclude_real = None
    if output_path_to_exclude:
        try:
            exclude_real = output_path_to_exclude.resolve()
        except Exception:
            exclude_real = None

    huge_patterns = set(DEFAULT_HUGE_GLOBS) if skip_huge_globs else set()
    package_roots = discover_package_roots(root)
    json_config_dirs = [root / name for name in JSON_CONFIG_ROOT_DIRS if (root / name).exists()]
    allowed_prefixes = build_allowed_prefixes(root, package_roots, include_tests, include_json)

    for dirpath, dirnames, filenames in os.walk(root):
        dp = Path(dirpath)
        dirnames[:] = [
            d for d in dirnames
            if not should_prune_dir(dp / d, root, allowed_prefixes, mode, include_tests, package_roots)
            and not (dp / d).name.startswith(".DS_Store")
        ]

        for name in filenames:
            p = Path(dirpath) / name

            # Don't include the file we are writing to
            if exclude_real:
                try:
                    if p.resolve() == exclude_real:
                        continue
                except Exception:
                    pass

            rel = p.relative_to(root).as_posix()
            if should_exclude_by_glob(rel, extra_excludes):
                continue
            if huge_patterns and path_matches_any_glob(rel, huge_patterns):
                continue

            if not include_tests and is_test_path(p, root):
                continue

            suffix = p.suffix.lower()
            include = False

            if suffix == ".py":
                if mode == "all":
                    include = True
                else:
                    include = is_in_package_root(p, package_roots) or is_test_path(p, root)
            elif suffix == ".json" and include_json:
                if mode == "all":
                    include = True
                else:
                    include = should_include_json_core(p, root, package_roots, include_tests, json_config_dirs)
            else:
                include = is_core_root_file(p, root) or is_github_workflow_file(p, root)

            if include and math_focus and suffix == ".py":
                if not path_matches_any_glob(rel, MATH_FOCUS_GLOBS):
                    continue

            if include:
                files.append(p)

    files.sort(key=lambda x: x.relative_to(root).as_posix().lower())
    return files


def make_reports(files: list[Path], root: Path, tsv_path: Path, top: int = 20) -> None:
    """
    Generate:
      - a TSV of file sizes
      - a top-N largest files printout
    """
    rows: list[tuple[int, str]] = []
    for p in files:
        try:
            size = p.stat().st_size
        except Exception:
            size = 0
        rows.append((size, p.relative_to(root).as_posix()))
    rows.sort(reverse=True)
    # TSV
    with tsv_path.open("w", encoding="utf-8", newline="") as f:
        f.write("size_bytes\tpath\n")
        for sz, rel in rows:
            f.write(f"{sz}\t{rel}\n")
    # Top-N to stdout
    print(f"Top {min(top, len(rows))} largest included files under {root}:")
    for sz, rel in rows[:top]:
        mb = sz / 1024 / 1024
        print(f"  {mb:8.1f}MB  {rel}")


def make_flat_document(
    root: Path,
    files: list[Path],
    strip_comments: bool,
    per_file_cap: int | None,
    total_cap: int | None,
    mode: str,
    include_tests: bool,
    include_json: bool,
) -> str:
    """
    Build the flattened document as a single string.
    Insert markers between files and enforce per-file and total caps.
    """
    chunks: list[str] = []
    total = 0

    header = (
        f"# Flattened repo view\n"
        f"# Root: {root}\n"
        f"# Generated: {datetime.now().isoformat(timespec='seconds')}\n"
        f"# Files included: {len(files)}\n"
        f"# Strip comments: {strip_comments}\n"
        f"# Mode: {mode}\n"
        f"# Include tests: {include_tests}\n"
        f"# Include JSON: {include_json}\n"
        f"# Per-file cap: {per_file_cap or 'none'} bytes\n"
        f"# Total cap: {total_cap or 'none'} bytes\n"
        f"#\n\n"
    )
    chunks.append(header)
    total += len(header)

    for i, p in enumerate(files, 1):
        rel = p.relative_to(root).as_posix()
        banner = (
            "\n"
            f"# =====================================================================\n"
            f"# File {i}/{len(files)}: {rel}\n"
            f"# =====================================================================\n\n"
        )
        if total_cap is not None and total + len(banner) >= total_cap:
            chunks.append("<<GLOBAL TRUNCATION: total cap reached before adding banner>>\n")
            break
        chunks.append(banner)
        total += len(banner)

        try:
            txt = read_text_safely(p, per_file_cap)
        except Exception as e:
            txt = f"<<ERROR reading file {rel}: {e}>>\n"

        if strip_comments:
            txt = strip_docstrings_and_comments(txt)

        if total_cap is not None and total + len(txt) >= total_cap:
            # Truncate this chunk to fit the remaining space
            remaining = max(total_cap - total, 0)
            truncated_txt = txt[:remaining]
            chunks.append(truncated_txt)
            chunks.append("\n<<GLOBAL TRUNCATION: total cap reached mid-file>>\n")
            break

        chunks.append(txt)
        total += len(txt)

    return "".join(chunks)


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """Write text to a temp file then atomically move into place."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding=encoding, newline="") as f:
        f.write(text)
    os.replace(tmp, path)


def run_self_test() -> None:
    """Basic sanity check for core/all selection."""
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "pyproject.toml").write_text("[build-system]\n", encoding="utf-8")
        (root / "README.md").write_text("# README\n", encoding="utf-8")

        pkg = root / "electrodrive"
        pkg.mkdir()
        (pkg / "__init__.py").write_text("pass\n", encoding="utf-8")
        (pkg / "core.py").write_text("x = 1\n", encoding="utf-8")

        src_pkg = root / "src" / "pkg"
        src_pkg.mkdir(parents=True)
        (src_pkg / "__init__.py").write_text("y = 2\n", encoding="utf-8")

        tests = root / "tests"
        tests.mkdir()
        (tests / "test_core.py").write_text("def test_core():\n    assert True\n", encoding="utf-8")

        tools = root / "tools"
        tools.mkdir()
        (tools / "scratch.py").write_text("print('tool')\n", encoding="utf-8")

        runs = root / "runs"
        runs.mkdir()
        (runs / "debug.py").write_text("print('run')\n", encoding="utf-8")

        files_core = collect_files(
            root,
            [],
            output_path_to_exclude=None,
            skip_huge_globs=True,
            mode="core",
            include_tests=True,
            include_json=False,
            math_focus=False,
        )
        rel_core = {p.relative_to(root).as_posix() for p in files_core}
        assert "electrodrive/__init__.py" in rel_core
        assert "electrodrive/core.py" in rel_core
        assert "src/pkg/__init__.py" in rel_core
        assert "tests/test_core.py" in rel_core
        assert "pyproject.toml" in rel_core
        assert "README.md" in rel_core
        assert "tools/scratch.py" not in rel_core
        assert "runs/debug.py" not in rel_core

        files_no_tests = collect_files(
            root,
            [],
            output_path_to_exclude=None,
            skip_huge_globs=True,
            mode="core",
            include_tests=False,
            include_json=False,
            math_focus=False,
        )
        rel_no_tests = {p.relative_to(root).as_posix() for p in files_no_tests}
        assert "tests/test_core.py" not in rel_no_tests

        files_all = collect_files(
            root,
            [],
            output_path_to_exclude=None,
            skip_huge_globs=True,
            mode="all",
            include_tests=True,
            include_json=False,
            math_focus=False,
        )
        rel_all = {p.relative_to(root).as_posix() for p in files_all}
        assert "tools/scratch.py" in rel_all
        assert "runs/debug.py" in rel_all

    print("[self-test] passed")


def main():
    ap = argparse.ArgumentParser(
        description="Flatten a repository's core Python files into one document (safely)."
    )
    ap.add_argument("--root", type=Path, default=Path("."), help="Repository root (default: .)")
    ap.add_argument(
        "--output",
        type=Path,
        default=Path("repo_flatten.txt"),
        help="Output file or '-' for stdout (default: repo_flatten.txt)",
    )
    ap.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Extra glob to exclude (repeatable). Example: --exclude '*/data/*'",
    )
    ap.add_argument(
        "--top",
        type=int,
        default=20,
        help="Show top-N largest files (default: 20)",
    )
    ap.add_argument(
        "--per-file-cap",
        type=int,
        default=256_000,
        help="Per-file byte cap (default: 256k). Use 0 for no cap.",
    )
    ap.add_argument(
        "--total-cap",
        type=int,
        default=30_000_000,
        help="Total-output cap (default: 30MB). Use 0 for no cap.",
    )
    ap.add_argument(
        "--skip-huge-generated",
        dest="skip_huge_generated",
        action="store_true",
        default=True,
        help="Skip likely huge generated files like *_pb2.py (default: on).",
    )
    ap.add_argument(
        "--nc",
        "--no-comments",
        dest="no_comments",
        action="store_true",
        help="Strip # comments and docstrings from Python code before flattening.",
    )
    ap.add_argument(
        "--include-json",
        dest="include_json",
        action="store_true",
        help="Also include .json files (specs/configs). Core mode limits this to packages/tests/configs.",
    )
    ap.add_argument(
        "--mode",
        choices=["core", "all"],
        default="core",
        help="core=packages/tests/configs only (default), all=include every file allowed by excludes.",
    )
    ap.add_argument(
        "--no-tests",
        dest="no_tests",
        action="store_true",
        help="Exclude tests from the flattened output.",
    )
    ap.add_argument(
        "--self-test",
        action="store_true",
        help="Run a built-in sanity check and exit.",
    )
    
    ap.add_argument(
        "-m",
        "--math",
        dest="math_focus",
        action="store_true",
        help="MATH FOCUS: Only include files dealing with core math/physics/kernels/configs.",
    )
    args = ap.parse_args()

    if args.self_test:
        run_self_test()
        return

    if str(args.output).lower().endswith(".py"):
        raise SystemExit("Refusing to write a .py file. Use a .txt/.md extension for --output.")

    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"Root not found: {root}")

    output_file_to_exclude: Optional[Path] = None
    if str(args.output) != "-":
        output_file_to_exclude = Path(args.output).resolve()

    include_tests = not args.no_tests

    files = collect_files(
        root,
        args.exclude,
        output_file_to_exclude,
        skip_huge_globs=args.skip_huge_generated,
        mode=args.mode,
        include_tests=include_tests,
        math_focus=args.math_focus,
        include_json=args.include_json,
    )
    
    if args.math_focus:
        print(
            f"[info] MATH FOCUS mode enabled. {len(files)} files selected "
            f"(mode={args.mode}, tests={'on' if include_tests else 'off'})."
        )
    else:
        print(
            f"[info] Found {len(files)} files under {root} "
            f"(mode={args.mode}, tests={'on' if include_tests else 'off'})."
        )

    # Make the reports
    tsv = args.output.with_suffix(".sizes.tsv") if str(args.output) != "-" else root / "repo_flatten_sizes.tsv"
    make_reports(files, root, tsv, top=args.top)

    per_file_cap = None if args.per_file_cap == 0 else args.per_file_cap
    total_cap = None if args.total_cap == 0 else args.total_cap

    doc = make_flat_document(
        root=root,
        files=files,
        strip_comments=args.no_comments,
        per_file_cap=per_file_cap,
        total_cap=total_cap,
        mode=args.mode,
        include_tests=include_tests,
        include_json=args.include_json,
    )

    if str(args.output) == "-":
        print(doc, end="")
    else:
        atomic_write_text(args.output, doc, encoding="utf-8")
        print(f"Wrote {args.output} ({len(files)} files).")
        print(f"Wrote size report: {tsv}")
        if total_cap is not None and len(doc) >= total_cap:
            print("[warn] Output was truncated by total cap.")
        import hashlib

        h = hashlib.sha256(doc.encode("utf-8")).hexdigest()
        print(f"SHA256({args.output.name}) = {h}")


if __name__ == "__main__":
    main()
