#!/usr/bin/env python
"""
flatten_repo.py

Walks the repository and writes a single text file containing:

  - All developer-relevant file paths (relative to repo root)
  - The full contents of each included file

Each file is delimited in the output with BEGIN/END markers.

Usage:
    python flatten_repo.py
    python flatten_repo.py --root PATH/TO/REPO --output repo_flattened.txt
"""

import argparse
import os
from pathlib import Path
from typing import Iterable, List

# Directories we typically don't care about
EXCLUDED_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
    ".idea",
    ".vscode",
    ".vs",
    "__pycache__",
    ".ipynb_checkpoints",
    ".DS_Store",
    ".cache",
    "build",
    "dist",
    "site-packages",
    ".venv",
    "venv",
    "env",
    ".env",
    ".eggs",
    "node_modules",  # keep if you really want node_modules contents
}

# File extensions that are usually NOT edited by devs
EXCLUDED_FILE_EXTS = {
    ".pyc",
    ".pyo",
    ".pyd",
    ".so",
    ".dll",
    ".dylib",
    ".o",
    ".obj",
    ".a",
    ".lib",
    ".npy",
    ".npz",
    ".pkl",
    ".h5",
    ".hdf5",
    ".pt",
    ".ckpt",
    ".log",
    ".tmp",
}

# File extensions that developers usually *do* work on
INCLUDED_FILE_EXTS = {
    # Python / CUDA / C / C++
    ".py",
    ".pyi",
    ".pyx",
    ".pxd",
    ".c",
    ".cpp",
    ".cc",
    ".cxx",
    ".h",
    ".hpp",
    ".hh",
    ".hxx",
    ".cu",
    ".cuh",
    # Build / config
    ".cmake",
    ".cfg",
    ".ini",
    ".toml",
    ".json",
    ".yml",
    ".yaml",
    ".xml",
    ".conf",
    # Shell / scripting
    ".sh",
    ".bash",
    ".ps1",
    ".bat",
    ".cmd",
    # Docs
    ".md",
    ".rst",
    ".txt",
    ".tex",
    # Other "texty" project files
    ".csv",
    ".tsv",
}

# File *names* that are important even if they have no extension
IMPORTANT_FILENAMES = {
    "Makefile",
    "CMakeLists.txt",
    "Dockerfile",
    "LICENSE",
    "LICENSE.txt",
    "README",
    "README.md",
    "README.rst",
    ".gitignore",
    ".gitattributes",
    "pyproject.toml",
    "requirements.txt",
    "environment.yml",
    "Pipfile",
    "Pipfile.lock",
    "setup.py",
    "setup.cfg",
    "manage.py",
}


def should_skip_dir(dir_name: str) -> bool:
    """Return True if this directory name should be skipped entirely."""
    return dir_name in EXCLUDED_DIR_NAMES


def is_developer_file(path: Path) -> bool:
    """
    Decide whether a file is something developers are likely to work on.

    Heuristic: include if
      - extension in INCLUDED_FILE_EXTS, OR
      - basename in IMPORTANT_FILENAMES
    and NOT in EXCLUDED_FILE_EXTS.
    """
    if path.name in IMPORTANT_FILENAMES:
        return True

    ext = path.suffix.lower()

    if ext in EXCLUDED_FILE_EXTS:
        return False

    if ext in INCLUDED_FILE_EXTS:
        return True

    # By default, ignore unknown binary-ish extensions.
    return False


def collect_files(root: Path) -> List[Path]:
    """Walk the tree and collect all developer-relevant files."""
    files: List[Path] = []

    for dirpath, dirnames, filenames in os.walk(root):
        # Mutate dirnames in-place to skip excluded dirs
        dirnames[:] = [d for d in dirnames if not should_skip_dir(d)]

        dir_path = Path(dirpath)
        for fname in filenames:
            fpath = dir_path / fname
            if is_developer_file(fpath):
                files.append(fpath.relative_to(root))

    # Sort for deterministic output
    files.sort()
    return files


def write_files_with_contents(root: Path, paths: Iterable[Path], output_path: Path) -> None:
    """
    Write all files and their contents to a single text file.

    Each file is wrapped as:

        ===== BEGIN FILE: relative/path =====
        <contents>
        ===== END FILE: relative/path =====
    """
    with output_path.open("w", encoding="utf-8") as out:
        out.write("# Flattened repository file listing\n")
        out.write(f"# Root: {root}\n\n")

        for rel_path in paths:
            abs_path = root / rel_path
            out.write("=" * 80 + "\n")
            out.write(f"===== BEGIN FILE: {rel_path} =====\n")
            out.write("=" * 80 + "\n\n")

            try:
                with abs_path.open("r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        out.write(line)
            except Exception as e:
                # If we can't read the file, note the error and continue
                out.write(f"\n<<< ERROR READING FILE: {e} >>>\n")

            # Ensure there's a trailing newline, then write the end marker
            out.write("\n")
            out.write("=" * 80 + "\n")
            out.write(f"===== END FILE: {rel_path} =====\n")
            out.write("=" * 80 + "\n\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate a .txt file containing developer-relevant files and their contents."
        )
    )
    parser.add_argument(
        "--root",
        type=str,
        default=".",
        help="Root of the repository (default: current directory).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="flattened_repo_with_contents.txt",
        help="Output text file (default: flattened_repo_with_contents.txt).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.root).resolve()
    output_path = Path(args.output).resolve()

    print(f"[flatten_repo] Scanning root: {root}")
    files = collect_files(root)
    print(f"[flatten_repo] Found {len(files)} developer-relevant files.")
    print(f"[flatten_repo] Writing contents to: {output_path}")

    write_files_with_contents(root, files, output_path)

    print(f"[flatten_repo] Done.")


if __name__ == "__main__":
    main()
