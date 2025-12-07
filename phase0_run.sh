#!/usr/bin/env bash
set -euo pipefail

# Determine repo root as the directory containing this script.
SCRIPT_PATH="$0"
SCRIPT_DIR="$(cd "$(dirname "$SCRIPT_PATH")" && pwd)"

cd "$SCRIPT_DIR"

# Use the default python on PATH; assumes .venv is already activated if desired.
python scripts/phase0_baseline.py "$@"
