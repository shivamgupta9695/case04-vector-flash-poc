#!/usr/bin/env bash
# Case 04 Vector Search POC — Shell Runner
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="./.venv/Scripts/python.exe"
if [ ! -f "${PYTHON_BIN}" ]; then
    PYTHON_BIN="python3"
fi

"${PYTHON_BIN}" run_poc.py "$@"
