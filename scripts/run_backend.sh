#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Could not find ${PYTHON_BIN}. Create the environment with: python3 -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt" >&2
  exit 1
fi

cd "${ROOT_DIR}"
if [[ "${RELOAD:-0}" == "1" ]]; then
  "${PYTHON_BIN}" -m uvicorn app.main:app --reload --reload-dir app --host 127.0.0.1 --port 8000
else
  "${PYTHON_BIN}" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
fi
