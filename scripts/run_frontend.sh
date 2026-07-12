#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Could not find ${PYTHON_BIN}. Create the environment with: python3 -m venv .venv && source .venv/bin/activate && python -m pip install -r requirements.txt" >&2
  exit 1
fi

cd "${ROOT_DIR}"
"${PYTHON_BIN}" -m streamlit run app/ui/streamlit_app.py
