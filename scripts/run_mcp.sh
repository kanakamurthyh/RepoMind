#!/usr/bin/env bash
# Start RepoMind as an MCP server over stdio (for Claude Code, Cursor, and other MCP clients).
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
exec "${ROOT_DIR}/.venv/bin/python" -m app.mcp_server
