#!/usr/bin/env bash
# Run ml-intern natively on macOS — no Docker.
# - Backend: FastAPI on port 7860 with full MPS access for any tools that train models
# - Frontend: served from ./static if pre-built, else from Vite dev on :5173
# - Sessions write to ./session_logs/ on host (no volume needed)
# - Agent's "local tools" (bash/write/read/edit) run on this machine — has MPS
#
# Usage:
#   ./run-native.sh           # backend only (faster) — open http://localhost:7860 (uses pre-built frontend)
#   ./run-native.sh dev       # backend + Vite dev frontend with HMR — open http://localhost:5173
#   ./run-native.sh --build   # rebuild frontend then backend
#
# Stop: Ctrl-C in this terminal; both servers exit cleanly.

set -e
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$REPO_ROOT"

# Sanity: deps installed
command -v uv >/dev/null || { echo "uv not installed — run: brew install uv"; exit 1; }
[ -d .venv ] || uv sync

# Build the frontend if asked, or if static/ doesn't exist yet
if [ "$1" = "--build" ] || ([ ! -d static ] && [ "$1" != "dev" ]); then
  echo "→ building frontend (one-time, takes ~30s)…"
  (cd frontend && npm install && npm run build)
fi

# Start backend
echo "→ starting backend on http://localhost:7860"
cd "$REPO_ROOT/backend"
exec uv run uvicorn main:app --host 0.0.0.0 --port 7860 --reload-dir "$REPO_ROOT/backend" --reload-dir "$REPO_ROOT/agent"
