#!/usr/bin/env bash
# ml-intern single-command launcher.
#
# What it does:
#   1. Validates dependencies (uv, node, .env)
#   2. Installs Python + Node deps if missing
#   3. Builds the React frontend if static/ is stale
#   4. Stops any existing native or Docker backend on port 7860
#   5. Starts native backend with MPS access + ML_INTERN_LOCAL_MODE=1
#   6. Tails logs and waits for ready signal
#   7. Opens http://localhost:7860 in your default browser
#
# Stop the server:
#   ./up.sh stop
#
# Force rebuild frontend:
#   ./up.sh --rebuild

set -euo pipefail
cd "$(dirname "$0")"
REPO_ROOT="$(pwd)"
PORT="${ML_INTERN_PORT:-7860}"
LOG="/tmp/ml-intern.log"

GREEN='\033[0;32m'; YELLOW='\033[0;33m'; RED='\033[0;31m'; CYAN='\033[0;36m'; NC='\033[0m'
info()  { echo -e "${CYAN}→${NC} $1"; }
ok()    { echo -e "${GREEN}✓${NC} $1"; }
warn()  { echo -e "${YELLOW}!${NC} $1"; }
fail()  { echo -e "${RED}✗${NC} $1"; exit 1; }

# ── stop subcommand ──────────────────────────────────────────────────
if [[ "${1:-}" == "stop" ]]; then
  info "stopping ml-intern…"
  pids=$(lsof -ti:"$PORT" 2>/dev/null || true)
  [[ -n "$pids" ]] && kill $pids && ok "killed PIDs: $pids" || warn "nothing on :$PORT"
  exit 0
fi

# ── dep checks ───────────────────────────────────────────────────────
command -v uv >/dev/null   || fail "uv not installed. Run: brew install uv"
command -v node >/dev/null || fail "node not installed. Run: brew install node"
[[ -f .env ]]              || fail ".env missing. Copy values from your secret store."

# ── one-time installs ────────────────────────────────────────────────
if [[ ! -d .venv ]]; then
  info "installing python deps (uv sync)…"
  uv sync
fi
if [[ ! -d frontend/node_modules ]]; then
  info "installing frontend deps (npm install)…"
  (cd frontend && npm install)
fi
if ! .venv/bin/python -c "import torch" 2>/dev/null; then
  info "installing torch + torchvision (MPS-capable, ~100 MB)…"
  uv pip install torch torchvision
fi

# ── frontend build (if static/ missing or --rebuild flag) ───────────
need_build=0
if [[ "${1:-}" == "--rebuild" ]] || [[ ! -d static ]] || [[ ! -f static/index.html ]]; then
  need_build=1
fi
# Also rebuild if any .tsx/.ts/.css source is newer than static/index.html
if [[ -f static/index.html ]]; then
  if find frontend/src -newer static/index.html -name '*.tsx' -o -newer static/index.html -name '*.ts' -o -newer static/index.html -name '*.css' 2>/dev/null | grep -q .; then
    need_build=1
  fi
fi
if [[ $need_build -eq 1 ]]; then
  info "building frontend (vite)…"
  (cd frontend && npm run build)
  rm -rf static
  cp -r frontend/dist static
  ok "static/ rebuilt"
fi

# ── stop existing backend (native or docker) ────────────────────────
existing=$(lsof -ti:"$PORT" 2>/dev/null || true)
if [[ -n "$existing" ]]; then
  warn "port $PORT busy (pids: $existing) — stopping"
  kill $existing 2>/dev/null || true
  sleep 2
fi
if docker ps --format '{{.Names}}' 2>/dev/null | grep -q "^ml-intern$"; then
  info "stopping the dockerized ml-intern container…"
  docker compose stop ml-intern 2>&1 | tail -2
fi

# ── start backend ────────────────────────────────────────────────────
info "starting backend on http://localhost:$PORT (logs: $LOG)…"
cd backend
nohup uv run uvicorn main:app --host 0.0.0.0 --port "$PORT" \
  --reload-dir "$REPO_ROOT/backend" --reload-dir "$REPO_ROOT/agent" \
  > "$LOG" 2>&1 &
echo $! > /tmp/ml-intern.pid
cd "$REPO_ROOT"

# ── wait for ready ───────────────────────────────────────────────────
info "waiting for /api/health…"
for i in {1..30}; do
  if curl -sf "http://localhost:$PORT/api/health" >/dev/null 2>&1; then
    ok "backend healthy after ${i}s"
    break
  fi
  sleep 1
  if [[ $i -eq 30 ]]; then
    warn "backend not responding after 30s — check $LOG"
    tail -30 "$LOG"
    exit 1
  fi
done

# ── status summary ───────────────────────────────────────────────────
echo
echo -e "${GREEN}──────────────────────────────────────────────${NC}"
ok  "ml-intern is up"
echo -e "  Web UI:        ${CYAN}http://localhost:$PORT${NC}"
echo -e "  Backend log:   $LOG  (\`tail -f $LOG\` to watch)"
echo -e "  PID:           $(cat /tmp/ml-intern.pid 2>/dev/null || echo '?')"
echo -e "  Local mode:    ML_INTERN_LOCAL_MODE=$(grep -E '^ML_INTERN_LOCAL_MODE' .env | cut -d= -f2)"
proj=$(grep -E '^ML_INTERN_PROJECT_ROOT' .env 2>/dev/null | cut -d= -f2 || true)
echo -e "  Project root:  ${proj:-<per-session, set via web UI>}"
echo -e "${GREEN}──────────────────────────────────────────────${NC}"
echo
echo "Existing project folders:"
if [[ -d projects ]]; then
  ls -1 projects 2>/dev/null | grep -v '^_template$' | sed 's/^/  • /' || echo "  (none yet)"
fi
echo
echo "From the web UI's \"New Task\" button, type a project name to lock the session."
echo "Sessions are listed at $PORT/api/sessions and resumable via $PORT/api/sessions/archived"
echo
echo "Stop with: ./up.sh stop"

# ── open browser ─────────────────────────────────────────────────────
if command -v open >/dev/null; then
  sleep 1
  open "http://localhost:$PORT" || true
fi
