# ML Intern — Web UI Usage

Short guide for running the web interface locally and invoking agent runs that survive tab closes.

## Start the stack

### Recommended: Docker Compose (persists session memory across restarts)

```bash
# from the repo root, .env already populated
docker compose up -d --build

# tail logs
docker compose logs -f

# stop (data in volumes survives)
docker compose down

# nuke session/cache volumes too (fresh start)
docker compose down -v
```

Open <http://localhost:7860/> in any browser. The container serves the built React app and the API on the same port.

**What persists across container restarts** (defined in [docker-compose.yml](../docker-compose.yml)):

| Volume | Mount point | Holds |
| --- | --- | --- |
| `ml-intern-sessions` | `/app/backend/session_logs` | Per-session JSONL traces written by [agent/core/session.py](../agent/core/session.py) |
| `ml-intern-hf-cache` | `/home/user/.cache/huggingface` | HF model/dataset downloads — avoid re-pulling on restart |
| `ml-intern-uv-cache` | `/home/user/.cache/uv` | Python wheel cache |

`.env` is bind-mounted via `env_file:` — edit and run `docker compose restart` to pick up new keys without rebuilding.

### Native dev (alternative — hot-reload while editing code)

Two terminals, run from the repo root.

```bash
# Terminal A — backend (port 7860, FastAPI + SSE)
cd backend && uv run uvicorn main:app --host 0.0.0.0 --port 7860

# Terminal B — frontend dev server (port 5173, Vite + React)
cd frontend && npm run dev
```

Open <http://localhost:5173/> for the Vite dev server (proxies to `:7860`).

## Configure providers

Create `.env` in the repo root with whichever keys you have. Missing keys just hide the corresponding model from the working set — the server still boots.

| Variable | Unlocks |
| --- | --- |
| `ANTHROPIC_API_KEY` | `anthropic/claude-opus-4-6` (direct API) |
| `OPENAI_API_KEY` | `openai/gpt-5.5` |
| `DEEPSEEK_API_KEY` | `deepseek/deepseek-v4-pro`, `deepseek/deepseek-v4-flash` |
| `HF_TOKEN` | Free models via HF Router (Kimi, MiniMax, GLM) + `Claude Opus 4.6` (Bedrock-routed) |
| `GITHUB_TOKEN` | The agent's GitHub code-search tool |

Example:

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
HF_TOKEN=hf_...
GITHUB_TOKEN=ghp_...
```

## Pick a model

The model dropdown in the UI (top of the chat pane) lists everything declared in `AVAILABLE_MODELS` ([backend/routes/agent.py](../backend/routes/agent.py)). Default working set:

- **Kimi K2.6** — free, HF Router (recommended for casual use)
- **GPT-5.5** — OpenAI direct
- **DeepSeek V4 Pro / V4 Flash** — DeepSeek direct API at <https://api.deepseek.com>. Both models support thinking and non-thinking modes via the request body (toggled by the agent at runtime, not via OpenAI's `reasoning_effort`). V4 Pro is the recommended default.
- **Claude Opus 4.6** — Bedrock-routed (HF staff only, gated by `_require_hf_for_anthropic`)
- **Claude Opus 4.6 (direct)** — Anthropic direct API (uses your `ANTHROPIC_API_KEY`)
- **MiniMax M2.7**, **GLM 5.1** — free, HF Router

You can switch model mid-conversation; new turns route to the new provider.

## Sample prompts

Try these from the chat input to exercise different parts of the agent:

1. *"Fine-tune `distilbert-base-uncased` on the `ag_news` dataset for 2 epochs and report eval accuracy."* — exercises the HF jobs + sandbox tools.
2. *"Find a vision-language model under 3B params with strong OCR scores and write a minimal `transformers` inference snippet."* — exercises HF model search + code generation.
3. *"Search arXiv for papers on speculative decoding from the last 30 days and produce a one-paragraph summary of each."* — exercises the HF papers tool + summarization.

## Tab-close survival

Agent runs are **server-side background tasks**, not browser-driven. When you submit a turn:

1. The backend creates an `AgentSession` with a detached `asyncio.create_task` ([session_manager.py:217](../backend/session_manager.py)).
2. Events are pushed into a fan-out broadcaster ([session_manager.py:49](../backend/session_manager.py)) that supports multiple concurrent SSE subscribers.
3. The browser tab is just one such subscriber — closing it only unsubscribes; the agent keeps running.

To resume monitoring after closing a tab:

```bash
# List active sessions
curl http://localhost:7860/api/sessions

# Re-subscribe to a session's live event stream
curl -N http://localhost:7860/api/events/<session_id>

# Fetch full message history (in-memory)
curl http://localhost:7860/api/session/<session_id>/messages
```

In the UI, opening <http://localhost:5173/> after closing the tab restores the session list from `/api/sessions` automatically. Sessions also flush to `smolagents/ml-intern-sessions` on HF Datasets at backend shutdown ([backend/main.py:47-60](../backend/main.py)) as a final safety net.

## Troubleshooting

- **Backend won't start**: must be run from `backend/` directory — its imports are flat (`from routes.agent import ...`).
- **Provider returns 401**: key missing or wrong in `.env`. Hit `GET /api/health/llm` for a per-provider probe.
- **DeepSeek thinking mode**: V4 models toggle thinking via the request body. The agent does not forward OpenAI-style `reasoning_effort` to DeepSeek; default is thinking-on. See [agent/core/llm_params.py](../agent/core/llm_params.py) for the routing branch.
- **Anthropic models gated**: `bedrock/us.anthropic.claude-opus-4-6-v1` is HF-staff-only; switch to `anthropic/claude-opus-4-6` (direct) if you have your own `ANTHROPIC_API_KEY`.
- **Frontend can't reach backend**: confirm backend on `:7860` and CORS origins in [backend/main.py:73-77](../backend/main.py) include your dev port.

## API surface (used by the UI)

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/api/health` | Liveness + active session count |
| `GET` | `/api/health/llm` | Per-provider key probe |
| `GET` | `/api/config/model` | Current model + full `AVAILABLE_MODELS` list |
| `POST` | `/api/session` | Create a new agent session |
| `GET` | `/api/sessions` | List sessions for the authenticated user |
| `POST` | `/api/submit` | Submit a user message (returns immediately, agent runs async) |
| `GET` | `/api/events/{session_id}` | SSE stream of live events |
| `POST` | `/api/chat/{session_id}` | Submit + stream events in one request (used by the UI) |
| `POST` | `/api/interrupt/{session_id}` | Cancel an in-flight turn |
| `GET` | `/api/session/{session_id}/messages` | Full in-memory message history |
| `POST` | `/api/session/{session_id}/model` | Swap model on an existing session |
| `DELETE` | `/api/session/{session_id}` | Delete + cleanup a session |

Full OpenAPI at <http://localhost:7860/docs>.
