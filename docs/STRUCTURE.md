# ml-intern Structure & How To Run Native + With Your Own Data

## Top-level map

```
ml-intern/
├── 🔒 CORE (do not edit unless changing agent behavior)
│   ├── agent/                          ← agent loop, tools, system prompt
│   │   ├── prompts/system_prompt_v3.yaml   ← what the agent sees as system instructions
│   │   ├── tools/                          ← 17 tool implementations
│   │   │   ├── local_tools.py              ← bash/write/read/edit (subprocess in agent process)
│   │   │   ├── sandbox_tool.py             ← sandbox_create + sandbox-routed bash/write/read/edit (HF Space)
│   │   │   ├── hf_jobs.py                  ← cloud GPU training jobs (paid)
│   │   │   └── (etc — papers, datasets, github, mcp, …)
│   │   ├── core/agent_loop.py              ← LLM call → tool call → repeat
│   │   ├── core/llm_params.py              ← provider routing (anthropic/openai/deepseek/HF)
│   │   └── context_manager/manager.py      ← message history + auto-compaction
│   ├── backend/                        ← FastAPI server
│   │   ├── main.py                         ← entrypoint (port 7860)
│   │   ├── routes/agent.py                 ← REST + SSE
│   │   └── session_manager.py              ← per-session asyncio task lifecycle
│   ├── frontend/                       ← React + Vite UI (built into ./static for prod)
│   ├── configs/main_agent_config.json  ← model + MCP servers + save_sessions toggle
│   └── pyproject.toml                  ← Python deps
│
├── 🟡 GENERATED (per-run output)
│   └── session_logs/
│       └── session_<uuid>_<timestamp>.json   ← full message history + telemetry per chat
│
├── 🟢 YOUR INPUT/OUTPUT (everything you control)
│   └── projects/
│       └── <project>/
│           ├── data/        ← drop your datasets HERE
│           ├── code/        ← agent writes Python here
│           ├── models/      ← agent writes weights/ONNX here
│           ├── notes/       ← markdown reports
│           ├── prompts/     ← snapshots of what you asked
│           └── README.md    ← project goal + status
│
└── 🔧 WRAPPERS (optional)
    ├── Dockerfile + docker-compose.yml ← Docker mode
    └── run-native.sh                   ← native macOS run (uses MPS)
```

## Run native (no Docker, full MPS GPU access)

```bash
cd /Users/mazhar/Downloads/Code/Personal/ml-intern

# One-time setup
uv sync                                    # python deps
uv pip install torch torchvision           # for MPS-aware ML
(cd frontend && npm install && npm run build)
cp -r frontend/dist static                 # backend serves /static at /

# Run
./run-native.sh                            # port 7860
# or:
cd backend && uv run uvicorn main:app --host 0.0.0.0 --port 7860
```

Open <http://localhost:7860/>. Same UI, no Docker, MPS available to any tool the agent runs.

## Where do session traces live?

When the agent runs a chat session, it persists the full message history to:

```
session_logs/session_<uuid>_<YYYYMMDD_HHMMSS>.json
```

These contain every user message, assistant message, tool call, tool result, and telemetry (latency, tokens, cost). Even after a server restart, **session_logs/ is read-only output** — the running session is in-memory. The file gets written incrementally during the run AND uploaded to the HF dataset configured in `configs/main_agent_config.json` (`session_dataset_repo`) if `save_sessions: true` and you have HF_TOKEN.

## How to do a research project with your own data

### 1. Copy the template

```bash
cp -r projects/_template projects/credit-fraud
```

### 2. Drop your data in

```bash
# CSV
cp ~/Downloads/transactions.csv projects/credit-fraud/data/

# Image folder
cp -r ~/datasets/x-rays projects/credit-fraud/data/

# Or unzip a Kaggle dump
unzip ~/Downloads/kaggle-dump.zip -d projects/credit-fraud/data/
```

### 3. Edit the project README to describe the goal

```bash
$EDITOR projects/credit-fraud/README.md
```

### 4. In the chat at <http://localhost:7860/>, paste this prompt

```
We're working on a self-contained ML research project at:
  /Users/mazhar/Downloads/Code/Personal/ml-intern/projects/credit-fraud/

Rules:
- Do NOT call sandbox_create. Use the local bash/write/read/edit tools only.
- Input data is at projects/credit-fraud/data/
- Save code to projects/credit-fraud/code/
- Save models to projects/credit-fraud/models/
- Save plots/notes to projects/credit-fraud/notes/
- Read README.md first, then begin.

Goal: <describe your research task here, e.g. "Train a fraud detector on transactions.csv. Use XGBoost and a small neural net. Compare ROC-AUC.">
```

The "Do NOT call sandbox_create" line is the magic that keeps everything local — the agent will use its `local_tools` which run **directly on your Mac with MPS**.

### 5. Watch progress

- **In the browser:** the chat panel shows tool-call cards live
- **In your editor:** open `projects/credit-fraud/` — files appear/update as the agent writes them
- **Backend logs:** `tail -f /tmp/native-backend.log` (or whatever you redirected)

### 6. Take it elsewhere

```bash
tar -czf credit-fraud.tar.gz projects/credit-fraud/
# Move to another machine, untar — it has data + code + models + history.
```

## Models you can use

The dropdown reads from `backend/routes/agent.py:AVAILABLE_MODELS`. Currently configured:

- `deepseek/deepseek-v4-pro` (default, paid via DEEPSEEK_API_KEY in `.env`)
- `deepseek/deepseek-v4-flash`
- `openai/gpt-5.5` (needs OPENAI_API_KEY)
- `anthropic/claude-opus-4-6` (needs ANTHROPIC_API_KEY)
- `moonshotai/Kimi-K2.6` (free via HF Router with HF_TOKEN — recommended for agent loops)
- `MiniMaxAI/MiniMax-M2.7` (free)
- `zai-org/GLM-5.1` (free)
- `bedrock/us.anthropic.claude-opus-4-6-v1` (HF staff only)

To set a different default, edit `configs/main_agent_config.json` → `model_name`.
