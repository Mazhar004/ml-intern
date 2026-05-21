# ML Intern — Quick Instructions

## Start the system

```bash
./up.sh           # boots backend, opens http://localhost:7860
./up.sh stop      # shuts down
```

That's it. MPS GPU enabled, no Docker required, sessions persist across restarts.

---

## Start a new project

1. Open <http://localhost:7860/>
2. Click **"+ New Task"** (bottom-left)
3. Type a project name in the prompt — e.g. `customer-churn`
4. The folder `projects/customer-churn/` is created automatically:
   ```
   projects/customer-churn/
   ├── README.md       ← you can edit to describe the goal
   ├── data/           ← drop your input datasets here
   ├── code/           ← agent writes Python here
   ├── models/         ← agent writes weights/ONNX here
   ├── notes/          ← agent writes plots/reports
   └── prompts/        ← snapshots of what you asked
   ```
5. Drop your data into `projects/customer-churn/data/` from Finder or your editor
6. Tell the agent the goal, e.g.:
   *"Read README.md, then train a fraud detector on data/transactions.csv. Save the best model to models/."*

The agent is locked to that folder — it can only read/write inside `projects/customer-churn/`.

---

## Link a new session to an EXISTING project

Same flow, but type the **existing** project name:

1. Click **"+ New Task"**
2. The prompt **prefills with your last-used project name** — just press Enter to reuse it.
   (Or type `oculomics`, `customer-churn`, etc. — whichever folder you want.)
3. The new session sees all prior `code/`, `models/`, `notes/`, `data/` immediately.

**Key point:** one project can have many sessions. Deleting a session does NOT delete the project folder. Examples of legitimate multi-session-per-project:

- Session 1: literature review + planning
- Session 2: implement model
- Session 3: train and evaluate
- Session 4: write up results

All four read/write the same `projects/<name>/` folder. Each is its own conversation thread.

---

## Resume an old conversation

After a server restart, prior sessions appear in the sidebar with a **"needs a catch-up"** badge. Click one → click **"Catch me up"** → full chat history restored, you can continue.

To do it from the terminal:
```bash
# List what's on disk
curl -s http://localhost:7860/api/sessions/archived | jq

# Resume one
curl -X POST http://localhost:7860/api/session/import \
  -H 'content-type: application/json' \
  -d '{"session_id":"<id-from-list>"}'
```

---

## Cleanup

```bash
# Drop sessions that look like noise (< 10 messages, no real reply)
curl -X DELETE 'http://localhost:7860/api/sessions/archived/dummies?min_messages=10'

# Collapse duplicate copies of the same conversation
curl -X DELETE http://localhost:7860/api/sessions/archived/duplicates

# Delete one specific archived session (file on disk)
curl -X DELETE http://localhost:7860/api/sessions/archived/<id>
```

Or just click the trash icon on any session row in the sidebar — that wipes in-memory + disk + frontend cache.

---

## Models you can use

In the dropdown at the bottom of the chat:

| Model | Cost | Best for |
|---|---|---|
| **Kimi K2.6** | Free (HF Router) | Agent loops, default recommendation |
| **DeepSeek V4 Pro** | $$ (DEEPSEEK_API_KEY) | Strong reasoning, long context |
| **GPT-5.5** | $$$ (OPENAI_API_KEY) | High quality |
| **Claude Opus 4.6 (direct)** | $$$$ (ANTHROPIC_API_KEY) | Best for agentic work |

Set keys in `.env` at the repo root, then `./up.sh stop` and `./up.sh` to reload.

---

## Where everything lives

```
ml-intern/
├── projects/<name>/         ← your work (data, code, models, notes) — gitignored
├── backend/session_logs/    ← chat history JSONs — gitignored
├── .env                     ← your API keys — gitignored
└── (everything else)        ← ml-intern source code
```

- **Project folders** persist forever until you `rm -rf` them.
- **Session logs** are kept on disk; deletable from the UI or `curl`.
- **HF Hub uploads** the agent makes are **always private** (per system prompt rule).
- Nothing is auto-pushed to git; you decide what gets committed.
