"""Agent API routes — REST + SSE endpoints.

All routes (except /health) require authentication via the get_current_user
dependency. In dev mode (no OAUTH_CLIENT_ID), auth is bypassed automatically.
"""

import asyncio
import json
import logging
import os
from typing import Any

from dependencies import get_current_user, require_huggingface_org_member
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.responses import StreamingResponse
from litellm import acompletion
from models import (
    ApprovalRequest,
    HealthResponse,
    LLMHealthResponse,
    SessionInfo,
    SessionResponse,
    SubmitRequest,
    TruncateRequest,
)
from session_manager import MAX_SESSIONS, AgentSession, SessionCapacityError, session_manager

import user_quotas

from agent.core.llm_params import _resolve_llm_params

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["agent"])

AVAILABLE_MODELS = [
    {
        "id": "moonshotai/Kimi-K2.6",
        "label": "Kimi K2.6",
        "provider": "huggingface",
        "tier": "free",
        "recommended": True,
    },
    {
        "id": "bedrock/us.anthropic.claude-opus-4-6-v1",
        "label": "Claude Opus 4.6",
        "provider": "anthropic",
        "tier": "pro",
        "recommended": True,
    },
    {
        "id": "MiniMaxAI/MiniMax-M2.7",
        "label": "MiniMax M2.7",
        "provider": "huggingface",
        "tier": "free",
    },
    {
        "id": "zai-org/GLM-5.1",
        "label": "GLM 5.1",
        "provider": "huggingface",
        "tier": "free",
    },
    {
        "id": "openai/gpt-5.5",
        "label": "GPT-5.5",
        "provider": "openai",
        "tier": "pro",
    },
    {
        "id": "deepseek/deepseek-v4-pro",
        "label": "DeepSeek V4 Pro",
        "provider": "deepseek",
        "tier": "pro",
        "recommended": True,
    },
    {
        "id": "deepseek/deepseek-v4-flash",
        "label": "DeepSeek V4 Flash",
        "provider": "deepseek",
        "tier": "pro",
    },
    {
        "id": "anthropic/claude-opus-4-6",
        "label": "Claude Opus 4.6 (direct)",
        "provider": "anthropic",
        "tier": "pro",
    },
]


def _is_anthropic_model(model_id: str) -> bool:
    return "anthropic" in model_id


async def _require_hf_for_anthropic(request: Request, model_id: str) -> None:
    """403 if a non-``huggingface``-org user tries to select an Anthropic model.

    Anthropic models are billed to the Space's ``ANTHROPIC_API_KEY``; every
    other model in ``AVAILABLE_MODELS`` is routed through HF Router and
    billed via ``X-HF-Bill-To``. The gate only fires for Anthropic so
    non-HF users can still freely switch between the free models.

    Pattern: https://github.com/huggingface/ml-intern/pull/63
    """
    if not _is_anthropic_model(model_id):
        return
    if not await require_huggingface_org_member(request):
        raise HTTPException(
            status_code=403,
            detail={
                "error": "anthropic_restricted",
                "message": (
                    "Opus is gated to HF staff. Pick a free model — "
                    "Kimi K2.6, MiniMax M2.7, or GLM 5.1 — instead."
                ),
            },
        )


async def _enforce_claude_quota(
    user: dict[str, Any],
    agent_session: AgentSession,
) -> None:
    """Charge the user's daily Claude quota on first use of Anthropic in a session.

    Runs at *message-submit* time, not session-create time — so spinning up a
    Claude session to look around doesn't burn quota. The ``claude_counted``
    flag on ``AgentSession`` guards against re-counting the same session.

    No-ops when the session's current model isn't Anthropic, or when this
    session has already been charged. Raises 429 when the user has hit
    their daily cap.
    """
    if agent_session.claude_counted:
        return
    model_name = agent_session.session.config.model_name
    if not _is_anthropic_model(model_name):
        return
    user_id = user["user_id"]
    used = await user_quotas.get_claude_used_today(user_id)
    cap = user_quotas.daily_cap_for(user.get("plan"))
    if used >= cap:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "claude_daily_cap",
                "plan": user.get("plan", "free"),
                "cap": cap,
                "message": (
                    "Daily Claude limit reached. Upgrade to HF Pro for "
                    f"{user_quotas.CLAUDE_PRO_DAILY}/day or use a free model."
                ),
            },
        )
    await user_quotas.increment_claude(user_id)
    agent_session.claude_counted = True


def _check_session_access(session_id: str, user: dict[str, Any]) -> None:
    """Verify the user has access to the given session. Raises 403 or 404."""
    info = session_manager.get_session_info(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session_manager.verify_session_access(session_id, user["user_id"]):
        raise HTTPException(status_code=403, detail="Access denied to this session")


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        active_sessions=session_manager.active_session_count,
        max_sessions=MAX_SESSIONS,
    )


@router.get("/health/llm", response_model=LLMHealthResponse)
async def llm_health_check() -> LLMHealthResponse:
    """Check if the LLM provider is reachable and the API key is valid.

    Makes a minimal 1-token completion call.  Catches common errors:
    - 401 → invalid API key
    - 402/insufficient_quota → out of credits
    - 429 → rate limited
    - timeout / network → provider unreachable
    """
    model = session_manager.config.model_name
    try:
        llm_params = _resolve_llm_params(model, reasoning_effort="high")
        await acompletion(
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1,
            timeout=10,
            **llm_params,
        )
        return LLMHealthResponse(status="ok", model=model)
    except Exception as e:
        err_str = str(e).lower()
        error_type = "unknown"

        if (
            "401" in err_str
            or "auth" in err_str
            or "invalid" in err_str
            or "api key" in err_str
        ):
            error_type = "auth"
        elif (
            "402" in err_str
            or "credit" in err_str
            or "quota" in err_str
            or "insufficient" in err_str
            or "billing" in err_str
        ):
            error_type = "credits"
        elif "429" in err_str or "rate" in err_str:
            error_type = "rate_limit"
        elif "timeout" in err_str or "connect" in err_str or "network" in err_str:
            error_type = "network"

        logger.warning(f"LLM health check failed ({error_type}): {e}")
        return LLMHealthResponse(
            status="error",
            model=model,
            error=str(e)[:500],
            error_type=error_type,
        )


@router.get("/config/model")
async def get_model() -> dict:
    """Get current model and available models. No auth required."""
    return {
        "current": session_manager.config.model_name,
        "available": AVAILABLE_MODELS,
    }


_TITLE_STRIP_CHARS = str.maketrans("", "", "`*_~#[]()")


@router.post("/title")
async def generate_title(
    request: SubmitRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Generate a short title for a chat session based on the first user message.

    Always uses gpt-oss-120b via Cerebras on the HF router. The tab headline
    renders as plain text, so the model is told to avoid markdown and any
    stray formatting characters are stripped before returning. gpt-oss is a
    reasoning model — reasoning_effort=low keeps the reasoning budget small
    so the 60-token output budget isn't consumed before the title is written.
    """
    api_key = (
        os.environ.get("INFERENCE_TOKEN")
        or (user.get("hf_token") if isinstance(user, dict) else None)
        or os.environ.get("HF_TOKEN")
    )
    try:
        response = await acompletion(
            # Double openai/ prefix: LiteLLM strips the first as its provider
            # prefix, leaving the HF model id on the wire for the router.
            model="openai/openai/gpt-oss-120b:cerebras",
            api_base="https://router.huggingface.co/v1",
            api_key=api_key,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Generate a very short title (max 6 words) for a chat conversation "
                        "that starts with the following user message. "
                        "Reply with ONLY the title in plain text. "
                        "Do NOT use markdown, backticks, asterisks, quotes, brackets, or any "
                        "formatting characters. No punctuation at the end."
                    ),
                },
                {"role": "user", "content": request.text[:500]},
            ],
            max_tokens=60,
            temperature=0.3,
            timeout=10,
            reasoning_effort="low",
        )
        title = response.choices[0].message.content.strip().strip('"').strip("'")
        title = title.translate(_TITLE_STRIP_CHARS).strip()
        if len(title) > 50:
            title = title[:50].rstrip() + "…"
        return {"title": title}
    except Exception as e:
        logger.warning(f"Title generation failed: {e}")
        fallback = request.text.strip()
        title = fallback[:40].rstrip() + "…" if len(fallback) > 40 else fallback
        return {"title": title}


@router.post("/session", response_model=SessionResponse)
async def create_session(
    request: Request, user: dict = Depends(get_current_user)
) -> SessionResponse:
    """Create a new agent session bound to the authenticated user.

    The user's HF access token is extracted from the Authorization header
    and stored in the session so that tools (e.g. hf_jobs) can act on
    behalf of the user.

    Optional body ``{"model"?: <id>}`` selects the session's LLM; unknown
    ids are rejected (400). The Claude-quota gate runs at message-submit
    time, not here — spinning up an Opus session to look around is free.

    Returns 503 if the server or user has reached the session limit.
    """
    # Extract the user's HF token (Bearer header, HttpOnly cookie, or env var)
    hf_token = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        hf_token = auth_header[7:]
    if not hf_token:
        hf_token = request.cookies.get("hf_access_token")
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")

    # Optional model override. Empty body falls back to the config default.
    model: str | None = None
    project: str | None = None
    try:
        body = await request.json()
    except Exception:
        body = None
    if isinstance(body, dict):
        model = body.get("model")
        project = body.get("project")

    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model and model not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    # Opus is gated to HF staff (PR #63). Only fires when the resolved model
    # is Anthropic; free models pass through.
    resolved_model = model or session_manager.config.model_name
    await _require_hf_for_anthropic(request, resolved_model)

    try:
        session_id = await session_manager.create_session(
            user_id=user["user_id"], hf_token=hf_token, model=model, project=project,
        )
    except SessionCapacityError as e:
        raise HTTPException(status_code=503, detail=str(e))

    return SessionResponse(session_id=session_id, ready=True)


@router.post("/session/restore-summary", response_model=SessionResponse)
async def restore_session_summary(
    request: Request, body: dict, user: dict = Depends(get_current_user)
) -> SessionResponse:
    """Create a new session seeded with a summary of the caller's prior
    conversation. The client sends its cached messages; we run the standard
    summarization prompt on them and drop the result into the new
    session's context as a user-role system note.

    Optional ``"model"`` in the body overrides the session's LLM. The
    Claude-quota gate runs at message-submit time, not here.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="Missing 'messages' array")

    hf_token = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        hf_token = auth_header[7:]
    if not hf_token:
        hf_token = request.cookies.get("hf_access_token")
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")

    model = body.get("model")
    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model and model not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model}")

    resolved_model = model or session_manager.config.model_name
    await _require_hf_for_anthropic(request, resolved_model)

    try:
        session_id = await session_manager.create_session(
            user_id=user["user_id"], hf_token=hf_token, model=model
        )
    except SessionCapacityError as e:
        raise HTTPException(status_code=503, detail=str(e))

    try:
        summarized = await session_manager.seed_from_summary(session_id, messages)
    except ValueError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.exception("seed_from_summary failed")
        raise HTTPException(status_code=500, detail=f"Summary failed: {e}")

    logger.info(
        f"Seeded session {session_id} for {user.get('username', 'unknown')} "
        f"(summary of {summarized} messages)"
    )
    return SessionResponse(session_id=session_id, ready=True)


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> SessionInfo:
    """Get session information. Only accessible by the session owner."""
    _check_session_access(session_id, user)
    info = session_manager.get_session_info(session_id)
    return SessionInfo(**info)


@router.post("/session/{session_id}/model")
async def set_session_model(
    session_id: str,
    body: dict,
    request: Request,
    user: dict = Depends(get_current_user),
) -> dict:
    """Switch the active model for a single session (tab-scoped).

    Takes effect on the next LLM call in that session — other sessions
    (including other browser tabs) are unaffected. Model switches don't
    charge quota — the Claude-quota gate only fires at message-submit time.

    Switching TO an Anthropic model requires HF org membership (PR #63);
    free-model switches are unrestricted.
    """
    _check_session_access(session_id, user)
    model_id = body.get("model")
    if not model_id:
        raise HTTPException(status_code=400, detail="Missing 'model' field")
    valid_ids = {m["id"] for m in AVAILABLE_MODELS}
    if model_id not in valid_ids:
        raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")
    await _require_hf_for_anthropic(request, model_id)
    agent_session = session_manager.sessions.get(session_id)
    if not agent_session:
        raise HTTPException(status_code=404, detail="Session not found")
    agent_session.session.update_model(model_id)
    logger.info(
        f"Session {session_id} model → {model_id} "
        f"(by {user.get('username', 'unknown')})"
    )
    return {"session_id": session_id, "model": model_id}


@router.get("/user/quota")
async def get_user_quota(user: dict = Depends(get_current_user)) -> dict:
    """Return the user's plan tier and today's Claude-session quota state."""
    plan = user.get("plan", "free")
    used = await user_quotas.get_claude_used_today(user["user_id"])
    cap = user_quotas.daily_cap_for(plan)
    return {
        "plan": plan,
        "claude_used_today": used,
        "claude_daily_cap": cap,
        "claude_remaining": max(0, cap - used),
    }


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(user: dict = Depends(get_current_user)) -> list[SessionInfo]:
    """List sessions belonging to the authenticated user."""
    sessions = session_manager.list_sessions(user_id=user["user_id"])
    return [SessionInfo(**s) for s in sessions]


@router.get("/projects")
async def list_projects() -> list[dict]:
    """List subdirectories under ``./projects/`` — these are the project
    folders the agent can be locked to via ``POST /api/session {project: ...}``.

    Each entry: ``{name, path, has_data, has_code, has_models, last_modified}``.
    The frontend uses this to populate a project picker in the New Task dialog.
    """
    from pathlib import Path as _Path
    # Resolve projects/ relative to repo root (parent of backend/), not the
    # backend cwd. Matches what session_manager.create_session writes to.
    repo_root = _Path(__file__).resolve().parent.parent.parent
    proj_dir = repo_root / "projects"
    if not proj_dir.exists():
        return []
    out: list[dict] = []
    for d in sorted(proj_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir() or d.name.startswith("."):
            continue
        out.append({
            "name": d.name,
            "path": str(d.resolve()),
            "has_data": (d / "data").exists() and any((d / "data").iterdir()) if (d / "data").is_dir() else False,
            "has_code": (d / "code").exists() and any((d / "code").iterdir()) if (d / "code").is_dir() else False,
            "has_models": (d / "models").exists() and any((d / "models").iterdir()) if (d / "models").is_dir() else False,
            "last_modified": d.stat().st_mtime,
        })
    return out


@router.get("/sessions/archived")
async def list_archived_sessions(
    user: dict = Depends(get_current_user),
    min_messages: int = 10,
    include_dummy: bool = False,
    requires_project: bool = False,
) -> list[dict]:
    """List session JSON snapshots in ./session_logs/ on disk.

    These persist across container/process restarts. Each returns the
    session_id, save time, model, message count, project (if available),
    and a preview of the first user message.

    Default filters out:
      • sessions with ``message_count < min_messages`` (default 10) —
        these are typically empty smoke-tests, dummy sessions, or
        single-prompt-no-reply experiments
      • sessions where the agent never produced a real assistant turn
        with text content

    Pass ``?include_dummy=1`` to see everything (useful for cleanup).
    Pass ``?requires_project=1`` to only show sessions tied to a project
    folder (i.e. created via the per-session project picker).

    Use ``POST /api/session/import`` to rehydrate.
    """
    import json as _json
    from pathlib import Path as _Path

    log_dir = _Path(__file__).resolve().parent.parent / "session_logs"
    if not log_dir.exists():
        return []
    out: list[dict] = []
    for f in sorted(log_dir.glob("session_*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = _json.loads(f.read_text())
        except Exception:
            continue
        msgs = data.get("messages") or []
        # Filter dummy sessions unless explicitly asked for them.
        if not include_dummy:
            if len(msgs) < min_messages:
                continue
            # Reject sessions that have no real assistant text reply
            # (just system + a single user message + maybe failed turn).
            has_real_assistant = any(
                isinstance(m, dict)
                and m.get("role") == "assistant"
                and (
                    (m.get("content") and (str(m["content"]).strip() if not isinstance(m.get("content"), list) else any(c for c in m["content"])))
                    or m.get("tool_calls")
                )
                for m in msgs
            )
            if not has_real_assistant:
                continue
        # Optional: only show sessions tied to a project folder.
        proj = data.get("project") or data.get("project_name")
        if requires_project and not proj:
            continue
        first_user = next((m for m in msgs if isinstance(m, dict) and m.get("role") == "user"), {})
        first_user_text = first_user.get("content") or ""
        if isinstance(first_user_text, list):
            first_user_text = " ".join(
                x.get("text", "") if isinstance(x, dict) else str(x) for x in first_user_text
            )
        out.append({
            "session_id": data.get("session_id") or f.stem,
            "filename": f.name,
            "saved_at": data.get("last_save_time") or data.get("session_end_time"),
            "model": data.get("model_name"),
            "project": data.get("project") or data.get("project_name"),  # populated for new sessions
            "message_count": len(msgs),
            "first_user_preview": (first_user_text[:200] + "…") if len(first_user_text) > 200 else first_user_text,
        })
    return out


@router.delete("/sessions/archived/dummies")
async def cleanup_dummy_archived_sessions(
    user: dict = Depends(get_current_user),
    min_messages: int = 4,
) -> dict:
    """Permanently delete session_logs/ JSON files that look like noise:
    sessions with ``message_count < min_messages`` (default 4) AND no real
    assistant reply. This is destructive — files cannot be recovered.

    Returns ``{"deleted": [...filenames...], "kept": <int>}``.
    """
    import json as _json
    from pathlib import Path as _Path

    log_dir = _Path(__file__).resolve().parent.parent / "session_logs"
    if not log_dir.exists():
        return {"deleted": [], "kept": 0}

    deleted: list[str] = []
    kept = 0
    for f in log_dir.glob("session_*.json"):
        try:
            data = _json.loads(f.read_text())
        except Exception:
            continue
        msgs = data.get("messages") or []
        is_dummy = len(msgs) < min_messages
        if not is_dummy:
            has_real_assistant = any(
                isinstance(m, dict) and m.get("role") == "assistant"
                and ((m.get("content") and str(m.get("content")).strip()) or m.get("tool_calls"))
                for m in msgs
            )
            is_dummy = not has_real_assistant
        if is_dummy:
            try:
                f.unlink()
                deleted.append(f.name)
            except Exception:
                pass
        else:
            kept += 1
    logger.info(f"cleanup_dummy_archived_sessions: deleted {len(deleted)} files, kept {kept}")
    return {"deleted": deleted, "kept": kept}


@router.delete("/sessions/archived/duplicates")
async def deduplicate_archived_sessions(
    user: dict = Depends(get_current_user),
) -> dict:
    """Permanently delete redundant archives. Within each
    ``(project, first_60_chars_of_first_user_prompt)`` group, keeps the
    file with the largest ``message_count`` and deletes the rest.

    This is exactly the case where the same conversation auto-saved
    multiple times across runs — different ``session_id`` and timestamp,
    but identical content trail. The kept file preserves the most work.

    Returns ``{"deleted": [...], "kept": [...]}``. Destructive.
    """
    import json as _json
    from pathlib import Path as _Path
    from collections import defaultdict

    log_dir = _Path(__file__).resolve().parent.parent / "session_logs"
    if not log_dir.exists():
        return {"deleted": [], "kept": []}

    groups: dict[tuple, list] = defaultdict(list)
    for f in log_dir.glob("session_*.json"):
        try:
            data = _json.loads(f.read_text())
        except Exception:
            continue
        msgs = data.get("messages") or []
        proj = data.get("project") or data.get("project_name") or "(none)"
        first_user = next((m for m in msgs if isinstance(m, dict) and m.get("role") == "user"), {})
        c = first_user.get("content", "") or ""
        if isinstance(c, list):
            c = " ".join(str(x) for x in c)
        key = (proj, c[:120].strip())
        groups[key].append((len(msgs), f))

    deleted: list[str] = []
    kept: list[str] = []
    for sessions in groups.values():
        if len(sessions) <= 1:
            kept.append(sessions[0][1].name if sessions else "")
            continue
        # Sort by message count descending — keep largest, delete the rest.
        sessions.sort(key=lambda x: -x[0])
        kept.append(sessions[0][1].name)
        for _, f in sessions[1:]:
            try:
                f.unlink()
                deleted.append(f.name)
            except Exception:
                pass
    logger.info(f"deduplicate_archived_sessions: deleted {len(deleted)}, kept {len(kept)}")
    return {"deleted": deleted, "kept": kept}


@router.delete("/sessions/archived/{session_id}")
async def delete_archived_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Permanently delete a specific archived session JSON from disk.
    Idempotent: returns the list of files removed (empty if none matched).
    """
    from pathlib import Path as _Path

    log_dir = _Path(__file__).resolve().parent.parent / "session_logs"
    if not log_dir.exists():
        return {"deleted": []}
    deleted: list[str] = []
    for f in log_dir.glob(f"session_{session_id}_*.json"):
        try:
            f.unlink()
            deleted.append(f.name)
        except Exception:
            pass
    return {"deleted": deleted}


@router.post("/session/import", response_model=SessionResponse)
async def import_session_from_disk(
    request: Request, user: dict = Depends(get_current_user)
) -> SessionResponse:
    """Rehydrate a session from a session_logs/ JSON file into memory.

    Body: ``{"session_id": "..."}`` — looks up the matching JSON in
    ./session_logs/ and creates a new in-memory session pre-loaded with
    that conversation's messages. The new session keeps the original's
    model_name and full message history; the agent can continue exactly
    where the saved session left off.

    Different from ``restore-summary`` (which summarizes prior turns into
    a single seed message) — this preserves every original message and
    every tool call.
    """
    import json as _json
    from pathlib import Path as _Path

    try:
        body = await request.json()
    except Exception:
        body = {}
    sid_filter = (body.get("session_id") or "").strip()
    if not sid_filter:
        raise HTTPException(status_code=400, detail="Missing 'session_id' in body")

    log_dir = _Path(__file__).resolve().parent.parent / "session_logs"
    matches = list(log_dir.glob(f"session_{sid_filter}_*.json"))
    if not matches:
        # Fall back to filename match in case the user passed a full filename
        matches = list(log_dir.glob(sid_filter)) + list(log_dir.glob(f"*{sid_filter}*"))
    if not matches:
        raise HTTPException(status_code=404, detail=f"No session log matching '{sid_filter}' in ./session_logs/")
    # Pick the most recent
    src = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)[0]
    try:
        data = _json.loads(src.read_text())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not parse {src.name}: {e}")

    saved_messages = data.get("messages") or []
    saved_model = data.get("model_name")

    # Pull the user's HF token + extract token from auth (same as create_session)
    hf_token: str | None = None
    auth_header = request.headers.get("Authorization", "")
    if auth_header.startswith("Bearer "):
        hf_token = auth_header[7:]
    if not hf_token:
        hf_token = request.cookies.get("hf_access_token")
    if not hf_token:
        hf_token = os.environ.get("HF_TOKEN")

    try:
        new_session_id = await session_manager.create_session(
            user_id=user["user_id"], hf_token=hf_token, model=saved_model
        )
    except SessionCapacityError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Seed the new session's context with the saved messages (skip the
    # boilerplate system message — the new session already has its own).
    from litellm import Message as _Message
    agent_session = session_manager.sessions.get(new_session_id)
    if agent_session is not None:
        seeded: list[_Message] = []
        for m in saved_messages:
            if not isinstance(m, dict):
                continue
            if m.get("role") == "system":
                continue
            try:
                seeded.append(_Message(**{k: v for k, v in m.items() if v is not None}))
            except Exception:
                continue
        if seeded:
            agent_session.session.context_manager.items.extend(seeded)
            logger.info(
                f"Imported {len(seeded)} messages from {src.name} into new session {new_session_id}"
            )

    return SessionResponse(session_id=new_session_id, ready=True)


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Delete a session. Only accessible by the session owner."""
    _check_session_access(session_id, user)
    success = await session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.post("/submit")
async def submit_input(
    request: SubmitRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Submit user input to a session. Only accessible by the session owner."""
    _check_session_access(request.session_id, user)
    agent_session = session_manager.sessions.get(request.session_id)
    if agent_session is not None:
        await _enforce_claude_quota(user, agent_session)
    success = await session_manager.submit_user_input(request.session_id, request.text)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "submitted", "session_id": request.session_id}


@router.post("/approve")
async def submit_approval(
    request: ApprovalRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Submit tool approvals to a session. Only accessible by the session owner."""
    _check_session_access(request.session_id, user)
    approvals = [
        {
            "tool_call_id": a.tool_call_id,
            "approved": a.approved,
            "feedback": a.feedback,
            "edited_script": a.edited_script,
        }
        for a in request.approvals
    ]
    success = await session_manager.submit_approval(request.session_id, approvals)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "submitted", "session_id": request.session_id}


@router.post("/chat/{session_id}")
async def chat_sse(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """SSE endpoint: submit input or approval, then stream events until turn ends."""
    _check_session_access(session_id, user)

    agent_session = session_manager.sessions.get(session_id)
    if not agent_session or not agent_session.is_active:
        raise HTTPException(status_code=404, detail="Session not found or inactive")

    # Parse body
    body = await request.json()

    # Subscribe BEFORE submitting so we never miss events — even if the
    # agent loop processes the submission before this coroutine continues.
    broadcaster = agent_session.broadcaster
    sub_id, event_queue = broadcaster.subscribe()

    # Submit the operation
    text = body.get("text")
    approvals = body.get("approvals")

    # Gate user-message sends against the daily Claude quota. Approvals are
    # continuations of an in-progress turn — the session was already charged
    # on its first message, so we skip the gate there.
    if text is not None and not approvals:
        try:
            await _enforce_claude_quota(user, agent_session)
        except HTTPException:
            broadcaster.unsubscribe(sub_id)
            raise

    try:
        if approvals:
            formatted = [
                {
                    "tool_call_id": a["tool_call_id"],
                    "approved": a["approved"],
                    "feedback": a.get("feedback"),
                    "edited_script": a.get("edited_script"),
                }
                for a in approvals
            ]
            success = await session_manager.submit_approval(session_id, formatted)
        elif text is not None:
            success = await session_manager.submit_user_input(session_id, text)
        else:
            broadcaster.unsubscribe(sub_id)
            raise HTTPException(status_code=400, detail="Must provide 'text' or 'approvals'")

        if not success:
            broadcaster.unsubscribe(sub_id)
            raise HTTPException(status_code=404, detail="Session not found or inactive")
    except HTTPException:
        raise
    except Exception:
        broadcaster.unsubscribe(sub_id)
        raise

    return _sse_response(broadcaster, event_queue, sub_id)


# ---------------------------------------------------------------------------
# Shared SSE helpers
# ---------------------------------------------------------------------------
_TERMINAL_EVENTS = {"turn_complete", "approval_required", "error", "interrupted", "shutdown"}
_SSE_KEEPALIVE_SECONDS = 15


def _sse_response(broadcaster, event_queue, sub_id) -> StreamingResponse:
    """Build a StreamingResponse that drains *event_queue* as SSE,
    sending keepalive comments every 15 s to prevent proxy timeouts."""

    async def event_generator():
        try:
            while True:
                try:
                    msg = await asyncio.wait_for(
                        event_queue.get(), timeout=_SSE_KEEPALIVE_SECONDS
                    )
                except asyncio.TimeoutError:
                    # SSE comment — ignored by parsers, keeps connection alive
                    yield ": keepalive\n\n"
                    continue
                event_type = msg.get("event_type", "")
                yield f"data: {json.dumps(msg)}\n\n"
                if event_type in _TERMINAL_EVENTS:
                    break
        finally:
            broadcaster.unsubscribe(sub_id)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/events/{session_id}")
async def subscribe_events(
    session_id: str,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """Subscribe to events for a running session without submitting new input.

    Used by the frontend to re-attach after a connection drop (e.g. screen
    sleep).  Returns 404 if the session isn't active or isn't processing.
    """
    _check_session_access(session_id, user)

    agent_session = session_manager.sessions.get(session_id)
    if not agent_session or not agent_session.is_active:
        raise HTTPException(status_code=404, detail="Session not found or inactive")

    broadcaster = agent_session.broadcaster
    sub_id, event_queue = broadcaster.subscribe()
    return _sse_response(broadcaster, event_queue, sub_id)


@router.post("/interrupt/{session_id}")
async def interrupt_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Interrupt the current operation in a session."""
    _check_session_access(session_id, user)
    success = await session_manager.interrupt(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "interrupted", "session_id": session_id}


@router.get("/session/{session_id}/messages")
async def get_session_messages(
    session_id: str, user: dict = Depends(get_current_user)
) -> list[dict]:
    """Return the session's message history from memory."""
    _check_session_access(session_id, user)
    agent_session = session_manager.sessions.get(session_id)
    if not agent_session or not agent_session.is_active:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return [msg.model_dump() for msg in agent_session.session.context_manager.items]


@router.post("/undo/{session_id}")
async def undo_session(session_id: str, user: dict = Depends(get_current_user)) -> dict:
    """Undo the last turn in a session."""
    _check_session_access(session_id, user)
    success = await session_manager.undo(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "undo_requested", "session_id": session_id}


@router.post("/truncate/{session_id}")
async def truncate_session(
    session_id: str, body: TruncateRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Truncate conversation to before a specific user message."""
    _check_session_access(session_id, user)
    success = await session_manager.truncate(session_id, body.user_message_index)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found, inactive, or message index out of range")
    return {"status": "truncated", "session_id": session_id}


@router.post("/compact/{session_id}")
async def compact_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Compact the context in a session."""
    _check_session_access(session_id, user)
    success = await session_manager.compact(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "compact_requested", "session_id": session_id}


@router.post("/shutdown/{session_id}")
async def shutdown_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Shutdown a session."""
    _check_session_access(session_id, user)
    success = await session_manager.shutdown_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "shutdown_requested", "session_id": session_id}


@router.post("/feedback/{session_id}")
async def submit_feedback(
    session_id: str,
    body: dict,
    user: dict = Depends(get_current_user),
) -> dict:
    """Attach a user feedback signal to a session's event log.

    Body: {rating: "up"|"down"|"outcome_success"|"outcome_fail",
           turn_index?: int, comment?: str, message_id?: str}
    Appended as a `feedback` event and saved with the session trajectory.
    """
    _check_session_access(session_id, user)
    agent_session = session_manager.sessions.get(session_id)
    if not agent_session:
        raise HTTPException(status_code=404, detail="Session not found")

    rating = body.get("rating")
    if rating not in {"up", "down", "outcome_success", "outcome_fail"}:
        raise HTTPException(status_code=400, detail="invalid rating")

    from agent.core import telemetry
    await telemetry.record_feedback(
        agent_session.session,
        rating=rating,
        turn_index=body.get("turn_index"),
        message_id=body.get("message_id"),
        comment=body.get("comment"),
    )
    # Fire-and-forget save so feedback reaches the dataset even if the user
    # closes the tab right after clicking.
    if agent_session.session.config.save_sessions:
        agent_session.session.save_and_upload_detached(
            agent_session.session.config.session_dataset_repo
        )
    return {"status": "ok"}


