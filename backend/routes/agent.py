"""Agent API routes — REST + SSE endpoints.

All routes (except /health) require authentication via the get_current_user
dependency. In dev mode (no OAUTH_CLIENT_ID), auth is bypassed automatically.
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any

from dependencies import (
    INTERNAL_HF_TOKEN_KEY,
    get_current_user,
)
from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Request,
)
from fastapi.exceptions import RequestValidationError
from fastapi.responses import StreamingResponse
from huggingface_hub.errors import HfHubHTTPError
from litellm import Message, acompletion
from pydantic import ValidationError
from starlette.datastructures import FormData, UploadFile
from dataset_uploads import (
    MAX_DATASET_UPLOAD_BYTES,
    dataset_context_note,
    push_dataset_upload_to_hub,
)
from models import (
    ApprovalRequest,
    DatasetUploadResponse,
    HealthResponse,
    LLMHealthResponse,
    SessionInfo,
    SessionNotificationsRequest,
    SessionResponse,
    SessionYoloRequest,
    SubmitRequest,
    TruncateRequest,
    UsageResponse,
)
from session_manager import (
    MAX_SESSIONS,
    AgentSession,
    SessionCapacityError,
    session_manager,
)

from agent.core.hf_access import get_jobs_access
from agent.core.hf_tokens import resolve_hf_request_token
from agent.core.local_models import local_model_provider
from agent.core.llm_params import _resolve_llm_params
from agent.core.model_ids import (
    CLAUDE_OPUS_48_MODEL_ID,
    DEEPSEEK_V4_PRO_MODEL_ID,
    GLM_52_MODEL_ID,
    GPT_55_MODEL_ID,
    KIMI_K27_CODE_MODEL_ID,
    MINIMAX_M3_MODEL_ID,
    strip_huggingface_model_prefix,
)
from agent.core.prompt_caching import with_prompt_cache_params
from usage import build_usage_response

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["agent"])
_background_route_tasks: set[asyncio.Task] = set()

DEFAULT_GPT_MODEL_ID = GPT_55_MODEL_ID
DEFAULT_MODEL_ID = GLM_52_MODEL_ID
DATASET_UPLOAD_MULTIPART_SLACK_BYTES = 1024 * 1024


async def _reset_usage_window(session_id: str) -> dict[str, Any] | None:
    return await session_manager.reset_session_usage_window(
        session_id,
        started_at=datetime.utcnow(),
    )


async def _refresh_usage_and_upload(
    agent_session: AgentSession,
    *,
    error_code: str,
) -> None:
    session = agent_session.session
    try:
        await session_manager.refresh_session_usage_metrics(
            agent_session,
            error_code=error_code,
        )
        session.save_and_upload_detached(session.config.session_dataset_repo)
    except Exception as e:
        logger.warning(
            "Background usage refresh/upload failed for %s: %s",
            agent_session.session_id,
            e,
        )


def _schedule_usage_refresh_and_upload(
    agent_session: AgentSession,
    *,
    error_code: str,
) -> None:
    task = asyncio.create_task(
        _refresh_usage_and_upload(agent_session, error_code=error_code)
    )
    _background_route_tasks.add(task)
    task.add_done_callback(_background_route_tasks.discard)


def _available_models() -> list[dict[str, Any]]:
    # Curated dropdown for this fork: free HF-routed models plus premium
    # direct-API options. DeepSeek uses the direct ``deepseek/`` provider
    # (DEEPSEEK_API_KEY) to match agent/core/llm_params.py routing. The Claude
    # entry is pinned to the Bedrock id rather than the config default, since
    # this fork defaults to deepseek/deepseek-v4-pro.
    models = [
        {
            "id": CLAUDE_OPUS_48_MODEL_ID,
            "label": "Claude Opus 4.8",
        },
        {
            "id": DEFAULT_GPT_MODEL_ID,
            "label": "GPT-5.5",
        },
        {
            "id": KIMI_K27_CODE_MODEL_ID,
            "label": "Kimi K2.7 Code",
        },
        {
            "id": MINIMAX_M3_MODEL_ID,
            "label": "MiniMax M3",
        },
        {
            "id": DEFAULT_MODEL_ID,
            "label": "GLM 5.2",
            "recommended": True,
        },
        # DeepSeek via the direct ``deepseek/`` provider (DEEPSEEK_API_KEY →
        # api.deepseek.com), matching agent/core/llm_params.py routing. Distinct
        # from the HF-router DeepSeek id so the user's own DeepSeek key is used.
        # Selectable from the picker; GLM 5.2 remains the default.
        {
            "id": "deepseek/deepseek-v4-pro",
            "label": "DeepSeek V4 Pro",
        },
        {
            "id": "deepseek/deepseek-v4-flash",
            "label": "DeepSeek V4 Flash",
        },
    ]
    return models


AVAILABLE_MODELS = _available_models()


def _valid_model_ids() -> set[str]:
    return {m["id"] for m in AVAILABLE_MODELS}


def _validate_model_id(model_id: str | None) -> None:
    if not model_id or model_id in _valid_model_ids():
        return
    raise HTTPException(status_code=400, detail=f"Unknown model: {model_id}")


def _default_model() -> str:
    return DEFAULT_MODEL_ID


def _model_override_for_new_session(requested_model: str | None) -> str | None:
    """Return the model override to use when creating a new session.

    Explicit model requests are honored. Empty web requests default to GLM 5.2.
    """
    return requested_model or _default_model()


def _user_hf_token(user: dict[str, Any] | None) -> str | None:
    if not isinstance(user, dict):
        return None
    return user.get(INTERNAL_HF_TOKEN_KEY)


def _model_requires_hf_router_token(model_id: str | None) -> bool:
    normalized = strip_huggingface_model_prefix(model_id) or model_id or ""
    return local_model_provider(normalized) is None


def _reject_oversize_dataset_upload(request: Request) -> None:
    raw_content_length = request.headers.get("content-length")
    if raw_content_length is None:
        return
    try:
        content_length = int(raw_content_length)
    except (TypeError, ValueError):
        return
    if content_length > MAX_DATASET_UPLOAD_BYTES + DATASET_UPLOAD_MULTIPART_SLACK_BYTES:
        raise HTTPException(
            status_code=413,
            detail="Dataset upload exceeds the 100 MB limit.",
        )


def _dataset_upload_file_from_form(form: FormData) -> UploadFile:
    uploaded_files = [
        (key, value)
        for key, value in form.multi_items()
        if isinstance(value, UploadFile)
    ]
    if len(uploaded_files) != 1:
        raise HTTPException(
            status_code=400,
            detail="Upload exactly one dataset file.",
        )
    field_name, upload = uploaded_files[0]
    if field_name != "file":
        raise HTTPException(
            status_code=400,
            detail="Missing 'file' upload field.",
        )
    return upload


def _dataset_upload_hub_http_exception(error: HfHubHTTPError) -> HTTPException:
    status_code = getattr(error.response, "status_code", None)
    if status_code == 401:
        detail = "Hugging Face rejected the token used for the dataset upload."
        return HTTPException(status_code=401, detail=detail)
    if status_code == 403:
        detail = (
            "Hugging Face denied permission to create or write to the dataset repo."
        )
        return HTTPException(status_code=403, detail=detail)
    if status_code == 404:
        detail = "Could not find the Hugging Face namespace or dataset repo."
        return HTTPException(status_code=404, detail=detail)
    if status_code == 429:
        detail = "Hugging Face Hub rate limit reached while uploading the dataset."
        return HTTPException(status_code=429, detail=detail)
    return HTTPException(
        status_code=502,
        detail="Hugging Face Hub upload failed. Please try again.",
    )


async def _check_session_access(
    session_id: str,
    user: dict[str, Any],
    request: Request | None = None,
    preload_sandbox: bool = True,
) -> AgentSession:
    """Verify and lazily load the user's session. Raises 403 or 404."""
    hf_token = (
        resolve_hf_request_token(request)
        if request is not None
        else _user_hf_token(user)
    )
    agent_session = await session_manager.ensure_session_loaded(
        session_id,
        user["user_id"],
        hf_token=hf_token,
        hf_username=user.get("username"),
        user_plan=user.get("plan"),
        preload_sandbox=preload_sandbox,
    )
    if not agent_session:
        raise HTTPException(status_code=404, detail="Session not found")
    if user["user_id"] != "dev" and agent_session.user_id not in {
        user["user_id"],
        "dev",
    }:
        raise HTTPException(status_code=403, detail="Access denied to this session")
    return agent_session


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(
        status="ok",
        active_sessions=session_manager.active_session_count,
        max_sessions=MAX_SESSIONS,
    )


@router.get("/health/llm", response_model=LLMHealthResponse)
async def llm_health_check(
    request: Request,
    user: dict = Depends(get_current_user),
) -> LLMHealthResponse:
    """Check if the LLM provider is reachable and the API key is valid.

    Makes a minimal 1-token completion call against the authenticated user's
    default model when a token is available. For token-less HF Router requests,
    returns ``status="skipped"`` instead of making an unauthenticated probe.
    Catches common errors:
    - 401 → invalid API key
    - 402/insufficient_quota → out of credits
    - 429 → rate limited
    - timeout / network → provider unreachable
    """
    model = _default_model()
    hf_token = resolve_hf_request_token(request)
    if _model_requires_hf_router_token(model) and not hf_token:
        return LLMHealthResponse(status="skipped", model=model)

    try:
        llm_params = _resolve_llm_params(
            model,
            hf_token,
            reasoning_effort="high",
        )
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
    try:
        await _check_session_access(request.session_id, user)
        llm_params = _resolve_llm_params(
            "openai/gpt-oss-120b:cerebras",
            _user_hf_token(user),
            reasoning_effort="low",
        )
        llm_params = with_prompt_cache_params(llm_params)
        response = await acompletion(
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
            **llm_params,
        )
        title = response.choices[0].message.content.strip().strip('"').strip("'")
        title = title.translate(_TITLE_STRIP_CHARS).strip()
        if len(title) > 50:
            title = title[:50].rstrip() + "…"
        try:
            await session_manager.update_session_title(request.session_id, title)
        except Exception:
            logger.debug(
                "Skipping title persistence for missing session %s", request.session_id
            )
        return {"title": title}
    except Exception as e:
        logger.warning(f"Title generation failed: {e}")
        fallback = request.text.strip()
        title = fallback[:40].rstrip() + "…" if len(fallback) > 40 else fallback
        try:
            await _check_session_access(request.session_id, user)
            await session_manager.update_session_title(request.session_id, title)
        except Exception:
            logger.debug(
                "Skipping fallback title persistence for missing session %s",
                request.session_id,
            )
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
    ids are rejected (400). Empty requests use the web default.

    Returns 503 if the server or user has reached the session limit.
    """
    # Extract the user's HF token (Bearer header, HttpOnly cookie, or env var)
    hf_token = resolve_hf_request_token(request)

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

    _validate_model_id(model)

    # Empty requests use the web default.
    model = _model_override_for_new_session(model)

    try:
        session_id = await session_manager.create_session(
            user_id=user["user_id"],
            hf_username=user.get("username"),
            hf_token=hf_token,
            user_plan=user.get("plan"),
            model=model,
            project=project,
            is_pro=user.get("plan") == "pro",
        )
    except SessionCapacityError as e:
        raise HTTPException(status_code=503, detail=str(e))

    await _reset_usage_window(session_id)

    return SessionResponse(
        session_id=session_id,
        ready=True,
        model=model,
    )


@router.post("/session/restore-summary", response_model=SessionResponse)
async def restore_session_summary(
    request: Request, body: dict, user: dict = Depends(get_current_user)
) -> SessionResponse:
    """Create a new session seeded with a summary of the caller's prior
    conversation. The client sends its cached messages; we run the standard
    summarization prompt on them and drop the result into the new
    session's context as a user-role system note.

    Optional ``"model"`` in the body overrides the session's LLM; otherwise
    the new session uses the web default.
    """
    messages = body.get("messages")
    if not isinstance(messages, list) or not messages:
        raise HTTPException(status_code=400, detail="Missing 'messages' array")

    hf_token = resolve_hf_request_token(request)

    model = body.get("model")
    _validate_model_id(model)

    model = _model_override_for_new_session(model)

    try:
        session_id = await session_manager.create_session(
            user_id=user["user_id"],
            hf_username=user.get("username"),
            hf_token=hf_token,
            user_plan=user.get("plan"),
            model=model,
            is_pro=user.get("plan") == "pro",
        )
    except SessionCapacityError as e:
        raise HTTPException(status_code=503, detail=str(e))

    await _reset_usage_window(session_id)

    await _check_session_access(
        session_id,
        user,
        request,
        preload_sandbox=False,
    )
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
    return SessionResponse(
        session_id=session_id,
        ready=True,
        model=model,
    )


@router.get("/session/{session_id}", response_model=SessionInfo)
async def get_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> SessionInfo:
    """Get session information. Only accessible by the session owner."""
    await _check_session_access(session_id, user)
    info = session_manager.get_session_info(session_id)
    return SessionInfo(**info)


@router.post("/session/{session_id}/activate", response_model=SessionInfo)
async def activate_session(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> SessionInfo:
    """Mark a session as actively revisited without resetting usage."""
    await _check_session_access(session_id, user, request)
    info = await session_manager.activate_session(session_id)
    if not info:
        raise HTTPException(status_code=404, detail="Session not found")
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
    (including other browser tabs) are unaffected.
    """
    agent_session = await _check_session_access(session_id, user, request)
    model_id = body.get("model")
    if not model_id:
        raise HTTPException(status_code=400, detail="Missing 'model' field")
    _validate_model_id(model_id)
    if not agent_session:
        raise HTTPException(status_code=404, detail="Session not found")
    await session_manager.update_session_model(session_id, model_id)
    logger.info(
        f"Session {session_id} model → {model_id} "
        f"(by {user.get('username', 'unknown')})"
    )
    return {"session_id": session_id, "model": model_id}


@router.post("/session/{session_id}/notifications")
async def set_session_notifications(
    session_id: str,
    body: SessionNotificationsRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """Replace the session's auto-notification destinations."""
    agent_session = await _check_session_access(session_id, user)
    try:
        destinations = session_manager.set_notification_destinations(
            session_id, body.destinations
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    await session_manager.persist_session_snapshot(agent_session)
    return {
        "session_id": session_id,
        "notification_destinations": destinations,
    }


@router.post("/session/{session_id}/datasets", response_model=DatasetUploadResponse)
async def upload_session_dataset(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> DatasetUploadResponse:
    """Upload a CSV/JSON dataset file to a private Hub dataset for this session."""
    file: UploadFile | None = None
    try:
        _reject_oversize_dataset_upload(request)
        agent_session = await _check_session_access(session_id, user, request)
        if not agent_session or not agent_session.is_active:
            raise HTTPException(status_code=404, detail="Session not found")
        if agent_session.is_processing:
            raise HTTPException(
                status_code=409,
                detail="Cannot upload a dataset while the agent is processing.",
            )
        if agent_session.session.pending_approval:
            raise HTTPException(
                status_code=409,
                detail="Resolve pending approvals before uploading a dataset.",
            )

        hf_token = (
            resolve_hf_request_token(request, include_env_fallback=False)
            or _user_hf_token(user)
            or resolve_hf_request_token(request)
        )
        if not hf_token:
            raise HTTPException(
                status_code=401,
                detail="A Hugging Face token is required to upload datasets.",
            )

        form = await request.form(
            max_files=1,
            max_fields=1,
            max_part_size=MAX_DATASET_UPLOAD_BYTES,
        )
        file = _dataset_upload_file_from_form(form)
        hf_username = user.get("username") or agent_session.hf_username
        uploaded = await push_dataset_upload_to_hub(
            upload=file,
            session_id=session_id,
            hf_username=hf_username,
            hf_token=hf_token,
        )
        agent_session.session.context_manager.add_message(
            Message(role="user", content=dataset_context_note(uploaded))
        )
        session_manager._touch(agent_session)
        await session_manager.persist_session_snapshot(agent_session)
        logger.info(
            "Uploaded dataset file %s to %s for session %s",
            uploaded.filename,
            uploaded.repo_id,
            session_id,
        )
        return DatasetUploadResponse(**uploaded.response_payload())
    except HTTPException:
        raise
    except HfHubHTTPError as e:
        logger.warning(
            "Hub rejected dataset upload for session %s: status=%s request_id=%s",
            session_id,
            getattr(e.response, "status_code", None),
            getattr(e, "request_id", None),
        )
        raise _dataset_upload_hub_http_exception(e)
    except Exception:
        logger.exception("Dataset upload failed for session %s", session_id)
        raise HTTPException(
            status_code=502,
            detail="Dataset upload failed. Please try again.",
        )
    finally:
        if file is not None:
            await file.close()


@router.patch("/session/{session_id}/yolo")
async def set_session_yolo(
    session_id: str,
    body: SessionYoloRequest,
    user: dict = Depends(get_current_user),
) -> dict:
    """Update the session-scoped auto-approval policy."""
    await _check_session_access(session_id, user)
    try:
        summary = await session_manager.update_session_auto_approval(
            session_id,
            enabled=body.enabled,
            cost_cap_usd=body.cost_cap_usd,
            cap_provided="cost_cap_usd" in body.model_fields_set,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"session_id": session_id, **summary}


@router.get("/user/jobs-access")
async def get_jobs_access_info(
    request: Request, user: dict = Depends(get_current_user)
) -> dict:
    """Return the namespaces the current token can run HF Jobs under.

    Credits are enforced by the HF API at job-creation time, not here —
    the response only describes which wallets the caller is allowed to
    pick from. Pro is irrelevant.
    """
    token = resolve_hf_request_token(request)

    access = await get_jobs_access(token or "")
    return {
        "eligible_namespaces": access.eligible_namespaces if access else [],
        "default_namespace": access.default_namespace if access else None,
        "billing_url": "https://huggingface.co/settings/billing",
    }


@router.get("/usage", response_model=UsageResponse)
async def get_usage(
    request: Request,
    session_id: str | None = None,
    tz: str | None = None,
    user: dict = Depends(get_current_user),
) -> dict:
    """Return app-attributed usage for the current user."""
    if session_id:
        await _check_session_access(
            session_id,
            user,
            request,
            preload_sandbox=False,
        )
    usage = await build_usage_response(
        session_manager,
        user_id=user["user_id"],
        hf_token=(
            resolve_hf_request_token(request, include_env_fallback=False)
            or _user_hf_token(user)
            or resolve_hf_request_token(request)
        ),
        session_id=session_id,
        timezone_name=tz,
    )
    if session_id:
        auto_approval = (
            await session_manager.reconcile_session_auto_approval_from_usage(
                session_id,
                usage,
            )
        )
        if auto_approval is not None:
            usage["auto_approval"] = auto_approval
    return usage


@router.get("/sessions", response_model=list[SessionInfo])
async def list_sessions(user: dict = Depends(get_current_user)) -> list[SessionInfo]:
    """List sessions belonging to the authenticated user."""
    sessions = await session_manager.list_sessions(user_id=user["user_id"])
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


@router.post("/session/{session_id}/sandbox/teardown")
async def teardown_session_sandbox(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Best-effort sandbox teardown that preserves durable chat history."""
    await _check_session_access(session_id, user, preload_sandbox=False)
    task = asyncio.create_task(session_manager.teardown_sandbox(session_id))
    _background_route_tasks.add(task)
    task.add_done_callback(_background_route_tasks.discard)
    return {"status": "teardown_requested", "session_id": session_id}


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Delete a session. Only accessible by the session owner."""
    await _check_session_access(session_id, user, preload_sandbox=False)
    success = await session_manager.delete_session(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "deleted", "session_id": session_id}


@router.post("/submit")
async def submit_input(
    request: Request, user: dict = Depends(get_current_user)
) -> dict:
    """Submit user input to a session. Only accessible by the session owner."""
    # Parse the body manually so session ownership can be checked before the
    # text-length constraints fire — otherwise a non-owner sending an empty
    # or oversized text gets a 422 leaking the constraint instead of the 404
    # they'd get for any other access to a session they don't own.
    try:
        payload = await request.json()
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    if not isinstance(payload, dict):
        raise HTTPException(status_code=422, detail="Body must be a JSON object")
    raw_session_id = payload.get("session_id")
    if not isinstance(raw_session_id, str) or not raw_session_id:
        raise RequestValidationError(
            [
                {
                    "type": "missing",
                    "loc": ("body", "session_id"),
                    "msg": "Field required",
                    "input": payload,
                }
            ]
        )
    await _check_session_access(raw_session_id, user)
    try:
        body = SubmitRequest(**payload)
    except ValidationError as exc:
        raise RequestValidationError(exc.errors()) from exc
    success = await session_manager.submit_user_input(body.session_id, body.text)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "submitted", "session_id": body.session_id}


@router.post("/approve")
async def submit_approval(
    request: ApprovalRequest, user: dict = Depends(get_current_user)
) -> dict:
    """Submit tool approvals to a session. Only accessible by the session owner."""
    await _check_session_access(request.session_id, user)
    approvals = [
        {
            "tool_call_id": a.tool_call_id,
            "approved": a.approved,
            "feedback": a.feedback,
            "edited_script": a.edited_script,
            "namespace": a.namespace,
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
    agent_session = await _check_session_access(session_id, user, request)
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

    try:
        if approvals:
            formatted = [
                {
                    "tool_call_id": a["tool_call_id"],
                    "approved": a["approved"],
                    "feedback": a.get("feedback"),
                    "edited_script": a.get("edited_script"),
                    "namespace": a.get("namespace"),
                }
                for a in approvals
            ]
            success = await session_manager.submit_approval(session_id, formatted)
        elif text is not None:
            success = await session_manager.submit_user_input(session_id, text)
        else:
            broadcaster.unsubscribe(sub_id)
            raise HTTPException(
                status_code=400, detail="Must provide 'text' or 'approvals'"
            )

        if not success:
            broadcaster.unsubscribe(sub_id)
            raise HTTPException(status_code=404, detail="Session not found or inactive")
    except HTTPException:
        broadcaster.unsubscribe(sub_id)
        raise
    except Exception:
        broadcaster.unsubscribe(sub_id)
        raise

    return _sse_response(broadcaster, event_queue, sub_id)


@router.post("/pro-click/{session_id}")
async def record_pro_click(
    session_id: str,
    body: dict,
    user: dict = Depends(get_current_user),
) -> dict:
    """Record a click on a Pro upgrade CTA shown from inside a session."""
    agent_session = await _check_session_access(session_id, user)

    from agent.core import telemetry

    await telemetry.record_pro_cta_click(
        agent_session.session,
        source=str(body.get("source") or "unknown"),
        target=str(body.get("target") or "pro_pricing"),
    )
    if agent_session.session.config.save_sessions:
        _schedule_usage_refresh_and_upload(
            agent_session,
            error_code="pro_click_billing_snapshot_error",
        )
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Shared SSE helpers
# ---------------------------------------------------------------------------
_TERMINAL_EVENTS = {
    "turn_complete",
    "approval_required",
    "error",
    "interrupted",
    "shutdown",
}
_SSE_KEEPALIVE_SECONDS = 15


def _last_event_seq(request: Request) -> int:
    raw = (
        request.headers.get("last-event-id") or request.query_params.get("after") or "0"
    )
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return 0


def _format_sse(msg: dict[str, Any]) -> str:
    seq = msg.get("seq")
    body = {"event_type": msg.get("event_type"), "data": msg.get("data") or {}}
    if seq is not None:
        body["seq"] = seq
        return f"id: {seq}\ndata: {json.dumps(body)}\n\n"
    return f"data: {json.dumps(body)}\n\n"


def _event_doc_to_msg(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "event_type": doc.get("event_type"),
        "data": doc.get("data") or {},
        "seq": doc.get("seq"),
    }


def _sse_response(
    broadcaster,
    event_queue,
    sub_id,
    *,
    replay_events: list[dict[str, Any]] | None = None,
    after_seq: int = 0,
) -> StreamingResponse:
    """Build a StreamingResponse that drains *event_queue* as SSE,
    sending keepalive comments every 15 s to prevent proxy timeouts."""

    async def event_generator():
        try:
            for doc in replay_events or []:
                msg = _event_doc_to_msg(doc)
                seq = msg.get("seq")
                if isinstance(seq, int) and seq <= after_seq:
                    continue
                yield _format_sse(msg)
                if msg.get("event_type", "") in _TERMINAL_EVENTS:
                    return

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
                yield _format_sse(msg)
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
    request: Request,
    user: dict = Depends(get_current_user),
) -> StreamingResponse:
    """Subscribe to events for a running session without submitting new input.

    Used by the frontend to re-attach after a connection drop (e.g. screen
    sleep).  Returns 404 if the session isn't active or isn't processing.
    """
    agent_session = await _check_session_access(session_id, user, request)
    if not agent_session or not agent_session.is_active:
        raise HTTPException(status_code=404, detail="Session not found or inactive")

    after_seq = _last_event_seq(request)
    replay_events = await session_manager._store().load_events_after(
        session_id, after_seq
    )
    broadcaster = agent_session.broadcaster
    sub_id, event_queue = broadcaster.subscribe()
    return _sse_response(
        broadcaster,
        event_queue,
        sub_id,
        replay_events=replay_events,
        after_seq=after_seq,
    )


@router.post("/interrupt/{session_id}")
async def interrupt_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Interrupt the current operation in a session."""
    await _check_session_access(session_id, user)
    success = await session_manager.interrupt(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "interrupted", "session_id": session_id}


@router.get("/session/{session_id}/messages")
async def get_session_messages(
    session_id: str, user: dict = Depends(get_current_user)
) -> list[dict]:
    """Return the session's message history from memory."""
    agent_session = await _check_session_access(session_id, user)
    if not agent_session or not agent_session.is_active:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return [
        msg.model_dump(mode="json")
        for msg in agent_session.session.context_manager.items
    ]


@router.post("/undo/{session_id}")
async def undo_session(session_id: str, user: dict = Depends(get_current_user)) -> dict:
    """Undo the last turn in a session."""
    await _check_session_access(session_id, user)
    success = await session_manager.undo(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "undo_requested", "session_id": session_id}


@router.post("/truncate/{session_id}")
async def truncate_session(
    session_id: str,
    request: Request,
    user: dict = Depends(get_current_user),
) -> dict:
    """Truncate conversation to before a specific user message."""
    # Check session ownership before parsing the request body so a 404 on a
    # non-existent / non-owned session_id beats the 422 schema-validation error
    # (otherwise the response leaks the required field name to non-owners).
    await _check_session_access(session_id, user)
    try:
        body = TruncateRequest(**(await request.json()))
    except ValidationError as exc:
        # Re-raise as RequestValidationError so FastAPI returns its standard
        # structured 422 schema (`{"detail": [{"type":..., "loc":..., ...}]}`)
        # instead of a string-stringified Pydantic dump.
        raise RequestValidationError(exc.errors()) from exc
    except (json.JSONDecodeError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    success = await session_manager.truncate(session_id, body.user_message_index)
    if not success:
        raise HTTPException(
            status_code=404,
            detail="Session not found, inactive, or message index out of range",
        )
    return {"status": "truncated", "session_id": session_id}


@router.post("/compact/{session_id}")
async def compact_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Compact the context in a session."""
    await _check_session_access(session_id, user)
    success = await session_manager.compact(session_id)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found or inactive")
    return {"status": "compact_requested", "session_id": session_id}


@router.post("/shutdown/{session_id}")
async def shutdown_session(
    session_id: str, user: dict = Depends(get_current_user)
) -> dict:
    """Shutdown a session."""
    await _check_session_access(session_id, user)
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
    agent_session = await _check_session_access(session_id, user)

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
        _schedule_usage_refresh_and_upload(
            agent_session,
            error_code="feedback_billing_snapshot_error",
        )
    return {"status": "ok"}
