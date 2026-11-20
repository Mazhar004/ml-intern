"""Web model picker and default-model resolution.

Shared by ``routes/agent.py`` (picker, new sessions, LLM health check,
``/api/config/model``) and ``session_manager.py`` (session fallbacks), so it
imports neither of them.

``AVAILABLE_MODELS`` is the full picker: HF Router models plus direct-provider
models (Anthropic, OpenAI, DeepSeek). The web app shows
``visible_picker_models()``, which leaves out a direct-provider model while
its API key is unset. Every picker id stays valid for new sessions and model
switches, which reject a missing key with a clear 400, so saved sessions keep
their model after a key is removed.

The web default is Claude Opus 5.5 on the direct Anthropic API, or
``ML_INTERN_DEFAULT_MODEL_ID`` when that names a picker model. A default that
needs a direct-provider key which isn't set falls back to GLM 5.2, so a new
chat never fails just because the key is missing, and the default is always
a visible model. Env vars are read on every call, never at import, so the
answer follows the current environment.
"""

import functools
import logging
import os
from typing import Any

from agent.core.model_ids import (
    CLAUDE_FABLE_52_MODEL_ID,
    CLAUDE_HAIKU_55_MODEL_ID,
    CLAUDE_OPUS_48_MODEL_ID,
    CLAUDE_OPUS_55_MODEL_ID,
    CLAUDE_SONNET_55_MODEL_ID,
    DIRECT_PROVIDER_API_KEY_ENV,
    GLM_52_MODEL_ID,
    GPT_55_DIRECT_MODEL_ID,
    GPT_55_MODEL_ID,
    GPT_61_TERRA_MODEL_ID,
    GPT_62_ASTRA_MODEL_ID,
    GPT_62_SOL_MODEL_ID,
    GPT_6_LUNA_MODEL_ID,
    KIMI_K27_CODE_MODEL_ID,
    MINIMAX_M3_MODEL_ID,
    direct_provider_for,
    strip_huggingface_model_prefix,
)

logger = logging.getLogger(__name__)

DEFAULT_MODEL_ENV = "ML_INTERN_DEFAULT_MODEL_ID"
BUILTIN_DEFAULT_MODEL_ID = CLAUDE_OPUS_55_MODEL_ID
KEYLESS_FALLBACK_MODEL_ID = GLM_52_MODEL_ID

# Curated dropdown for this fork, in display order: HF Router models plus
# direct-API options, matching agent/core/llm_params.py routing. Claude Opus
# 4.8, GPT-5.5, Kimi, MiniMax and GLM go through the HF Router (HF token).
# Claude Opus/Fable/Sonnet/Haiku 5.x call api.anthropic.com with
# ANTHROPIC_API_KEY, "GPT-5.5 (OpenAI key)" and the GPT-6.x models call
# api.openai.com with OPENAI_API_KEY, and DeepSeek uses the direct
# ``deepseek/`` provider (DEEPSEEK_API_KEY). A direct model that the HF Router
# also offers gets a "(<Provider> key)" label suffix. "recommended" marks the
# built-in default, Claude Opus 5.5. Entries stay {id, label[, recommended]}:
# clients derive who pays from the id.
AVAILABLE_MODELS: list[dict[str, Any]] = [
    {
        "id": CLAUDE_OPUS_48_MODEL_ID,
        "label": "Claude Opus 4.8",
    },
    {
        "id": CLAUDE_OPUS_55_MODEL_ID,
        "label": "Claude Opus 5.5",
        "recommended": True,
    },
    {
        "id": CLAUDE_FABLE_52_MODEL_ID,
        "label": "Claude Fable 5.2",
    },
    {
        "id": CLAUDE_SONNET_55_MODEL_ID,
        "label": "Claude Sonnet 5.5",
    },
    {
        "id": CLAUDE_HAIKU_55_MODEL_ID,
        "label": "Claude Haiku 5.5",
    },
    {
        "id": GPT_55_MODEL_ID,
        "label": "GPT-5.5",
    },
    {
        "id": GPT_55_DIRECT_MODEL_ID,
        "label": "GPT-5.5 (OpenAI key)",
    },
    {
        "id": GPT_62_ASTRA_MODEL_ID,
        "label": "GPT-6.2 Astra",
    },
    {
        "id": GPT_62_SOL_MODEL_ID,
        "label": "GPT-6.2 Sol",
    },
    {
        "id": GPT_61_TERRA_MODEL_ID,
        "label": "GPT-6.1 Terra",
    },
    {
        "id": GPT_6_LUNA_MODEL_ID,
        "label": "GPT-6 Luna",
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
        "id": GLM_52_MODEL_ID,
        "label": "GLM 5.2",
    },
    # DeepSeek via the direct ``deepseek/`` provider (DEEPSEEK_API_KEY →
    # api.deepseek.com), matching agent/core/llm_params.py routing. Distinct
    # from the HF-router DeepSeek id so the user's own DeepSeek key is used.
    {
        "id": "deepseek/deepseek-v4-pro",
        "label": "DeepSeek V4 Pro",
    },
    {
        "id": "deepseek/deepseek-v4-flash",
        "label": "DeepSeek V4 Flash",
    },
]

# Every picker id, shown or not: requests validate against the full list.
PICKER_MODEL_IDS = frozenset(model["id"] for model in AVAILABLE_MODELS)


def missing_direct_provider_key_env(model_id: str | None) -> str | None:
    """Name the unset API-key env var a direct-provider model needs, if any.

    Returns None for HF Router / local models and when the key is set. An
    empty or whitespace-only value counts as unset. Only the variable's name
    is ever returned, never its value.
    """
    provider = direct_provider_for(model_id)
    if provider is None:
        return None
    env_name = DIRECT_PROVIDER_API_KEY_ENV[provider]
    if (os.environ.get(env_name) or "").strip():
        return None
    return env_name


def visible_picker_models() -> list[dict[str, Any]]:
    """The picker entries the web app shows right now, in display order.

    HF Router entries are always listed. A direct-provider entry is listed
    only while its API key is set, read on every call like the default.
    Returns new dicts, so callers can't change ``AVAILABLE_MODELS``.
    """
    return [
        dict(model)
        for model in AVAILABLE_MODELS
        if missing_direct_provider_key_env(model["id"]) is None
    ]


@functools.lru_cache(maxsize=None)
def _warn_invalid_default_once(requested: str) -> None:
    # Called on every default lookup (page loads, health checks), so each bad
    # value is logged only the first time it is seen.
    logger.warning(
        "%s=%r is not a model picker id; using %r",
        DEFAULT_MODEL_ENV,
        requested,
        BUILTIN_DEFAULT_MODEL_ID,
    )


def configured_default_model_id() -> str:
    """``ML_INTERN_DEFAULT_MODEL_ID`` if it is a picker id, else Claude Opus 5.5.

    A leading ``huggingface/`` is accepted, as session model ids strip it too.
    """
    requested = (os.environ.get(DEFAULT_MODEL_ENV) or "").strip()
    normalized = strip_huggingface_model_prefix(requested)
    if normalized in PICKER_MODEL_IDS:
        return normalized
    if requested:
        _warn_invalid_default_once(requested)
    return BUILTIN_DEFAULT_MODEL_ID


def resolve_default_model_id() -> str:
    """The model the web app uses when nobody picked one.

    The configured default, unless it is a direct-provider model whose API key
    is unset right now: then GLM 5.2 on the HF Router.
    """
    model_id = configured_default_model_id()
    missing_env = missing_direct_provider_key_env(model_id)
    if missing_env is None:
        return model_id
    logger.debug(
        "Default model %r needs %s; using %r",
        model_id,
        missing_env,
        KEYLESS_FALLBACK_MODEL_ID,
    )
    return KEYLESS_FALLBACK_MODEL_ID
