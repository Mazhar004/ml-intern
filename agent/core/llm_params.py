"""LiteLLM kwargs resolution for the model ids this agent accepts.

Kept separate from ``agent_loop`` so tools (research, context compaction, etc.)
can import it without pulling in the whole agent loop / tool router and
creating circular imports.
"""

import os

from agent.core.hf_tokens import resolve_hf_router_token
from agent.core.local_models import (
    LOCAL_MODEL_API_KEY_DEFAULT,
    LOCAL_MODEL_API_KEY_ENV,
    LOCAL_MODEL_BASE_URL_ENV,
    is_reserved_local_model_id,
    local_model_name,
    local_model_provider,
)
from agent.core.model_ids import (
    ANTHROPIC_API_BASE_URL,
    DIRECT_PROVIDER_API_KEY_ENV,
    HF_ROUTER_BASE_URL,
    is_direct_anthropic_model_id,
    strip_huggingface_model_prefix,
)


def _resolve_hf_router_token(session_hf_token: str | None = None) -> str | None:
    """Backward-compatible private wrapper used by tests and older imports."""
    return resolve_hf_router_token(session_hf_token)


# Effort levels accepted on the wire.
# HF Router exposes reasoning controls through the OpenAI-compatible
# ``extra_body`` field. The probe cascade walks down when a provider rejects
# an accepted-looking value, so this stays intentionally small and generic.
_HF_EFFORTS = {"low", "medium", "high"}


def _hf_router_effort_level(reasoning_effort: str) -> str:
    level = "low" if reasoning_effort == "minimal" else reasoning_effort
    return level


class UnsupportedEffortError(ValueError):
    """The requested effort isn't valid for this provider's API surface.

    Raised synchronously before any network call so the probe cascade can
    skip levels the provider can't accept (e.g. ``max`` on HF router).
    """


class MissingProviderKeyError(RuntimeError):
    """A direct-provider model was requested but its API key isn't set.

    Raised before any network call: an empty ``ANTHROPIC_API_KEY`` would
    otherwise make LiteLLM send an unauthenticated request (opaque 401).
    """


_ANTHROPIC_API_KEY_ENV = DIRECT_PROVIDER_API_KEY_ENV["anthropic"]
_MISSING_ANTHROPIC_KEY_MESSAGE = (
    f"Claude via the Anthropic API needs {_ANTHROPIC_API_KEY_ENV}. "
    "Add it to the repo .env and restart the backend."
)

# Direct Anthropic (Claude Opus/Sonnet 5.5): thinking is always on, effort
# goes in ``output_config.effort``. LiteLLM 1.83 rejects ``xhigh`` and ``max``
# locally for these model names (anthropic/chat/transformation.py), so they
# are clamped to the ceiling at runtime and skipped by the strict probe.
_DIRECT_ANTHROPIC_EFFORTS = frozenset({"low", "medium", "high"})
_DIRECT_ANTHROPIC_CLAMPED_EFFORTS = frozenset({"xhigh", "max"})
_DIRECT_ANTHROPIC_EFFORT_CEILING = "high"
# Explicit output budget (thinking + answer). Without it LiteLLM falls back to
# the model map's 128K, which is too large for non-streaming callers.
_DIRECT_ANTHROPIC_MAX_TOKENS = 32000


def _direct_anthropic_api_key() -> str:
    api_key = (os.environ.get(_ANTHROPIC_API_KEY_ENV) or "").strip()
    if not api_key:
        raise MissingProviderKeyError(_MISSING_ANTHROPIC_KEY_MESSAGE)
    return api_key


def _direct_anthropic_effort_level(reasoning_effort: str, strict: bool) -> str | None:
    level = _hf_router_effort_level(reasoning_effort)
    if level in _DIRECT_ANTHROPIC_EFFORTS:
        return level
    if strict:
        raise UnsupportedEffortError(
            f"Direct Anthropic API (LiteLLM) doesn't accept effort={level!r}"
        )
    if level in _DIRECT_ANTHROPIC_CLAMPED_EFFORTS:
        return _DIRECT_ANTHROPIC_EFFORT_CEILING
    return None


def _resolve_direct_anthropic_params(
    model_name: str,
    reasoning_effort: str | None,
    strict: bool,
) -> dict:
    # api_base is explicit so an inherited ANTHROPIC_BASE_URL can't redirect
    # the key. Never send reasoning_effort (LiteLLM turns it into a rejected
    # thinking budget), thinking (always on), extra_body (sent literally as a
    # body key), or sampling params (non-default values are rejected).
    params: dict = {
        "model": model_name,
        "api_base": ANTHROPIC_API_BASE_URL,
        "api_key": _direct_anthropic_api_key(),
        "max_tokens": _DIRECT_ANTHROPIC_MAX_TOKENS,
    }
    if reasoning_effort:
        level = _direct_anthropic_effort_level(reasoning_effort, strict)
        if level is not None:
            params["output_config"] = {"effort": level}
    return params


def _local_api_base(base_url: str) -> str:
    base = base_url.strip().rstrip("/")
    if base.endswith("/v1"):
        return base
    return f"{base}/v1"


def _resolve_local_model_params(
    model_name: str,
    reasoning_effort: str | None = None,
    strict: bool = False,
) -> dict:
    if reasoning_effort and strict:
        raise UnsupportedEffortError(
            "Local OpenAI-compatible endpoints don't accept reasoning_effort"
        )

    local_name = local_model_name(model_name)
    if local_name is None:
        raise ValueError(f"Unsupported local model id: {model_name}")

    provider = local_model_provider(model_name)
    assert provider is not None
    raw_base = (
        os.environ.get(provider["base_url_env"])
        or os.environ.get(LOCAL_MODEL_BASE_URL_ENV)
        or provider["base_url_default"]
    )
    api_key = (
        os.environ.get(provider["api_key_env"])
        or os.environ.get(LOCAL_MODEL_API_KEY_ENV)
        or LOCAL_MODEL_API_KEY_DEFAULT
    )
    return {
        "model": f"openai/{local_name}",
        "api_base": _local_api_base(raw_base),
        "api_key": api_key,
    }


def _resolve_llm_params(
    model_name: str,
    session_hf_token: str | None = None,
    reasoning_effort: str | None = None,
    strict: bool = False,
) -> dict:
    """
    Build LiteLLM kwargs for a given model id.

    • ``ollama/<model>``, ``vllm/<model>``, ``lm_studio/<model>``, and
      ``llamacpp/<model>`` — local OpenAI-compatible endpoints. The id prefix
      selects a configurable localhost base URL, and the model suffix is sent
      to LiteLLM as ``openai/<model>``. These endpoints don't receive
      ``reasoning_effort``.

    • ``deepseek/<model>`` — LiteLLM's native DeepSeek provider with
      ``DEEPSEEK_API_KEY``. ``high`` / ``max`` go via ``extra_body``.

    • ``anthropic/claude-opus-5-5`` / ``anthropic/claude-sonnet-5-5`` (exact
      raw ids, see ``is_direct_anthropic_model_id``) — LiteLLM's native
      Anthropic provider at ``https://api.anthropic.com`` with
      ``ANTHROPIC_API_KEY``. The id is passed verbatim with an explicit
      ``api_base``/``api_key`` and ``max_tokens=32000``. Effort becomes
      ``output_config={"effort": level}``: "minimal" → "low"; low/medium/high
      pass through; xhigh/max raise ``UnsupportedEffortError`` when strict and
      clamp to "high" otherwise; no effort → no ``output_config``. Never emits
      ``reasoning_effort``, ``thinking``, ``extra_body`` or sampling params.
      An empty key raises ``MissingProviderKeyError``. The HF Router Claude
      (``anthropic/claude-opus-4.8:fal-ai``) and ``huggingface/``-prefixed
      ids are NOT direct and fall through to the router.

    • Anything else is treated as an HF Router id. We hit the auto-routing
      OpenAI-compatible endpoint at ``https://router.huggingface.co/v1``.
      The id can be bare or carry an HF routing suffix (``:fastest`` /
      ``:cheapest`` / ``:<provider>``). A leading ``huggingface/`` is
      stripped. ``reasoning_effort`` is forwarded via ``extra_body``.
      "minimal" normalizes to "low".

    ``strict=True`` raises ``UnsupportedEffortError`` when the requested
    effort isn't in the provider's accepted set, instead of silently
    dropping it. The probe cascade uses strict mode so it can walk down
    (``max`` → ``xhigh`` → ``high`` …) without making an API call. Regular
    runtime callers leave ``strict=False``, so a stale cached effort
    can't crash a turn — it just doesn't get sent.

    Token precedence for HF-router calls (first non-empty wins):
      1. session.hf_token — the user's own token (CLI / OAuth / cache file).
      2. huggingface_hub cache — ``HF_TOKEN`` / ``HUGGING_FACE_HUB_TOKEN`` /
         local ``hf auth login`` cache.
    """
    normalized_model = strip_huggingface_model_prefix(model_name) or model_name

    if is_reserved_local_model_id(normalized_model):
        raise ValueError(f"Unsupported local model id: {normalized_model}")

    if local_model_provider(normalized_model) is not None:
        return _resolve_local_model_params(normalized_model, reasoning_effort, strict)

    if model_name.startswith("deepseek/"):
        # LiteLLM's ``deepseek/`` provider picks up ``DEEPSEEK_API_KEY`` and
        # routes to https://api.deepseek.com (OpenAI-compatible). V4 models
        # (deepseek-v4-pro, deepseek-v4-flash) default to thinking-on, which
        # populates ``reasoning_content`` on assistant messages. DeepSeek's API
        # *requires* that ``reasoning_content`` be echoed back on the next
        # turn or it returns 400 — the agent loop's reasoning capture +
        # provider-scoped scrubber (``agent_loop._scrub_reasoning_for_provider``)
        # handles that round-trip, so thinking is left enabled here. Without
        # thinking the model hallucinates completion in long agentic loops.
        params: dict = {"model": model_name}
        if reasoning_effort in {"high", "max"}:
            # DeepSeek docs: thinking pairs with ``reasoning_effort`` of
            # ``high`` or ``max``. Pass through when supplied; otherwise
            # the model uses its default.
            params["extra_body"] = {"reasoning_effort": reasoning_effort}
        return params

    if is_direct_anthropic_model_id(model_name):
        return _resolve_direct_anthropic_params(model_name, reasoning_effort, strict)

    hf_model = normalized_model
    api_key = _resolve_hf_router_token(session_hf_token)
    params = {
        "model": f"openai/{hf_model}",
        "api_base": HF_ROUTER_BASE_URL,
        "api_key": api_key,
    }
    if reasoning_effort:
        hf_level = _hf_router_effort_level(reasoning_effort)
        if hf_level not in _HF_EFFORTS:
            if strict:
                raise UnsupportedEffortError(
                    f"HF Router doesn't accept effort={hf_level!r}"
                )
        else:
            params["extra_body"] = {"reasoning_effort": hf_level}
    return params
