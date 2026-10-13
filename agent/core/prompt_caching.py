"""Prompt-cache helpers for HF Router FAL requests.

The HF Router/OpenRouter path uses provider-native prompt caching. Anthropic
models keep explicit JSON ``cache_control`` content blocks for compatibility,
and also need the top-level ``cache_control`` hint on the OpenAI-compatible HF
Router path; the explicit markers alone are accepted there but do not produce
cache writes. OpenAI models cache eligible prefixes automatically and accept
routing/retention hints in the body.
Headers like ``X-OpenRouter-Cache`` control response caching, not prompt
caching through this route.

The direct Anthropic API route (``is_direct_anthropic_model_id``) only uses
block-level ``cache_control`` breakpoints on outgoing copies: the last tool,
the system prompt, and the newest tool/user message plus the last one of the
previous request, so long tool loops keep hitting the cache. Anthropic allows
at most 4 breakpoints. No HF Router hints (``extra_body``, session header)
are sent there: LiteLLM's native Anthropic provider would send ``extra_body``
as a literal body key, and the HF session id doesn't belong at Anthropic.
"""

from typing import Any

from agent.core.model_ids import HF_ROUTER_BASE_URL, is_direct_anthropic_model_id

_CACHE_CONTROL = {"type": "ephemeral"}
_CACHEABLE_ROLES = {"system", "user"}
_DIRECT_ANTHROPIC_MESSAGE_ROLES = {"user", "tool"}
_ANTHROPIC_MAX_CACHE_BREAKPOINTS = 4
_HF_ROUTER_SESSION_ID_MAX_LENGTH = 256
HF_ROUTER_SESSION_ID_HEADER = "X-HF-Session-id"


def router_session_id_for(session: Any) -> str | None:
    """Return the usage-window-scoped Router session ID for a runtime session."""
    billing_session_id = getattr(session, "inference_billing_session_id", None)
    if isinstance(billing_session_id, str) and billing_session_id:
        return billing_session_id
    session_id = getattr(session, "session_id", None)
    if isinstance(session_id, str) and session_id:
        return session_id
    return None


def _is_hf_router_request(llm_params: dict[str, Any]) -> bool:
    api_base = str(llm_params.get("api_base") or "").rstrip("/")
    return api_base == HF_ROUTER_BASE_URL


def _is_fal_router_request(llm_params: dict[str, Any]) -> bool:
    return _is_hf_router_request(llm_params) and ":fal" in _router_model(llm_params)


def _router_model(llm_params: dict[str, Any]) -> str:
    model = str(llm_params.get("model") or "")
    return model.removeprefix("openai/")


def _is_direct_anthropic_request(llm_params: dict[str, Any]) -> bool:
    return is_direct_anthropic_model_id(str(llm_params.get("model") or ""))


def _uses_explicit_cache_control(llm_params: dict[str, Any]) -> bool:
    if not _is_fal_router_request(llm_params):
        return False
    return _router_model(llm_params).startswith("anthropic/")


def _is_openai_gpt55(llm_params: dict[str, Any]) -> bool:
    if not _is_fal_router_request(llm_params):
        return False
    return _router_model(llm_params).startswith("openai/gpt-5.5")


def _merge_extra_body(
    llm_params: dict[str, Any], updates: dict[str, Any]
) -> dict[str, Any]:
    if not updates:
        return llm_params

    cached_params = dict(llm_params)
    extra_body = dict(cached_params.get("extra_body") or {})
    extra_body.update(updates)
    cached_params["extra_body"] = extra_body
    return cached_params


def _merge_extra_headers(
    llm_params: dict[str, Any], updates: dict[str, str]
) -> dict[str, Any]:
    if not updates:
        return llm_params

    cached_params = dict(llm_params)
    extra_headers = dict(cached_params.get("extra_headers") or {})
    extra_headers.update(updates)
    cached_params["extra_headers"] = extra_headers
    return cached_params


def with_prompt_cache_params(
    llm_params: dict[str, Any],
    *,
    session_id: str | None = None,
) -> dict[str, Any]:
    """Return LiteLLM params with provider-native prompt-cache body hints."""
    if _is_direct_anthropic_request(llm_params):
        # Block-level breakpoints only (``with_prompt_caching``).
        return llm_params

    updates: dict[str, Any] = {}
    headers: dict[str, str] = {}
    if session_id and _is_hf_router_request(llm_params):
        stable_session_id = session_id[:_HF_ROUTER_SESSION_ID_MAX_LENGTH]
        headers[HF_ROUTER_SESSION_ID_HEADER] = stable_session_id
        if _is_openai_gpt55(llm_params):
            updates["prompt_cache_key"] = stable_session_id

    if _uses_explicit_cache_control(llm_params):
        updates["cache_control"] = dict(_CACHE_CONTROL)

    if _is_openai_gpt55(llm_params):
        updates["prompt_cache_retention"] = "24h"

    return _merge_extra_headers(_merge_extra_body(llm_params, updates), headers)


def _message_role(message: Any) -> str | None:
    if isinstance(message, dict):
        role = message.get("role")
    else:
        role = getattr(message, "role", None)
    return role if isinstance(role, str) else None


def _message_content(message: Any) -> Any:
    if isinstance(message, dict):
        return message.get("content")
    return getattr(message, "content", None)


def _message_to_dict(message: Any) -> dict[str, Any]:
    if isinstance(message, dict):
        return dict(message)
    if hasattr(message, "model_dump"):
        return message.model_dump(exclude_none=True)
    raise TypeError(f"Unsupported message type for prompt caching: {type(message)!r}")


def _has_cacheable_text(content: Any) -> bool:
    if isinstance(content, str):
        return bool(content)
    if not isinstance(content, list):
        return False
    return any(
        isinstance(block, dict)
        and block.get("type") == "text"
        and isinstance(block.get("text"), str)
        and bool(block.get("text"))
        for block in content
    )


def _cache_target_index(messages: list[Any]) -> int | None:
    if len(messages) < 2:
        return None

    for idx in range(len(messages) - 2, -1, -1):
        message = messages[idx]
        if _message_role(message) not in _CACHEABLE_ROLES:
            continue
        if _has_cacheable_text(_message_content(message)):
            return idx
    return None


def _content_with_cache_control(content: Any) -> list[dict[str, Any]]:
    if isinstance(content, str):
        return [
            {"type": "text", "text": content, "cache_control": dict(_CACHE_CONTROL)}
        ]

    blocks = [dict(block) if isinstance(block, dict) else block for block in content]
    for idx in range(len(blocks) - 1, -1, -1):
        block = blocks[idx]
        if (
            isinstance(block, dict)
            and block.get("type") == "text"
            and isinstance(block.get("text"), str)
            and bool(block.get("text"))
        ):
            cached = dict(block)
            cached["cache_control"] = dict(_CACHE_CONTROL)
            blocks[idx] = cached
            break
    return blocks


def _tools_with_cache_control(tools: list[dict] | None) -> list[dict] | None:
    if not tools:
        return tools

    cached_tools = list(tools)
    last_tool = dict(cached_tools[-1])
    last_tool["cache_control"] = dict(_CACHE_CONTROL)
    cached_tools[-1] = last_tool
    return cached_tools


def _message_with_cache_control(message: Any) -> dict[str, Any]:
    cached_message = _message_to_dict(message)
    cached_message["content"] = _content_with_cache_control(
        cached_message.get("content")
    )
    return cached_message


def _direct_anthropic_message_with_cache_control(message: Any) -> dict[str, Any]:
    if _message_role(message) != "tool":
        return _message_with_cache_control(message)
    # LiteLLM moves a tool message's own cache_control onto the Anthropic
    # tool_result block, so the content can stay as-is.
    cached_message = _message_to_dict(message)
    cached_message["cache_control"] = dict(_CACHE_CONTROL)
    return cached_message


def _is_cacheable_message(message: Any, roles: set[str]) -> bool:
    return _message_role(message) in roles and _has_cacheable_text(
        _message_content(message)
    )


def _last_index_before(messages: list[Any], end: int, predicate) -> int | None:
    for idx in range(end - 1, -1, -1):
        if predicate(messages[idx]):
            return idx
    return None


def _first_system_index(messages: list[Any]) -> int | None:
    for idx, message in enumerate(messages):
        if _message_role(message) == "system":
            return idx if _has_cacheable_text(_message_content(message)) else None
    return None


def _direct_anthropic_cache_targets(messages: list[Any], budget: int) -> list[int]:
    """Message indexes to mark, in priority order, capped at ``budget``.

    1. The system prompt (stable prefix).
    2. The newest user/tool message: writes the cache for this request.
    3. The last user/tool message before the newest assistant turn, which is
       where the previous request ended: guarantees a read hit even when a
       turn added more blocks than Anthropic's 20-block lookback.
    """

    def is_turn_input(message: Any) -> bool:
        return _is_cacheable_message(message, _DIRECT_ANTHROPIC_MESSAGE_ROLES)

    def is_assistant(message: Any) -> bool:
        return _message_role(message) == "assistant"

    targets: list[int] = []
    system_idx = _first_system_index(messages)
    if system_idx is not None:
        targets.append(system_idx)

    newest_idx = _last_index_before(messages, len(messages), is_turn_input)
    if newest_idx is not None:
        targets.append(newest_idx)
        assistant_idx = _last_index_before(messages, newest_idx, is_assistant)
        if assistant_idx is not None:
            previous_idx = _last_index_before(messages, assistant_idx, is_turn_input)
            if previous_idx is not None:
                targets.append(previous_idx)
    return targets[:budget]


def _with_direct_anthropic_caching(
    messages: list[Any],
    tools: list[dict] | None,
) -> tuple[list[Any], list[dict] | None]:
    cached_tools = _tools_with_cache_control(tools)
    tool_breakpoints = 1 if cached_tools else 0
    targets = _direct_anthropic_cache_targets(
        messages, _ANTHROPIC_MAX_CACHE_BREAKPOINTS - tool_breakpoints
    )
    if not targets:
        return messages, cached_tools

    cached_messages = list(messages)
    for idx in targets:
        cached_messages[idx] = _direct_anthropic_message_with_cache_control(
            messages[idx]
        )
    return cached_messages, cached_tools


def with_prompt_caching(
    messages: list[Any],
    tools: list[dict] | None,
    llm_params: dict[str, Any],
) -> tuple[list[Any], list[dict] | None]:
    """Return outgoing messages with explicit cache breakpoints when needed.

    The newest message is treated as dynamic. For Anthropic FAL models, the
    cache breakpoint is placed on the closest earlier system/user text block so
    provider-side caching covers the stable prefix without changing persisted
    conversation history. The final tool spec is also marked so stable tool
    definitions are cached.

    Direct Anthropic API models get up to 4 breakpoints instead (see
    ``_direct_anthropic_cache_targets``). Marked messages are always copies;
    assistant turns (and their signed thinking blocks) are never touched.
    """
    if _is_direct_anthropic_request(llm_params):
        return _with_direct_anthropic_caching(messages, tools)

    if not _uses_explicit_cache_control(llm_params):
        return messages, tools

    cached_tools = _tools_with_cache_control(tools)
    idx = _cache_target_index(messages)
    if idx is None:
        return messages, cached_tools

    cached_messages = list(messages)
    cached_messages[idx] = _message_with_cache_control(messages[idx])
    return cached_messages, cached_tools
