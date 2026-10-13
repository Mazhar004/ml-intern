"""Thinking-safe history rewrites for the direct Anthropic API.

Claude models on the direct API always think. Signed thinking blocks must be
echoed back unchanged while a tool-use loop continues, and editing or dropping
earlier turns invalidates every later signature. When the history has to change
anyway (compaction, a rejected signature), the safe shape is plain text:
no thinking anywhere and no open tool call at the tail, because an assistant
``tool_use`` turn that is continued without its thinking block is rejected.

Every function here returns new objects and never modifies its input.
"""

import copy
import json
from typing import Any

from litellm import Message

_THINKING_CONTENT_TYPES = frozenset({"thinking", "redacted_thinking"})
_THINKING_FIELDS = ("thinking_blocks", "reasoning_content")
_NO_RESULT = "[no result recorded]"

# User messages starting with this prefix are internal nudges: the web UI
# hides them and undo/edit do not count them as user turns. The tool results
# of a closed exchange use it so they never render as a user bubble.
INTERNAL_NUDGE_PREFIX = "[SYSTEM:"
_TOOL_RESULTS_HEADER = f"{INTERNAL_NUDGE_PREFIX} tool results recorded as text]"


def message_field(message: Any, name: str) -> Any:
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)


def is_internal_nudge(message: Any) -> bool:
    """True for injected user messages the UI hides (see INTERNAL_NUDGE_PREFIX)."""
    content = message_field(message, "content")
    return isinstance(content, str) and content.lstrip().startswith(
        INTERNAL_NUDGE_PREFIX
    )


def copy_message(message: Any, updates: dict[str, Any]) -> Any:
    """Return a copy of ``message`` with ``updates`` applied (never in place).

    For dict messages a ``None`` update removes the key, so it is not sent as
    an explicit ``null``.
    """
    if isinstance(message, dict):
        merged = {**message, **updates}
        return {k: v for k, v in merged.items() if not (k in updates and v is None)}
    try:
        return message.model_copy(update=updates)
    except AttributeError:
        cleaned = copy.copy(message)
        for name, value in updates.items():
            setattr(cleaned, name, value)
        return cleaned


def without_thinking_blocks(message: Any) -> Any:
    """Return an assistant message without Anthropic thinking blocks.

    Covers both the top-level ``thinking_blocks`` field and the copy LiteLLM
    keeps in ``provider_specific_fields``. Copies only when something changes.
    """
    if message_field(message, "role") != "assistant":
        return message
    updates: dict[str, Any] = {}
    if message_field(message, "thinking_blocks") is not None:
        updates["thinking_blocks"] = None
    provider_fields = message_field(message, "provider_specific_fields")
    if isinstance(provider_fields, dict) and "thinking_blocks" in provider_fields:
        kept = {k: v for k, v in provider_fields.items() if k != "thinking_blocks"}
        updates["provider_specific_fields"] = kept or None
    return copy_message(message, updates) if updates else message


def _without_any_thinking(message: Any) -> Any:
    """Assistant copy without thinking blocks, reasoning text or thinking content."""
    if message_field(message, "role") != "assistant":
        return message
    updates: dict[str, Any] = {
        name: None
        for name in _THINKING_FIELDS
        if message_field(message, name) is not None
    }
    provider_fields = message_field(message, "provider_specific_fields")
    if isinstance(provider_fields, dict) and any(
        name in provider_fields for name in _THINKING_FIELDS
    ):
        kept = {k: v for k, v in provider_fields.items() if k not in _THINKING_FIELDS}
        updates["provider_specific_fields"] = kept or None
    content = message_field(message, "content")
    if isinstance(content, list) and any(_is_thinking_part(p) for p in content):
        updates["content"] = [p for p in content if not _is_thinking_part(p)]
    return copy_message(message, updates) if updates else message


def _is_thinking_part(part: Any) -> bool:
    return isinstance(part, dict) and part.get("type") in _THINKING_CONTENT_TYPES


def content_text(content: Any) -> str:
    """Plain text of a message ``content`` (string or list of parts)."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = [_part_text(part) for part in content if not _is_thinking_part(part)]
        return "\n".join(text for text in texts if text)
    return str(content)


def _part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if isinstance(part, dict):
        if part.get("type") == "text":
            return part.get("text") or ""
        return f"[{part.get('type', 'non-text')} content omitted]"
    return str(part)


def _tool_call_parts(tool_call: Any) -> tuple[str, str, str]:
    """``(id, name, arguments)`` of a tool call object or dict."""
    function = message_field(tool_call, "function")
    arguments = message_field(function, "arguments") if function is not None else None
    if not isinstance(arguments, str):
        arguments = json.dumps(arguments) if arguments else ""
    return (
        message_field(tool_call, "id") or "",
        (message_field(function, "name") if function is not None else None) or "",
        arguments.strip() or "{}",
    )


def render_tool_call(tool_call: Any) -> str:
    _, name, arguments = _tool_call_parts(tool_call)
    return f"[called tool `{name}` with arguments {arguments}]"


def _render_tool_output(name: str, call_id: str, output: str) -> str:
    return f"`{name}` (call id {call_id}):\n{output}"


def _assistant_text(message: Any) -> str:
    """Assistant text with every tool call it made rendered as a line."""
    lines = [content_text(message_field(message, "content"))]
    lines.extend(
        render_tool_call(tc) for tc in message_field(message, "tool_calls") or []
    )
    return "\n".join(line for line in lines if line)


def _open_exchange_start(items: list[Any]) -> int | None:
    """Index of the last assistant turn when it still has tool calls.

    Only tool results and user messages can follow it, so the next request
    would continue that tool-use loop.
    """
    for idx in range(len(items) - 1, -1, -1):
        if message_field(items[idx], "role") == "assistant":
            return idx if message_field(items[idx], "tool_calls") else None
    return None


def _tool_results_text(assistant: Any, results: list[Any]) -> str:
    """One entry per call (in call order), then any results without a call."""
    by_id = {message_field(r, "tool_call_id"): r for r in results}
    entries = []
    for tool_call in message_field(assistant, "tool_calls") or []:
        call_id, name, _ = _tool_call_parts(tool_call)
        result = by_id.pop(call_id, None)
        output = (
            _NO_RESULT
            if result is None
            else content_text(message_field(result, "content"))
        )
        entries.append(_render_tool_output(name, call_id, output))
    for call_id, result in by_id.items():
        name = message_field(result, "name") or ""
        output = content_text(message_field(result, "content"))
        entries.append(_render_tool_output(name, call_id or "", output))
    return f"{_TOOL_RESULTS_HEADER}\n" + "\n\n".join(entries)


def _closed_exchange(items: list[Any], start: int) -> list[Any]:
    """Rewrite ``items[start:]`` (an open tool exchange) as plain text."""
    assistant = items[start]
    tail = items[start + 1 :]
    results = [m for m in tail if message_field(m, "role") == "tool"]
    others = [m for m in tail if message_field(m, "role") != "tool"]
    return [
        Message(role="assistant", content=_assistant_text(assistant)),
        Message(role="user", content=_tool_results_text(assistant, results)),
        *others,
    ]


def repair_direct_anthropic_history(items: list[Any]) -> list[Any]:
    """Thinking-free copy of a history that is safe to send after any edit.

    1. An open tool exchange at the tail (the last assistant turn has tool
       calls and only its tool results / user messages follow) becomes an
       assistant text turn naming each call plus a user turn with the outputs.
    2. Thinking blocks and reasoning text are dropped from every assistant
       turn. Closed tool exchanges keep their tool calls and results.

    Unchanged messages are reused as-is; nothing in ``items`` is modified.
    """
    start = _open_exchange_start(items)
    rewritten = (
        list(items)
        if start is None
        else [*items[:start], *_closed_exchange(items, start)]
    )
    return [_without_any_thinking(m) for m in rewritten]


def plain_text_transcript(messages: list[Any]) -> str:
    """Role-labelled plain-text rendering of ``messages`` (no thinking)."""
    sections = []
    for message in messages:
        role = message_field(message, "role")
        if role == "assistant":
            sections.append(f"ASSISTANT:\n{_assistant_text(message)}")
        elif role == "tool":
            name = message_field(message, "name") or ""
            call_id = message_field(message, "tool_call_id") or ""
            output = content_text(message_field(message, "content"))
            sections.append(f"TOOL RESULT {_render_tool_output(name, call_id, output)}")
        else:
            text = content_text(message_field(message, "content"))
            sections.append(f"{str(role).upper()}:\n{text}")
    return "\n\n".join(sections)
