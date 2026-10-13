import asyncio
from types import SimpleNamespace

import httpx
import pytest
from litellm import ChatCompletionMessageToolCall, Message

from agent.core.agent_loop import (
    LLMResult,
    _assistant_message_from_result,
    _call_llm_streaming,
    _strip_thinking_state_from_messages,
)


def test_assistant_message_from_result_keeps_content_and_tool_calls():
    tool_call = ChatCompletionMessageToolCall(
        id="call_1",
        type="function",
        function={"name": "bash", "arguments": '{"command": "date"}'},
    )
    result = LLMResult(
        content="working",
        tool_calls_acc={},
        token_count=12,
        finish_reason="tool_calls",
    )

    message = _assistant_message_from_result(result, tool_calls=[tool_call])

    assert message.content == "working"
    assert message.tool_calls == [tool_call]
    assert getattr(message, "thinking_blocks", None) is None
    assert getattr(message, "reasoning_content", None) is None


def test_strip_thinking_state_from_saved_messages():
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "stale"},
                {"type": "text", "text": "done"},
            ],
            "thinking_blocks": [{"type": "thinking", "thinking": "stale"}],
            "reasoning_content": "stale",
            "provider_specific_fields": {
                "thinking_blocks": [{"type": "thinking", "thinking": "stale"}],
                "reasoning_content": "stale",
                "other": "kept",
            },
        }
    ]

    stripped = _strip_thinking_state_from_messages(messages)

    assert stripped == 5
    assert messages == [
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "done"}],
            "provider_specific_fields": {"other": "kept"},
        }
    ]


@pytest.mark.asyncio
async def test_streaming_call_returns_wire_safe_result(monkeypatch):
    async def fake_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="done", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=3))

    async def fake_acompletion(**_kwargs):
        return fake_stream()

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="anthropic/claude-opus-4.8:fal-ai"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "openai/anthropic/claude-opus-4.8:fal-ai"},
    )

    assert result.content == "done"
    assert result.token_count == 3


@pytest.mark.asyncio
async def test_streaming_call_retries_read_timeout_before_output(monkeypatch):
    async def timeout_stream():
        raise httpx.ReadTimeout("Timeout on reading data from socket")
        yield  # pragma: no cover

    async def success_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="ok", tool_calls=None),
                    finish_reason="stop",
                )
            ],
        )
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=2))

    attempts = 0

    async def fake_acompletion(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return timeout_stream()
        return success_stream()

    events = []

    async def send_event(event):
        events.append(event)

    sleeps = []

    async def fake_sleep(delay):
        sleeps.append(delay)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="MiniMaxAI/MiniMax-M3:novita"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)
    monkeypatch.setattr("agent.core.agent_loop.asyncio.sleep", fake_sleep)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "openai/MiniMaxAI/MiniMax-M3:novita"},
    )

    assert attempts == 2
    assert sleeps == [5]
    assert result.content == "ok"
    assert [event.event_type for event in events] == [
        "tool_log",
        "assistant_chunk",
        "llm_call",
    ]


@pytest.mark.asyncio
async def test_streaming_call_does_not_retry_after_partial_output(monkeypatch):
    async def partial_timeout_stream():
        yield SimpleNamespace(
            choices=[
                SimpleNamespace(
                    delta=SimpleNamespace(content="partial", tool_calls=None),
                    finish_reason=None,
                )
            ],
        )
        raise httpx.ReadTimeout("Timeout on reading data from socket")

    attempts = 0

    async def fake_acompletion(**_kwargs):
        nonlocal attempts
        attempts += 1
        return partial_timeout_stream()

    events = []

    async def send_event(event):
        events.append(event)

    session = SimpleNamespace(
        config=SimpleNamespace(model_name="MiniMaxAI/MiniMax-M3:novita"),
        is_cancelled=False,
        send_event=send_event,
    )
    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)

    with pytest.raises(httpx.ReadTimeout):
        await _call_llm_streaming(
            session,
            messages=[Message(role="user", content="hi")],
            tools=[],
            llm_params={"model": "openai/MiniMaxAI/MiniMax-M3:novita"},
        )

    assert attempts == 1
    assert [event.event_type for event in events] == [
        "assistant_chunk",
        "assistant_stream_end",
        "llm_call",
    ]


@pytest.mark.asyncio
async def test_streaming_call_stops_retry_when_cancelled_during_delay(monkeypatch):
    async def timeout_stream():
        raise httpx.ReadTimeout("Timeout on reading data from socket")
        yield  # pragma: no cover

    attempts = 0

    async def fake_acompletion(**_kwargs):
        nonlocal attempts
        attempts += 1
        return timeout_stream()

    events = []

    class CancellableSession:
        def __init__(self):
            self.config = SimpleNamespace(model_name="MiniMaxAI/MiniMax-M3:novita")
            self._cancelled = asyncio.Event()

        @property
        def is_cancelled(self):
            return self._cancelled.is_set()

        def cancel(self):
            self._cancelled.set()

        async def send_event(self, event):
            events.append(event)
            if event.event_type == "tool_log":
                self.cancel()

    session = CancellableSession()
    monkeypatch.setattr("agent.core.agent_loop.acompletion", fake_acompletion)

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params={"model": "openai/MiniMaxAI/MiniMax-M3:novita"},
    )

    assert attempts == 1
    assert result.content is None
    assert session.is_cancelled is True
    assert [event.event_type for event in events] == ["tool_log"]


# ── direct Anthropic history repair (copies only) ──────────────────────

SIGNED_THINKING = {"type": "thinking", "thinking": "", "signature": "SIG=="}


def _repair(items):
    from agent.core.thinking_history import repair_direct_anthropic_history

    return repair_direct_anthropic_history(items)


def _call(call_id, name="bash", arguments='{"command": "ls"}'):
    return ChatCompletionMessageToolCall(
        id=call_id, type="function", function={"name": name, "arguments": arguments}
    )


def _thinking_assistant(content=None, tool_calls=None):
    return Message(
        role="assistant",
        content=content,
        tool_calls=tool_calls,
        thinking_blocks=[dict(SIGNED_THINKING)],
        reasoning_content="private",
        provider_specific_fields={
            "thinking_blocks": [dict(SIGNED_THINKING)],
            "reasoning_content": "private",
            "other": "kept",
        },
    )


def _tool_result(call_id, content, name="bash"):
    return Message(role="tool", content=content, tool_call_id=call_id, name=name)


def _assert_thinking_intact(message):
    assert message.thinking_blocks == [SIGNED_THINKING]
    assert message.reasoning_content == "private"
    assert message.provider_specific_fields["thinking_blocks"] == [SIGNED_THINKING]


def _assert_no_thinking(message):
    assert getattr(message, "thinking_blocks", None) is None
    assert getattr(message, "reasoning_content", None) is None
    fields = getattr(message, "provider_specific_fields", None) or {}
    assert "thinking_blocks" not in fields
    assert "reasoning_content" not in fields


def test_repair_closes_open_tool_exchange_as_plain_text():
    items = [
        Message(role="system", content="system"),
        Message(role="user", content="task"),
        _thinking_assistant("Checking.", [_call("toolu_1")]),
        _tool_result("toolu_1", "file.txt"),
    ]

    repaired = _repair(items)

    assert repaired is not items
    assert [m.role for m in repaired] == ["system", "user", "assistant", "user"]
    assert repaired[0] is items[0]
    assert repaired[1] is items[1]
    assistant, results = repaired[2], repaired[3]
    assert assistant.content == (
        'Checking.\n[called tool `bash` with arguments {"command": "ls"}]'
    )
    assert assistant.tool_calls is None
    _assert_no_thinking(assistant)
    assert (
        results.content
        == "[SYSTEM: tool results recorded as text]\n`bash` (call id toolu_1):\nfile.txt"
    )
    assert len(items) == 4
    assert items[2].tool_calls[0].id == "toolu_1"
    _assert_thinking_intact(items[2])


def test_repair_without_open_exchange_only_strips_thinking():
    items = [
        Message(role="system", content="system"),
        Message(role="user", content="task"),
        _thinking_assistant("Done."),
    ]

    repaired = _repair(items)

    assert [m.role for m in repaired] == ["system", "user", "assistant"]
    assert repaired[2] is not items[2]
    assert repaired[2].content == "Done."
    _assert_no_thinking(repaired[2])
    assert repaired[2].provider_specific_fields == {"other": "kept"}
    _assert_thinking_intact(items[2])


def test_repair_renders_parallel_calls_in_call_order():
    calls = [
        _call("toolu_1"),
        _call("toolu_2", name="read", arguments='{"path": "a.txt"}'),
    ]
    items = [
        Message(role="user", content="task"),
        _thinking_assistant(None, calls),
        _tool_result("toolu_2", "contents of a", name="read"),
        _tool_result("toolu_1", "a.txt"),
    ]

    repaired = _repair(items)

    assert [m.role for m in repaired] == ["user", "assistant", "user"]
    assert repaired[1].content == (
        '[called tool `bash` with arguments {"command": "ls"}]\n'
        '[called tool `read` with arguments {"path": "a.txt"}]'
    )
    assert repaired[2].content == (
        "[SYSTEM: tool results recorded as text]\n"
        "`bash` (call id toolu_1):\na.txt\n\n"
        "`read` (call id toolu_2):\ncontents of a"
    )


def test_repair_marks_missing_tool_results():
    calls = [_call("toolu_1"), _call("toolu_2", name="read", arguments="")]
    items = [
        Message(role="user", content="task"),
        _thinking_assistant(None, calls),
        _tool_result("toolu_1", "a.txt"),
    ]

    repaired = _repair(items)

    assert repaired[1].content == (
        '[called tool `bash` with arguments {"command": "ls"}]\n'
        "[called tool `read` with arguments {}]"
    )
    assert repaired[2].content == (
        "[SYSTEM: tool results recorded as text]\n"
        "`bash` (call id toolu_1):\na.txt\n\n"
        "`read` (call id toolu_2):\n[no result recorded]"
    )


def test_repair_closes_exchange_with_no_results_at_all():
    items = [
        Message(role="user", content="task"),
        _thinking_assistant("Hm.", [_call("toolu_1")]),
    ]

    repaired = _repair(items)

    assert [m.role for m in repaired] == ["user", "assistant", "user"]
    assert repaired[2].content == (
        "[SYSTEM: tool results recorded as text]\n`bash` (call id toolu_1):\n[no result recorded]"
    )


def test_repair_keeps_user_messages_after_the_open_exchange():
    items = [
        Message(role="user", content="task"),
        _thinking_assistant(None, [_call("toolu_1")]),
        _tool_result("toolu_1", "a.txt"),
        Message(role="user", content="please continue"),
    ]

    repaired = _repair(items)

    assert [m.role for m in repaired] == ["user", "assistant", "user", "user"]
    assert repaired[3] is items[3]
    assert "a.txt" in repaired[2].content


def test_repair_leaves_closed_tool_exchanges_alone_except_thinking():
    first = _thinking_assistant(None, [_call("toolu_1")])
    items = [
        Message(role="system", content="system"),
        Message(role="user", content="task"),
        first,
        _tool_result("toolu_1", "a.txt"),
        _thinking_assistant("Found a.txt."),
    ]

    repaired = _repair(items)

    assert [m.role for m in repaired] == [
        "system",
        "user",
        "assistant",
        "tool",
        "assistant",
    ]
    assert repaired[2].tool_calls == first.tool_calls
    assert repaired[3] is items[3]
    _assert_no_thinking(repaired[2])
    _assert_no_thinking(repaired[4])
    _assert_thinking_intact(first)


def test_repair_strips_thinking_content_blocks_from_dict_messages():
    message = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "stale", "signature": "S"},
            {"type": "text", "text": "done"},
        ],
        "thinking_blocks": [{"type": "thinking", "thinking": "stale"}],
    }

    repaired = _repair([{"role": "user", "content": "hi"}, message])

    assert repaired[1] == {
        "role": "assistant",
        "content": [{"type": "text", "text": "done"}],
    }
    assert "thinking_blocks" in message


def test_plain_text_transcript_renders_roles_calls_and_outputs():
    from agent.core.thinking_history import plain_text_transcript

    transcript = plain_text_transcript(
        [
            Message(role="user", content="task"),
            _thinking_assistant("Checking.", [_call("toolu_1")]),
            _tool_result("toolu_1", "file.txt"),
        ]
    )

    assert transcript == (
        "USER:\ntask\n\n"
        'ASSISTANT:\nChecking.\n[called tool `bash` with arguments {"command": "ls"}]'
        "\n\nTOOL RESULT `bash` (call id toolu_1):\nfile.txt"
    )
    assert "SIG==" not in transcript
