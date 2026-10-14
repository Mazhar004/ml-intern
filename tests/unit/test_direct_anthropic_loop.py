"""Direct Anthropic route through the agent loop, compaction and research.

Everything here is offline. The end-to-end cases patch ``httpx.AsyncClient.send``
(the layer under both LiteLLM's Anthropic handler and the OpenAI SDK used for
the HF Router), return canned Anthropic SSE / OpenAI chunks, and fail loudly on
any request to an unexpected host. Keys are dummies set with monkeypatch.
"""

import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest
from litellm import ChatCompletionMessageToolCall, Message

from agent.config import Config
from agent.context_manager import manager as context_manager
from agent.context_manager.manager import ContextManager, summarize_messages
from agent.core import agent_loop
from agent.core.llm_params import MissingProviderKeyError
from agent.core.agent_loop import (
    Handlers,
    LLMResult,
    _accumulate_thinking_blocks,
    _assistant_message_from_result,
    _call_llm_non_streaming,
    _call_llm_streaming,
    _friendly_error_message,
    _is_invalid_thinking_signature_error,
    _scrub_reasoning_for_provider,
)
from agent.core.model_ids import (
    ANTHROPIC_API_BASE_URL,
    CLAUDE_OPUS_48_MODEL_ID,
    CLAUDE_OPUS_55_MODEL_ID,
)
from agent.core.session import Session
from agent.tools import research_tool

DIRECT_MODEL = CLAUDE_OPUS_55_MODEL_ID
ROUTER_MODEL = CLAUDE_OPUS_48_MODEL_ID
ROUTER_LITELLM_MODEL = f"openai/{ROUTER_MODEL}"
DUMMY_KEY = "sk-ant-test"
DUMMY_HF_TOKEN = "hf_test_dummy"
EMPTY_SIGNED_THINKING = {"type": "thinking", "thinking": "", "signature": "SIG_EMPTY=="}
MISSING_KEY_MESSAGE = (
    "Claude via the Anthropic API needs ANTHROPIC_API_KEY. "
    "Add it to the repo .env and restart the backend."
)


# ── fixtures and helpers ────────────────────────────────────────────────


def _direct_params(model_name: str = DIRECT_MODEL) -> dict:
    """The llm_params shape ``_resolve_llm_params`` returns for direct ids."""
    return {
        "model": model_name,
        "api_base": ANTHROPIC_API_BASE_URL,
        "api_key": DUMMY_KEY,
        "max_tokens": 32000,
    }


@pytest.fixture
def anthropic_env(monkeypatch):
    """Dummy Anthropic key; LiteLLM's beta-header allowlist stays local."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")


def _sse(events: list[dict]) -> bytes:
    return "".join(
        f"event: {event['type']}\ndata: {json.dumps(event)}\n\n" for event in events
    ).encode()


def _anthropic_stream(blocks: list[list[dict]], stop_reason: str) -> list[dict]:
    """Anthropic SSE events; ``blocks`` holds start/delta events per block."""
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "claude-opus-5-5",
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 100, "output_tokens": 1},
            },
        }
    ]
    for index, block_events in enumerate(blocks):
        events.extend({**event, "index": index} for event in block_events)
        events.append({"type": "content_block_stop", "index": index})
    events.append(
        {
            "type": "message_delta",
            "delta": {"stop_reason": stop_reason, "stop_sequence": None},
            "usage": {"output_tokens": 20},
        }
    )
    events.append({"type": "message_stop"})
    return events


def _signed_empty_thinking_block() -> list[dict]:
    return [
        {
            "type": "content_block_start",
            "content_block": {"type": "thinking", "thinking": "", "signature": ""},
        },
        {
            "type": "content_block_delta",
            "delta": {"type": "signature_delta", "signature": "SIG_EMPTY=="},
        },
    ]


def _bash_tool_use_block() -> list[dict]:
    return [
        {
            "type": "content_block_start",
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "bash",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "delta": {"type": "input_json_delta", "partial_json": '{"command": "ls"}'},
        },
    ]


def _text_block(text: str) -> list[dict]:
    return [
        {"type": "content_block_start", "content_block": {"type": "text", "text": ""}},
        {"type": "content_block_delta", "delta": {"type": "text_delta", "text": text}},
    ]


def _openai_stream(
    text: str | None, *, finish_reason: str = "stop", model: str = ROUTER_MODEL
) -> bytes:
    def chunk(delta, finish_reason=None):
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    usage_chunk = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [],
        "usage": {"prompt_tokens": 50, "completion_tokens": 2, "total_tokens": 52},
    }
    first_delta = {"role": "assistant"}
    if text is not None:
        first_delta["content"] = text
    payloads = [chunk(first_delta), chunk({}, finish_reason), usage_chunk]
    lines = [f"data: {json.dumps(payload)}\n\n" for payload in payloads]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


class FakeProviderHttp:
    """Queue of canned responses keyed by host; records every request body."""

    def __init__(self):
        self.requests: list[dict] = []
        self._queue: list[tuple[str, int, bytes, str]] = []

    def queue_anthropic_sse(self, events: list[dict]) -> None:
        self._queue.append(
            ("api.anthropic.com", 200, _sse(events), "text/event-stream")
        )

    def queue_anthropic_error(self, status: int, error: dict) -> None:
        body = json.dumps({"type": "error", "error": error}).encode()
        self._queue.append(("api.anthropic.com", status, body, "application/json"))

    def queue_anthropic_json(self, text: str) -> None:
        """A non-streaming (summary) reply: signed empty thinking + text."""
        body = {
            "id": "msg_s",
            "type": "message",
            "role": "assistant",
            "model": "claude-opus-5-5",
            "content": [
                {"type": "thinking", "thinking": "", "signature": "SIG_SUMMARY=="},
                {"type": "text", "text": text},
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        self._queue.append(
            ("api.anthropic.com", 200, json.dumps(body).encode(), "application/json")
        )

    def queue_router_stream(self, text: str) -> None:
        self.queue_openai_stream("router.huggingface.co", text)

    def queue_openai_stream(
        self,
        host: str,
        text: str | None,
        *,
        finish_reason: str = "stop",
        model: str = ROUTER_MODEL,
    ) -> None:
        stream = _openai_stream(text, finish_reason=finish_reason, model=model)
        self._queue.append((host, 200, stream, "text/event-stream"))

    def bodies_for(self, host: str) -> list[dict]:
        return [r["body"] for r in self.requests if r["host"] == host]

    async def send(self, _client, request, *_args, **_kwargs):
        host = request.url.host
        self.requests.append({"host": host, "body": json.loads(request.content)})
        if not self._queue:
            raise AssertionError(f"Unexpected request to {host}")
        expected_host, status, content, content_type = self._queue.pop(0)
        if host != expected_host:
            raise AssertionError(f"Expected a request to {expected_host}, got {host}")
        return httpx.Response(
            status,
            content=content,
            headers={"content-type": content_type},
            request=request,
        )


@pytest.fixture
def provider_http(monkeypatch):
    fake = FakeProviderHttp()

    async def send(client, request, *args, **kwargs):
        return await fake.send(client, request, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return fake


class BashToolRouter:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def get_tool_specs_for_llm(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "bash",
                    "description": "Run a shell command",
                    "parameters": {
                        "type": "object",
                        "properties": {"command": {"type": "string"}},
                        "required": ["command"],
                    },
                },
            }
        ]

    async def call_tool(self, name, arguments, session=None, tool_call_id=None):
        self.calls.append((name, arguments))
        return "file.txt", True


def _make_session(model_name: str = DIRECT_MODEL, *, stream: bool = True) -> Session:
    config = Config.model_validate({"model_name": model_name, "save_sessions": False})
    return Session(
        asyncio.Queue(),
        config,
        tool_router=BashToolRouter(),
        stream=stream,
        hf_token=DUMMY_HF_TOKEN,
        hf_username="tester",
    )


def _drain_events(session: Session) -> list:
    events = []
    while not session.event_queue.empty():
        events.append(session.event_queue.get_nowait())
    return events


def _field(message, name: str):
    if isinstance(message, dict):
        return message.get(name)
    return getattr(message, name, None)


def _role(message) -> str:
    return _field(message, "role")


def _error_events(events: list) -> list[str]:
    return [e.data["error"] for e in events if e.event_type == "error"]


def _assistant_with_thinking(content=None) -> Message:
    return Message(
        role="assistant",
        content=content,
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="toolu_1",
                type="function",
                function={"name": "bash", "arguments": '{"command": "ls"}'},
            )
        ],
        thinking_blocks=[dict(EMPTY_SIGNED_THINKING)],
    )


def _delta_chunk(*, thinking_blocks=None, content=None, finish_reason=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    tool_calls=None,
                    thinking_blocks=thinking_blocks,
                ),
                finish_reason=finish_reason,
            )
        ],
    )


async def _noop_send_event(_event):
    return None


# ── streamed thinking accumulation ─────────────────────────────────────


def test_accumulate_thinking_blocks_merges_fragments_and_keeps_signature_only_blocks():
    deltas = [
        [{"type": "thinking", "thinking": "first ", "signature": ""}],
        [{"type": "thinking", "thinking": "part", "signature": ""}],
        [{"type": "thinking", "thinking": "", "signature": "SIG_one"}],
        [{"type": "thinking", "thinking": "", "signature": "SIG_two"}],
        [{"type": "redacted_thinking", "data": "RD"}],
    ]

    blocks: list[dict] = []
    for delta in deltas:
        blocks = _accumulate_thinking_blocks(blocks, delta)

    assert blocks == [
        {"type": "thinking", "thinking": "first part", "signature": "SIG_one"},
        {"type": "thinking", "thinking": "", "signature": "SIG_two"},
        {"type": "redacted_thinking", "data": "RD"},
    ]


def test_accumulate_thinking_blocks_starts_new_block_after_a_signature():
    blocks = [{"type": "thinking", "thinking": "done", "signature": "SIG_A"}]

    result = _accumulate_thinking_blocks(
        blocks, [{"type": "thinking", "thinking": "next", "signature": ""}]
    )

    assert result == [
        {"type": "thinking", "thinking": "done", "signature": "SIG_A"},
        {"type": "thinking", "thinking": "next", "signature": ""},
    ]
    assert blocks == [{"type": "thinking", "thinking": "done", "signature": "SIG_A"}]


def test_accumulate_thinking_blocks_ignores_missing_deltas():
    blocks = [{"type": "thinking", "thinking": "a", "signature": ""}]

    assert _accumulate_thinking_blocks(blocks, None) == blocks


async def test_streaming_call_returns_signature_only_thinking_block(monkeypatch):
    async def fake_stream():
        yield _delta_chunk(thinking_blocks=[dict(EMPTY_SIGNED_THINKING)], content="")
        yield _delta_chunk(content="Done.", finish_reason="stop")
        yield SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=5))

    async def fake_acompletion(**_kwargs):
        return fake_stream()

    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    session = SimpleNamespace(
        config=SimpleNamespace(model_name=DIRECT_MODEL),
        is_cancelled=False,
        send_event=_noop_send_event,
    )

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params=_direct_params(),
    )

    assert result.thinking_blocks == [EMPTY_SIGNED_THINKING]
    assert result.content == "Done."


async def test_streaming_call_drops_unsigned_thinking_fragments(monkeypatch):
    async def fake_stream():
        yield _delta_chunk(
            thinking_blocks=[{"type": "thinking", "thinking": "half", "signature": ""}]
        )
        yield _delta_chunk(content="Done.", finish_reason="stop")

    async def fake_acompletion(**_kwargs):
        return fake_stream()

    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    session = SimpleNamespace(
        config=SimpleNamespace(model_name=DIRECT_MODEL),
        is_cancelled=False,
        send_event=_noop_send_event,
    )

    result = await _call_llm_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params=_direct_params(),
    )

    assert result.thinking_blocks is None


def _non_streaming_response(message) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="tool_calls")],
        usage=SimpleNamespace(total_tokens=9),
    )


@pytest.mark.parametrize(
    "message",
    [
        SimpleNamespace(
            content=None,
            tool_calls=None,
            thinking_blocks=[dict(EMPTY_SIGNED_THINKING)],
            provider_specific_fields=None,
        ),
        SimpleNamespace(
            content=None,
            tool_calls=None,
            thinking_blocks=None,
            provider_specific_fields={"thinking_blocks": [dict(EMPTY_SIGNED_THINKING)]},
        ),
    ],
    ids=["message_field", "provider_specific_fields"],
)
async def test_non_streaming_call_returns_thinking_blocks(monkeypatch, message):
    async def fake_acompletion(**_kwargs):
        return _non_streaming_response(message)

    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    session = SimpleNamespace(
        config=SimpleNamespace(model_name=DIRECT_MODEL),
        is_cancelled=False,
        send_event=_noop_send_event,
    )

    result = await _call_llm_non_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params=_direct_params(),
    )

    assert result.thinking_blocks == [EMPTY_SIGNED_THINKING]


@pytest.mark.parametrize(
    ("llm_params", "expected_max_tokens"),
    [
        (_direct_params(), 16_000),
        ({**_direct_params(), "max_tokens": 8_000}, 8_000),
        ({"model": ROUTER_LITELLM_MODEL}, None),
    ],
    ids=["direct_capped", "direct_smaller_kept", "router_untouched"],
)
async def test_non_streaming_call_caps_max_tokens_only_for_direct_anthropic(
    monkeypatch, llm_params, expected_max_tokens
):
    captured: list[dict] = []

    async def fake_acompletion(**kwargs):
        captured.append(kwargs)
        return _non_streaming_response(
            SimpleNamespace(content="ok", tool_calls=None, thinking_blocks=None)
        )

    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    session = SimpleNamespace(
        config=SimpleNamespace(model_name=DIRECT_MODEL),
        is_cancelled=False,
        send_event=_noop_send_event,
    )

    await _call_llm_non_streaming(
        session,
        messages=[Message(role="user", content="hi")],
        tools=[],
        llm_params=llm_params,
    )

    assert captured[0].get("max_tokens") == expected_max_tokens


# ── storing assistant turns ────────────────────────────────────────────


def _tool_call() -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id="toolu_1",
        type="function",
        function={"name": "bash", "arguments": '{"command": "ls"}'},
    )


def test_direct_anthropic_assistant_message_stores_thinking_verbatim_with_none_content():
    result = LLMResult(
        content="",
        tool_calls_acc={},
        token_count=10,
        finish_reason="tool_calls",
        thinking_blocks=[dict(EMPTY_SIGNED_THINKING)],
    )

    message = _assistant_message_from_result(
        result, model_name=DIRECT_MODEL, tool_calls=[_tool_call()]
    )

    assert message.content is None
    assert message.thinking_blocks == [EMPTY_SIGNED_THINKING]
    assert message.thinking_blocks is not result.thinking_blocks


@pytest.mark.parametrize(
    "model_name", [ROUTER_LITELLM_MODEL, "deepseek/deepseek-v4-pro", None]
)
def test_non_direct_assistant_message_never_stores_thinking_blocks(model_name):
    result = LLMResult(
        content="working",
        tool_calls_acc={},
        token_count=10,
        finish_reason="tool_calls",
        thinking_blocks=[dict(EMPTY_SIGNED_THINKING)],
    )

    message = _assistant_message_from_result(
        result, model_name=model_name, tool_calls=[_tool_call()]
    )

    assert getattr(message, "thinking_blocks", None) is None
    assert message.content == "working"


# ── outgoing scrub ─────────────────────────────────────────────────────


def _history_with_thinking() -> list[Message]:
    assistant = _assistant_with_thinking()
    assistant.provider_specific_fields = {
        "thinking_blocks": [dict(EMPTY_SIGNED_THINKING)],
        "other": "kept",
    }
    return [
        Message(role="user", content="list files"),
        assistant,
        Message(role="tool", content="file.txt", tool_call_id="toolu_1", name="bash"),
    ]


def test_scrub_for_router_target_drops_thinking_without_touching_history():
    history = _history_with_thinking()

    outgoing = _scrub_reasoning_for_provider(history, ROUTER_LITELLM_MODEL)

    assert getattr(outgoing[1], "thinking_blocks", None) is None
    assert outgoing[1].provider_specific_fields == {"other": "kept"}
    assert outgoing[1].tool_calls == history[1].tool_calls
    assert history[1].thinking_blocks == [EMPTY_SIGNED_THINKING]
    assert "thinking_blocks" in history[1].provider_specific_fields


def test_scrub_for_router_target_handles_dict_messages():
    history = [
        {"role": "user", "content": "hi"},
        {
            "role": "assistant",
            "content": "ok",
            "thinking_blocks": [dict(EMPTY_SIGNED_THINKING)],
        },
    ]

    outgoing = _scrub_reasoning_for_provider(history, ROUTER_LITELLM_MODEL)

    assert outgoing[1] == {"role": "assistant", "content": "ok"}
    assert "thinking_blocks" in history[1]


def test_scrub_for_direct_anthropic_target_keeps_thinking_verbatim():
    history = _history_with_thinking()

    outgoing = _scrub_reasoning_for_provider(history, DIRECT_MODEL)

    assert outgoing[1].thinking_blocks == [EMPTY_SIGNED_THINKING]


def test_scrub_for_deepseek_drops_thinking_and_keeps_reasoning_echo():
    history = _history_with_thinking()

    outgoing = _scrub_reasoning_for_provider(history, "deepseek/deepseek-v4-pro")

    assert getattr(outgoing[1], "thinking_blocks", None) is None
    assert outgoing[1].reasoning_content == ""
    assert history[1].thinking_blocks == [EMPTY_SIGNED_THINKING]


def test_scrub_returns_same_list_when_nothing_changes():
    history = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="ok"),
    ]

    assert _scrub_reasoning_for_provider(history, ROUTER_LITELLM_MODEL) is history


# ── friendly errors ────────────────────────────────────────────────────


def _provider_error(text: str, provider: str) -> Exception:
    error = Exception(text)
    error.llm_provider = provider
    return error


def test_anthropic_auth_error_names_anthropic_api_key():
    error = _provider_error(
        'litellm.AuthenticationError: AnthropicException - {"type":"error",'
        '"error":{"type":"authentication_error","message":"invalid x-api-key"}}',
        "anthropic",
    )

    message = _friendly_error_message(error, model_name=DIRECT_MODEL)

    assert message == (
        "Anthropic rejected the API key: ANTHROPIC_API_KEY is missing or invalid. "
        "Set it in the repo .env and restart the backend."
    )


def test_anthropic_credit_error_points_to_console_billing_not_hf():
    error = _provider_error(
        "litellm.BadRequestError: AnthropicException - Your credit balance is too "
        "low to access the Anthropic API. Please go to Plans & Billing to upgrade "
        "or purchase credits.",
        "anthropic",
    )

    message = _friendly_error_message(error, user_plan="free", model_name=DIRECT_MODEL)

    assert message is not None
    assert "Anthropic Console" in message
    assert "huggingface" not in message.lower()
    assert "HF PRO" not in message


def test_direct_model_auth_error_without_provider_attribute_uses_anthropic_wording():
    error = Exception("Unauthorized: invalid x-api-key")

    message = _friendly_error_message(error, model_name=DIRECT_MODEL)

    assert message is not None
    assert "ANTHROPIC_API_KEY" in message


def test_router_auth_error_keeps_hf_wording_even_in_direct_session():
    error = _provider_error(
        "litellm.AuthenticationError: OpenAIException - Unauthorized", "openai"
    )

    message = _friendly_error_message(error, model_name=DIRECT_MODEL)

    assert message is not None
    assert "Hugging Face token" in message
    assert "ANTHROPIC_API_KEY" not in message


def test_router_credit_error_keeps_hf_billing_wording():
    error = _provider_error("402 Payment Required: monthly credits exhausted", "openai")

    message = _friendly_error_message(error, user_plan="free", model_name=ROUTER_MODEL)

    assert message is not None
    assert "https://huggingface.co/settings/billing" in message


def test_missing_provider_key_error_message_is_shown_verbatim():
    error = MissingProviderKeyError(MISSING_KEY_MESSAGE)

    assert (
        _friendly_error_message(error, model_name=DIRECT_MODEL) == MISSING_KEY_MESSAGE
    )


# ── signature heal ─────────────────────────────────────────────────────


def test_signature_error_bound_to_a_different_conversation_is_detected():
    error = Exception(
        "litellm.BadRequestError: AnthropicException - messages.1.content.0: "
        "Invalid `signature` in `thinking` block: signature is bound to a "
        "different conversation"
    )

    assert _is_invalid_thinking_signature_error(error)
    assert not _is_invalid_thinking_signature_error(Exception("invalid tool name"))


async def test_second_signature_failure_after_strip_is_raised_not_looped(monkeypatch):
    attempts = 0

    async def fake_acompletion(**_kwargs):
        nonlocal attempts
        attempts += 1
        raise Exception("Invalid `signature` in `thinking` block")

    events = []

    async def send_event(event):
        events.append(event)

    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    session = SimpleNamespace(
        config=SimpleNamespace(model_name=DIRECT_MODEL),
        is_cancelled=False,
        send_event=send_event,
    )

    with pytest.raises(Exception, match="Invalid `signature`"):
        await _call_llm_streaming(
            session,
            messages=[Message(role="user", content="hi"), _assistant_with_thinking()],
            tools=[],
            llm_params=_direct_params(),
        )

    assert attempts == 2
    assert [e.event_type for e in events] == ["tool_log"]


async def test_heal_on_the_last_attempt_raises_instead_of_returning_none(monkeypatch):
    errors = [
        Exception("503 service unavailable"),
        Exception("503 service unavailable"),
        Exception("Invalid `signature` in `thinking` block"),
    ]

    async def fake_acompletion(**_kwargs):
        raise errors.pop(0)

    async def no_sleep(_session, _delay):
        return False

    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    monkeypatch.setattr(agent_loop, "_sleep_for_retry_or_cancel", no_sleep)
    session = SimpleNamespace(
        config=SimpleNamespace(model_name=DIRECT_MODEL),
        is_cancelled=False,
        send_event=_noop_send_event,
    )

    with pytest.raises(Exception, match="Invalid `signature`"):
        await _call_llm_non_streaming(
            session,
            messages=[Message(role="user", content="hi"), _assistant_with_thinking()],
            tools=[],
            llm_params=_direct_params(),
        )


@pytest.mark.parametrize(
    "error_text,heals",
    [
        ("Invalid `signature` in `thinking` block", True),
        ("thinking signature is invalid: bound to a different conversation", False),
    ],
    ids=["legacy-wording", "other-wording"],
)
async def test_router_signature_heal_keeps_head_matcher_and_in_place_strip(
    monkeypatch, error_text, heals
):
    sent: list[list] = []

    async def text_stream():
        yield _delta_chunk(content="ok", finish_reason="stop")

    async def fake_acompletion(**kwargs):
        sent.append(kwargs["messages"])
        if len(sent) == 1:
            raise Exception(error_text)
        return text_stream()

    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)
    session = SimpleNamespace(
        config=SimpleNamespace(model_name=ROUTER_MODEL),
        is_cancelled=False,
        send_event=_noop_send_event,
    )
    stale = {
        "role": "assistant",
        "content": [
            {"type": "thinking", "thinking": "stale", "signature": "S"},
            {"type": "text", "text": "done"},
        ],
    }
    call = _call_llm_streaming(
        session,
        messages=[{"role": "user", "content": "hi"}, stale],
        tools=[],
        llm_params={"model": ROUTER_LITELLM_MODEL},
    )

    if not heals:
        with pytest.raises(Exception, match="bound to a different conversation"):
            await call
        assert len(sent) == 1
        return
    result = await call
    assert result.content == "ok"
    assert len(sent) == 2
    assert stale["content"] == [{"type": "text", "text": "done"}]


# ── loop behaviour with a faked LLM call ───────────────────────────────


def _patch_llm_results(monkeypatch, results: list[LLMResult]) -> list:
    calls = []

    async def fake_call(session, messages, tools, llm_params):
        calls.append(list(messages))
        return results.pop(0)

    monkeypatch.setattr(agent_loop, "_call_llm_streaming", fake_call)
    monkeypatch.setattr(agent_loop, "_call_llm_non_streaming", fake_call)
    return calls


async def test_refusal_ends_turn_with_visible_error_and_skips_continuation_guard(
    monkeypatch, anthropic_env
):
    session = _make_session()
    session.current_plan = [{"id": "1", "content": "Do it", "status": "pending"}]
    calls = _patch_llm_results(
        monkeypatch,
        [
            LLMResult(
                content=None,
                tool_calls_acc={},
                token_count=10,
                finish_reason="content_filter",
                thinking_blocks=[dict(EMPTY_SIGNED_THINKING)],
            )
        ],
    )

    final = await Handlers.run_agent(session, "do the thing")

    events = _drain_events(session)
    assert final is None
    assert len(calls) == 1
    assert _error_events(events) == [
        "Claude declined this request (refusal stop reason). "
        "Rephrase the request or switch models to continue."
    ]
    assert [m.role for m in session.context_manager.items] == ["system", "user"]
    assert not any(e.event_type == "turn_complete" for e in events)


@pytest.mark.parametrize(
    "content,finish_reason",
    [("Partial answer.", "content_filter"), (None, "length")],
)
async def test_router_unproductive_stop_keeps_head_behaviour(
    monkeypatch, content, finish_reason
):
    session = _make_session(ROUTER_MODEL)
    monkeypatch.setattr(
        agent_loop,
        "_resolve_llm_params",
        lambda *_a, **_k: {"model": ROUTER_LITELLM_MODEL},
    )
    _patch_llm_results(
        monkeypatch,
        [
            LLMResult(
                content=content,
                tool_calls_acc={},
                token_count=10,
                finish_reason=finish_reason,
            )
        ],
    )

    final = await Handlers.run_agent(session, "do the thing")

    events = _drain_events(session)
    stored = ["system", "user", "assistant"] if content else ["system", "user"]
    assert final == content
    assert _error_events(events) == []
    assert any(e.event_type == "turn_complete" for e in events)
    assert [m.role for m in session.context_manager.items] == stored


async def test_length_stop_with_no_output_emits_visible_warning(
    monkeypatch, anthropic_env
):
    session = _make_session()
    _patch_llm_results(
        monkeypatch,
        [
            LLMResult(
                content=None,
                tool_calls_acc={},
                token_count=10,
                finish_reason="length",
                thinking_blocks=[dict(EMPTY_SIGNED_THINKING)],
            )
        ],
    )

    await Handlers.run_agent(session, "think hard")

    errors = _error_events(_drain_events(session))
    assert len(errors) == 1
    assert "output token limit" in errors[0]
    assert [m.role for m in session.context_manager.items] == ["system", "user"]


async def test_truncated_tool_turn_is_stored_without_its_thinking_blocks(
    monkeypatch, anthropic_env
):
    session = _make_session()
    session.config.max_iterations = 1
    _patch_llm_results(
        monkeypatch,
        [
            LLMResult(
                content="Writing the file now.",
                tool_calls_acc={
                    0: {
                        "id": "toolu_1",
                        "type": "function",
                        "function": {"name": "bash", "arguments": '{"comm'},
                    }
                },
                token_count=10,
                finish_reason="length",
                thinking_blocks=[dict(EMPTY_SIGNED_THINKING)],
            )
        ],
    )

    await Handlers.run_agent(session, "write a big file")

    assistant = session.context_manager.items[2]
    assert assistant.role == "assistant"
    assert assistant.content == "Writing the file now."
    assert getattr(assistant, "thinking_blocks", None) is None


async def test_missing_anthropic_key_during_turn_becomes_error_event(monkeypatch):
    session = _make_session()

    def missing_key(*_args, **_kwargs):
        raise MissingProviderKeyError(MISSING_KEY_MESSAGE)

    monkeypatch.setattr(agent_loop, "_resolve_llm_params", missing_key)

    await Handlers.run_agent(session, "hello")

    assert _error_events(_drain_events(session)) == [MISSING_KEY_MESSAGE]


# ── offline end-to-end through LiteLLM ─────────────────────────────────


async def _run_direct_tool_loop(session: Session, provider_http: FakeProviderHttp):
    provider_http.queue_anthropic_sse(
        _anthropic_stream(
            [_signed_empty_thinking_block(), _bash_tool_use_block()], "tool_use"
        )
    )
    provider_http.queue_anthropic_sse(
        _anthropic_stream([_text_block("Found file.txt.")], "end_turn")
    )
    return await Handlers.run_agent(session, "list files")


async def test_direct_anthropic_tool_loop_echoes_signed_thinking_before_tool_use(
    anthropic_env, provider_http
):
    session = _make_session()

    final = await _run_direct_tool_loop(session, provider_http)

    assert final == "Found file.txt."
    assert session.tool_router.calls == [("bash", {"command": "ls"})]
    second_body = provider_http.bodies_for("api.anthropic.com")[1]
    assistant_turn = second_body["messages"][1]
    assert assistant_turn["role"] == "assistant"
    assert [block["type"] for block in assistant_turn["content"]] == [
        "thinking",
        "tool_use",
    ]
    assert assistant_turn["content"][0] == EMPTY_SIGNED_THINKING
    assert assistant_turn["content"][1]["id"] == "toolu_1"
    assert assistant_turn["content"][1]["input"] == {"command": "ls"}
    for key in ("reasoning_effort", "thinking", "extra_body", "temperature"):
        assert key not in second_body


async def test_switching_to_router_model_sends_no_thinking_blocks(
    anthropic_env, provider_http
):
    session = _make_session()
    await _run_direct_tool_loop(session, provider_http)
    stored_assistant = session.context_manager.items[2]
    session.update_model(ROUTER_MODEL)
    provider_http.queue_router_stream("You're welcome.")

    final = await Handlers.run_agent(session, "thanks")

    assert final == "You're welcome."
    router_body = provider_http.bodies_for("router.huggingface.co")[0]
    serialized = json.dumps(router_body)
    assert "thinking_blocks" not in serialized
    assert "SIG_EMPTY" not in serialized
    assert router_body["model"] == ROUTER_MODEL
    assert stored_assistant.thinking_blocks == [EMPTY_SIGNED_THINKING]


async def test_direct_anthropic_refusal_stop_reason_surfaces_error(
    anthropic_env, provider_http
):
    session = _make_session()
    provider_http.queue_anthropic_sse(
        _anthropic_stream([_signed_empty_thinking_block()], "refusal")
    )

    final = await Handlers.run_agent(session, "something disallowed")

    errors = _error_events(_drain_events(session))
    assert final is None
    assert len(errors) == 1
    assert errors[0].startswith("Claude declined this request")
    assert len(provider_http.requests) == 1
    assert [m.role for m in session.context_manager.items] == ["system", "user"]


async def test_direct_anthropic_401_surfaces_anthropic_key_message(
    anthropic_env, provider_http
):
    session = _make_session()
    provider_http.queue_anthropic_error(
        401, {"type": "authentication_error", "message": "invalid x-api-key"}
    )

    await Handlers.run_agent(session, "hello")

    errors = _error_events(_drain_events(session))
    assert errors == [
        "Anthropic rejected the API key: ANTHROPIC_API_KEY is missing or invalid. "
        "Set it in the repo .env and restart the backend."
    ]
    assert len(provider_http.requests) == 1


# ── compaction ─────────────────────────────────────────────────────────


def _fake_summary_acompletion(captured: list[dict]):
    async def fake_acompletion(**kwargs):
        captured.append(kwargs)
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="summary"), finish_reason="stop"
                )
            ],
            usage=SimpleNamespace(completion_tokens=3),
        )

    return fake_acompletion


def _text_of(message) -> str:
    content = _field(message, "content")
    if isinstance(content, list):
        return "".join(part.get("text", "") for part in content)
    return content or ""


async def test_direct_anthropic_summary_sends_one_plain_text_user_turn(
    monkeypatch, anthropic_env
):
    captured: list[dict] = []
    monkeypatch.setattr(
        context_manager, "acompletion", _fake_summary_acompletion(captured)
    )
    history = [
        _assistant_with_thinking("checking"),
        Message(role="tool", content="file.txt", tool_call_id="toolu_1", name="bash"),
    ]

    summary, tokens = await summarize_messages(
        history, model_name=DIRECT_MODEL, max_tokens=100_000
    )

    sent = captured[0]
    (only,) = sent["messages"]
    text = _text_of(only)
    assert (summary, tokens) == ("summary", 3)
    assert sent["max_completion_tokens"] == 16_000
    assert _role(only) == "user"
    assert (
        'ASSISTANT:\nchecking\n[called tool `bash` with arguments {"command": "ls"}]'
        in text
    )
    assert "TOOL RESULT `bash` (call id toolu_1):\nfile.txt" in text
    assert text.endswith(context_manager._COMPACT_PROMPT)
    assert "SIG_EMPTY" not in json.dumps(sent["messages"])
    assert history[0].thinking_blocks == [EMPTY_SIGNED_THINKING]


async def test_router_summary_keeps_requested_tokens_and_message_order(
    monkeypatch,
):
    captured: list[dict] = []
    monkeypatch.setattr(
        context_manager, "acompletion", _fake_summary_acompletion(captured)
    )
    history = [Message(role="assistant", content="earlier reply")]

    await summarize_messages(
        history, model_name=ROUTER_MODEL, hf_token=DUMMY_HF_TOKEN, max_tokens=50_000
    )

    sent = captured[0]
    assert sent["max_completion_tokens"] == 50_000
    assert [_role(m) for m in sent["messages"]] == ["assistant", "user"]


def _context_manager_with_history(items: list[Message]) -> ContextManager:
    cm = ContextManager.__new__(ContextManager)
    cm.system_prompt = "system"
    cm.model_max_tokens = 100_000
    cm.compact_size = 1_000
    cm.running_context_usage = 95_000
    cm.untouched_messages = 3
    cm.items = items
    cm.on_message_added = None
    return cm


def _compaction_history() -> list[Message]:
    return [
        Message(role="system", content="system"),
        Message(role="user", content="task"),
        _assistant_with_thinking("old step"),
        Message(role="tool", content="old", tool_call_id="toolu_1", name="bash"),
        Message(role="user", content="next"),
        _assistant_with_thinking("new step"),
        Message(role="tool", content="new", tool_call_id="toolu_1", name="bash"),
    ]


async def _compact(
    cm: ContextManager, model_name: str, monkeypatch, token_counter=None
) -> None:
    async def fake_summarize(*_args, **_kwargs):
        return "summary", 3

    def fake_recompute(self, _model_name):
        self.running_context_usage = 1_000

    monkeypatch.setattr(context_manager, "summarize_messages", fake_summarize)
    monkeypatch.setattr(ContextManager, "_recompute_usage", fake_recompute)
    monkeypatch.setattr("litellm.token_counter", token_counter or (lambda **_kw: 100))
    await cm.compact(model_name=model_name, tool_specs=None, hf_token=None)


async def test_compaction_summary_branch_closes_open_tool_call_as_text(
    monkeypatch,
):
    history = _compaction_history()
    retained_original = history[5]
    cm = _context_manager_with_history(list(history))

    await _compact(cm, DIRECT_MODEL, monkeypatch)

    assert [m.role for m in cm.items] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert cm.items[3] is history[4]
    closed, results = cm.items[4], cm.items[5]
    assert closed.content == (
        'new step\n[called tool `bash` with arguments {"command": "ls"}]'
    )
    assert closed.tool_calls is None
    assert getattr(closed, "thinking_blocks", None) is None
    assert (
        results.content
        == "[SYSTEM: tool results recorded as text]\n`bash` (call id toolu_1):\nnew"
    )
    assert retained_original.thinking_blocks == [EMPTY_SIGNED_THINKING]
    assert retained_original.tool_calls[0].id == "toolu_1"


def _open_turn_history() -> list[Message]:
    return [
        Message(role="system", content="system"),
        Message(role="user", content="task"),
        _assistant_with_thinking("new step"),
        Message(role="tool", content="new", tool_call_id="toolu_1", name="bash"),
    ]


async def test_compaction_without_summary_keeps_open_turn_when_prefix_unchanged(
    monkeypatch,
):
    history = _open_turn_history()
    cm = _context_manager_with_history(list(history))

    await _compact(cm, DIRECT_MODEL, monkeypatch)

    assert cm.items == history
    assert all(new is old for new, old in zip(cm.items, history))
    assert cm.items[2].thinking_blocks == [EMPTY_SIGNED_THINKING]


@pytest.mark.parametrize(
    "oversized,repaired",
    [("task", True), ("new step", True), ("new", False)],
    ids=["first-user", "open-assistant", "trailing-tool-result"],
)
async def test_compaction_without_summary_repairs_only_when_a_signed_prefix_changed(
    monkeypatch, oversized, repaired
):
    history = _open_turn_history()
    cm = _context_manager_with_history(list(history))

    def token_counter(**kwargs):
        return 60_000 if kwargs["messages"][0]["content"] == oversized else 100

    await _compact(cm, DIRECT_MODEL, monkeypatch, token_counter)

    roles = [m.role for m in cm.items]
    if repaired:
        assert roles == ["system", "user", "assistant", "user"]
        assert cm.items[2].tool_calls is None
        assert getattr(cm.items[2], "thinking_blocks", None) is None
    else:
        assert roles == ["system", "user", "assistant", "tool"]
        assert cm.items[2] is history[2]
    assert history[2].thinking_blocks == [EMPTY_SIGNED_THINKING]


async def test_compaction_keeps_retained_items_untouched_for_router_model(
    monkeypatch,
):
    history = _compaction_history()
    cm = _context_manager_with_history(list(history))

    await _compact(cm, ROUTER_MODEL, monkeypatch)

    assert cm.items[4] is history[5]


# ── compaction / heal mid tool loop, end-to-end through LiteLLM ───────

MISSING_THINKING_400 = (
    "messages.1.content.0.type: Expected `thinking` or `redacted_thinking`, but "
    "found `tool_use`. When `thinking` is enabled, a final `assistant` message "
    "must start with a thinking block (preceeding the lastmost set of `tool_use` "
    "and `tool_result` blocks). We recommend you include thinking blocks from "
    "previous turns."
)
SIGNATURE_400 = (
    "messages.1.content.0: Invalid `signature` in `thinking` block: signature is "
    "bound to a different conversation"
)


def _over_threshold_stream(blocks: list[list[dict]], stop_reason: str) -> list[dict]:
    """Anthropic SSE whose usage is far over any compaction threshold."""
    events = _anthropic_stream(blocks, stop_reason)
    events[0]["message"]["usage"]["input_tokens"] = 5_000_000
    return events


def _block_types(message: dict) -> list[str]:
    content = message["content"]
    if isinstance(content, str):
        return ["text"]
    return [block["type"] for block in content]


def _assert_valid_continuation(body: dict) -> None:
    """Ends with a user turn; every assistant tool_use turn starts with thinking."""
    messages = body["messages"]
    assert messages[0]["role"] == "user"
    assert messages[-1]["role"] == "user"
    for message in messages:
        kinds = _block_types(message)
        if message["role"] == "assistant" and "tool_use" in kinds:
            assert kinds[0] in {"thinking", "redacted_thinking"}, kinds


def _record_added_messages(session: Session) -> list:
    added: list = []
    cm = session.context_manager
    previous = cm.on_message_added

    def record(message):
        added.append(message)
        if previous:
            previous(message)

    cm.on_message_added = record
    return added


async def test_compaction_with_summary_mid_tool_loop_sends_a_valid_request(
    anthropic_env, provider_http
):
    session = _make_session()
    cm = session.context_manager
    cm.untouched_messages = 2
    cm.items.extend(
        [
            Message(role="user", content="task one"),
            _assistant_with_thinking(),
            Message(role="tool", content="/", tool_call_id="toolu_1", name="bash"),
            Message(role="assistant", content="done one"),
        ]
    )
    provider_http.queue_anthropic_sse(
        _over_threshold_stream(
            [_signed_empty_thinking_block(), _bash_tool_use_block()], "tool_use"
        )
    )
    provider_http.queue_anthropic_json("SUMMARY")
    provider_http.queue_anthropic_sse(
        _anthropic_stream([_text_block("Found file.txt.")], "end_turn")
    )

    final = await Handlers.run_agent(session, "list files")

    bodies = provider_http.bodies_for("api.anthropic.com")
    summary_body, next_body = bodies[1], bodies[2]
    assert final == "Found file.txt."
    assert len(bodies) == 3
    assert [m["role"] for m in summary_body["messages"]] == ["user"]
    assert _block_types(summary_body["messages"][0]) == ["text"]
    assert "SIG_EMPTY" not in json.dumps(summary_body)
    _assert_valid_continuation(next_body)
    assert "file.txt" in json.dumps(next_body["messages"][-1])


async def test_compaction_without_summary_mid_tool_loop_echoes_open_turn_thinking(
    anthropic_env, provider_http
):
    session = _make_session()
    provider_http.queue_anthropic_sse(
        _over_threshold_stream(
            [_signed_empty_thinking_block(), _bash_tool_use_block()], "tool_use"
        )
    )
    provider_http.queue_anthropic_sse(
        _anthropic_stream([_text_block("Found file.txt.")], "end_turn")
    )

    final = await Handlers.run_agent(session, "list files")

    events = _drain_events(session)
    bodies = provider_http.bodies_for("api.anthropic.com")
    open_turn = [m for m in bodies[1]["messages"] if m["role"] == "assistant"][-1]
    assert final == "Found file.txt."
    assert any(e.event_type == "compacted" for e in events)
    assert len(bodies) == 2
    assert _block_types(open_turn) == ["thinking", "tool_use"]
    assert open_turn["content"][0] == EMPTY_SIGNED_THINKING
    _assert_valid_continuation(bodies[1])


@pytest.mark.parametrize(
    "error_message", [MISSING_THINKING_400, SIGNATURE_400], ids=["missing", "signature"]
)
async def test_thinking_400_mid_tool_loop_heals_once_with_a_valid_history(
    anthropic_env, provider_http, error_message
):
    session = _make_session()
    added = _record_added_messages(session)
    provider_http.queue_anthropic_sse(
        _anthropic_stream(
            [_signed_empty_thinking_block(), _bash_tool_use_block()], "tool_use"
        )
    )
    provider_http.queue_anthropic_error(
        400, {"type": "invalid_request_error", "message": error_message}
    )
    provider_http.queue_anthropic_sse(
        _anthropic_stream([_text_block("Found file.txt.")], "end_turn")
    )

    final = await Handlers.run_agent(session, "list files")

    events = _drain_events(session)
    bodies = provider_http.bodies_for("api.anthropic.com")
    original_turn = added[1]
    healed_turn = session.context_manager.items[2]
    assert final == "Found file.txt."
    assert len(bodies) == 3
    assert _error_events(events) == []
    assert sum(e.event_type == "tool_log" for e in events) == 1
    _assert_valid_continuation(bodies[2])
    assert "SIG_EMPTY" not in json.dumps(bodies[2]["messages"])
    assert "[called tool `bash`" in json.dumps(bodies[2]["messages"])
    assert original_turn.thinking_blocks == [EMPTY_SIGNED_THINKING]
    assert original_turn.tool_calls[0].id == "toolu_1"
    assert healed_turn is not original_turn
    assert healed_turn.tool_calls is None
    assert getattr(healed_turn, "thinking_blocks", None) is None


async def test_second_thinking_400_after_heal_is_surfaced_not_looped(
    anthropic_env, provider_http
):
    session = _make_session()
    provider_http.queue_anthropic_sse(
        _anthropic_stream(
            [_signed_empty_thinking_block(), _bash_tool_use_block()], "tool_use"
        )
    )
    for _ in range(2):
        provider_http.queue_anthropic_error(
            400, {"type": "invalid_request_error", "message": MISSING_THINKING_400}
        )

    final = await Handlers.run_agent(session, "list files")

    errors = _error_events(_drain_events(session))
    assert final is None
    assert len(provider_http.requests) == 3
    assert len(errors) == 1
    assert "thinking" in errors[0]


DEEPSEEK_MODEL = "deepseek/deepseek-v4-pro"


@pytest.mark.parametrize(
    "model_name,host",
    [
        (ROUTER_MODEL, "router.huggingface.co"),
        ("zai-org/GLM-5.2:novita", "router.huggingface.co"),
        (DEEPSEEK_MODEL, "api.deepseek.com"),
    ],
    ids=["router-claude", "glm", "deepseek"],
)
@pytest.mark.parametrize(
    "text,finish_reason",
    [("Partial answer.", "content_filter"), (None, "length")],
    ids=["content-filter", "empty-length"],
)
async def test_non_direct_unproductive_stop_keeps_head_behaviour_end_to_end(
    monkeypatch, provider_http, model_name, host, text, finish_reason
):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    session = _make_session(model_name)
    provider_http.queue_openai_stream(
        host, text, finish_reason=finish_reason, model=model_name
    )

    final = await Handlers.run_agent(session, "hi")

    events = _drain_events(session)
    stored = ["system", "user", "assistant"] if text else ["system", "user"]
    assert final == text
    assert _error_events(events) == []
    assert any(e.event_type == "turn_complete" for e in events)
    assert [m.role for m in session.context_manager.items] == stored


# ── research sub-agent ─────────────────────────────────────────────────


def test_research_model_is_not_downgraded_for_direct_anthropic():
    assert research_tool._get_research_model(DIRECT_MODEL) == DIRECT_MODEL


class ResearchToolRouter(BashToolRouter):
    def get_tool_specs_for_llm(self):
        specs = super().get_tool_specs_for_llm()
        return [
            {
                **specs[0],
                "function": {**specs[0]["function"], "name": "read"},
            }
        ]


def _research_session(model_name: str) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(model_name=model_name, reasoning_effort="high"),
        hf_token=DUMMY_HF_TOKEN,
        tool_router=ResearchToolRouter(),
        send_event=_noop_send_event,
        session_id="research-test",
    )


def _research_response(message, total_tokens: int = 100) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(total_tokens=total_tokens),
    )


def _research_tool_message() -> SimpleNamespace:
    return SimpleNamespace(
        content="",
        tool_calls=[
            ChatCompletionMessageToolCall(
                id="toolu_r1",
                type="function",
                function={"name": "read", "arguments": '{"path": "a.txt"}'},
            )
        ],
        thinking_blocks=[dict(EMPTY_SIGNED_THINKING)],
        provider_specific_fields={"thinking_blocks": [dict(EMPTY_SIGNED_THINKING)]},
        reasoning_content="",
    )


def _patch_research_calls(monkeypatch, responses: list) -> list[dict]:
    calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return responses.pop(0)

    async def fake_record(*_args, **_kwargs):
        return False

    monkeypatch.setattr(research_tool, "acompletion", fake_acompletion)
    monkeypatch.setattr(research_tool, "_record_research_llm_call", fake_record)
    return calls


async def test_research_keeps_thinking_blocks_and_caps_tokens_for_direct_anthropic(
    monkeypatch, anthropic_env
):
    calls = _patch_research_calls(
        monkeypatch,
        [
            _research_response(_research_tool_message()),
            _research_response(SimpleNamespace(content="Findings.", tool_calls=None)),
        ],
    )

    output, ok = await research_tool.research_handler(
        {"task": "find recipes"}, session=_research_session(DIRECT_MODEL)
    )

    assert (output, ok) == ("Findings.", True)
    assert all(call["max_tokens"] <= 16_000 for call in calls)
    assert calls[1]["model"] == DIRECT_MODEL
    echoed = calls[1]["messages"][2]
    assert echoed.role == "assistant"
    assert echoed.content is None
    assert echoed.thinking_blocks == [EMPTY_SIGNED_THINKING]
    assert getattr(echoed, "provider_specific_fields", None) is None


async def test_research_summary_call_keeps_tools_with_tool_choice_none_for_direct_anthropic(
    monkeypatch, anthropic_env
):
    calls = _patch_research_calls(
        monkeypatch,
        [
            _research_response(_research_tool_message(), total_tokens=195_000),
            _research_response(SimpleNamespace(content="Summary.", tool_calls=None)),
        ],
    )

    output, ok = await research_tool.research_handler(
        {"task": "find recipes"}, session=_research_session(DIRECT_MODEL)
    )

    assert (output, ok) == ("Summary.", True)
    summary_call = calls[1]
    assert summary_call["tools"] == calls[0]["tools"]
    assert summary_call["tool_choice"] == "none"


async def test_research_summary_call_for_router_model_still_sends_no_tools(
    monkeypatch,
):
    calls = _patch_research_calls(
        monkeypatch,
        [
            _research_response(_research_tool_message(), total_tokens=195_000),
            _research_response(SimpleNamespace(content="Summary.", tool_calls=None)),
        ],
    )

    await research_tool.research_handler(
        {"task": "find recipes"}, session=_research_session(ROUTER_MODEL)
    )

    summary_call = calls[1]
    assert summary_call["tools"] is None
    assert "tool_choice" not in summary_call
    echoed = calls[1]["messages"][2]
    assert getattr(echoed, "thinking_blocks", None) is None


async def test_research_reports_missing_anthropic_key_as_tool_error(monkeypatch):

    def missing_key(*_args, **_kwargs):
        raise MissingProviderKeyError(MISSING_KEY_MESSAGE)

    monkeypatch.setattr(research_tool, "_resolve_llm_params", missing_key)

    output, ok = await research_tool.research_handler(
        {"task": "find recipes"}, session=_research_session(DIRECT_MODEL)
    )

    assert (output, ok) == (MISSING_KEY_MESSAGE, False)


# ── corrupted tool-call arguments in saved history ─────────────────────

# What an older redactor left behind in saved sessions: the escape before
# the quote after `[REDACTED]` was eaten, so the JSON string ends early.
CORRUPTED_WRITE_ARGS = (
    '{"path": "gen.py", "content": "client = InferenceClient('
    'api_key=[REDACTED]"HF_TOKEN\\"], timeout=180)\\n"}'
)


def _write_call(arguments: str) -> ChatCompletionMessageToolCall:
    return ChatCompletionMessageToolCall(
        id="toolu_w",
        type="function",
        function={"name": "write", "arguments": arguments},
    )


def _history_with_tool_arguments(arguments: str) -> list[Message]:
    return [
        Message(role="user", content="write the generator"),
        Message(role="assistant", content=None, tool_calls=[_write_call(arguments)]),
        Message(
            role="tool", content="wrote gen.py", tool_call_id="toolu_w", name="write"
        ),
        Message(role="assistant", content="Done."),
    ]


def test_corrupted_arguments_fixture_is_not_json():
    with pytest.raises(json.JSONDecodeError):
        json.loads(CORRUPTED_WRITE_ARGS)


def test_scrub_for_direct_anthropic_wraps_unparseable_tool_arguments():
    history = _history_with_tool_arguments(CORRUPTED_WRITE_ARGS)

    outgoing = _scrub_reasoning_for_provider(history, DIRECT_MODEL)

    sent = outgoing[1].tool_calls[0]
    assert json.loads(sent.function.arguments) == {
        "_unparsed_arguments": CORRUPTED_WRITE_ARGS
    }
    assert sent.id == "toolu_w"
    assert sent.function.name == "write"
    assert history[1].tool_calls[0].function.arguments == CORRUPTED_WRITE_ARGS


@pytest.mark.parametrize("arguments", ['{"path": "gen.py"}', ""])
def test_scrub_for_direct_anthropic_leaves_parseable_or_empty_arguments(arguments):
    history = _history_with_tool_arguments(arguments)

    outgoing = _scrub_reasoning_for_provider(history, DIRECT_MODEL)

    assert outgoing[1].tool_calls[0].function.arguments == arguments


def test_scrub_wraps_non_object_json_arguments_for_direct_anthropic():
    history = _history_with_tool_arguments('["not", "an", "object"]')

    outgoing = _scrub_reasoning_for_provider(history, DIRECT_MODEL)

    assert json.loads(outgoing[1].tool_calls[0].function.arguments) == {
        "_unparsed_arguments": '["not", "an", "object"]'
    }


def test_scrub_for_dict_tool_calls_wraps_unparseable_arguments():
    history = [
        {"role": "user", "content": "go"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "toolu_w",
                    "type": "function",
                    "function": {"name": "write", "arguments": CORRUPTED_WRITE_ARGS},
                }
            ],
        },
    ]

    outgoing = _scrub_reasoning_for_provider(history, DIRECT_MODEL)

    sent = outgoing[1]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(sent) == {"_unparsed_arguments": CORRUPTED_WRITE_ARGS}
    assert history[1]["tool_calls"][0]["function"]["arguments"] == CORRUPTED_WRITE_ARGS


@pytest.mark.parametrize(
    "model_name", [ROUTER_LITELLM_MODEL, "deepseek/deepseek-v4-pro"]
)
def test_scrub_for_non_direct_targets_keeps_corrupted_arguments_as_is(model_name):
    history = _history_with_tool_arguments(CORRUPTED_WRITE_ARGS)

    outgoing = _scrub_reasoning_for_provider(history, model_name)

    assert outgoing[1].tool_calls[0].function.arguments == CORRUPTED_WRITE_ARGS


async def test_direct_turn_on_history_with_corrupted_tool_arguments_reaches_anthropic(
    anthropic_env, provider_http, monkeypatch
):
    monkeypatch.setattr(agent_loop, "_LLM_RETRY_DELAYS", [])
    session = _make_session()
    cm = session.context_manager
    cm.items = [cm.items[0], *_history_with_tool_arguments(CORRUPTED_WRITE_ARGS)]
    provider_http.queue_anthropic_sse(
        _anthropic_stream([_text_block("Continuing.")], "end_turn")
    )

    final = await Handlers.run_agent(session, "continue")

    assert final == "Continuing."
    body = provider_http.bodies_for("api.anthropic.com")[0]
    tool_use = body["messages"][1]["content"][0]
    assert tool_use["type"] == "tool_use"
    assert tool_use["input"] == {"_unparsed_arguments": CORRUPTED_WRITE_ARGS}
    assert cm.items[2].tool_calls[0].function.arguments == CORRUPTED_WRITE_ARGS


# ── classifying errors that echo the request ───────────────────────────

# LiteLLM appends the whole request to some local errors. That echo holds
# arbitrary conversation text, so it must not drive error classification.
_ECHOED_REQUEST = (
    " Received Messages=[{'content': 'the HF call failed with 402 Payment "
    "Required; credit balance is too low; authentication; rate limit; "
    "timeout=180; 500 internal server error', 'role': 'user'}]"
)


def _anthropic_local_conversion_error():
    import litellm

    return litellm.BadRequestError(
        message=(
            "AnthropicException - Failed to parse tool call arguments for tool "
            "'write' (Anthropic tool invoke). Error: Expecting ',' delimiter: "
            'line 1 column 70 (char 69). Arguments: {"path": "gen.py"}'
            + _ECHOED_REQUEST
        ),
        model="claude-opus-5-5",
        llm_provider="anthropic",
    )


def test_anthropic_bad_request_echoing_the_request_is_not_retried():
    error = _anthropic_local_conversion_error()

    assert agent_loop._is_transient_error(error) is False
    assert agent_loop._retry_delay_for(error, 0) is None


def test_anthropic_bad_request_echo_is_not_called_a_credit_or_key_problem():
    error = _anthropic_local_conversion_error()

    message = _friendly_error_message(error, user_plan="free", model_name=DIRECT_MODEL)

    assert message is None


def test_anthropic_overloaded_error_is_still_retried():
    import litellm

    error = litellm.InternalServerError(
        message='AnthropicException - {"type":"error","error":{"type":'
        '"overloaded_error","message":"Overloaded"}}',
        model="claude-opus-5-5",
        llm_provider="anthropic",
    )

    assert agent_loop._is_transient_error(error) is True


def test_anthropic_402_billing_error_uses_anthropic_billing_message():
    error = _provider_error(
        'litellm.APIError: AnthropicException - {"type":"error","error":'
        '{"type":"billing_error","message":"Payment method declined"}}',
        "anthropic",
    )

    message = _friendly_error_message(error, model_name=DIRECT_MODEL)

    assert message is not None
    assert "Anthropic Console" in message


def test_router_error_text_with_timeout_keeps_transient_retry():
    error = _provider_error("upstream request timeout", "openai")

    assert agent_loop._is_transient_error(error) is True
