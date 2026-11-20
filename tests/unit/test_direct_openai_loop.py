"""Direct OpenAI route through the agent loop: errors, retries, output caps,
unproductive stops, compaction summaries and a streamed tool loop.

Offline: LLM calls are faked at ``acompletion`` and every key is a dummy set
with monkeypatch (importing ``agent`` loads the repo .env).
"""

import asyncio
from types import SimpleNamespace

import httpx
import litellm
import pytest
from litellm import ChatCompletionMessageToolCall, Message

from agent.config import Config
from agent.context_manager import manager as context_manager
from agent.context_manager.manager import summarize_messages
from agent.core import agent_loop
from agent.core.agent_loop import (
    _CONTENT_FILTER_STOP_MESSAGE,
    _DIRECT_OPENAI_NON_STREAMING_MAX_COMPLETION_TOKENS,
    _EMPTY_LENGTH_STOP_MESSAGE,
    _LLM_RATE_LIMIT_RETRY_DELAYS,
    _LLM_RETRY_DELAYS,
    _OPENAI_AUTH_MESSAGE,
    _OPENAI_MODEL_ACCESS_MESSAGE,
    _OPENAI_QUOTA_MESSAGE,
    Handlers,
    LLMResult,
    _call_llm_non_streaming,
    _call_llm_streaming,
    _friendly_error_message,
    _retry_delay_for,
    _unproductive_stop_message,
    _with_non_streaming_max_tokens,
)
from agent.core.llm_params import MissingProviderKeyError, _resolve_llm_params
from agent.core.model_ids import (
    GPT_55_DIRECT_MODEL_ID,
    GPT_55_MODEL_ID,
    GPT_62_SOL_MODEL_ID,
    OPENAI_API_BASE_URL,
)
from agent.core.provider_errors import is_output_cap_error, openai_error_kind
from agent.core.session import Session

DIRECT_MODEL = GPT_62_SOL_MODEL_ID
ROUTER_MODEL = GPT_55_MODEL_ID
ROUTER_LITELLM_MODEL = f"openai/{ROUTER_MODEL}"
DUMMY_KEY = "sk-openai-test"
DUMMY_HF_TOKEN = "hf_test_dummy"
MISSING_KEY_MESSAGE = (
    "GPT via the OpenAI API needs OPENAI_API_KEY. "
    "Add it to the repo .env and restart the backend."
)
_ECHOED_REQUEST = (
    " Received Messages=[{'role': 'user', 'content': 'why did HF say "
    "insufficient_quota and 429 and timeout? Incorrect API key provided'}]"
)


@pytest.fixture(autouse=True)
def provider_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    monkeypatch.setenv("HF_TOKEN", DUMMY_HF_TOKEN)
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "OPENAI_ORGANIZATION"):
        monkeypatch.delenv(name, raising=False)


def _direct_params(model_name: str = DIRECT_MODEL) -> dict:
    """The llm_params shape ``_resolve_llm_params`` returns for direct ids."""
    return {
        "model": model_name,
        "api_base": OPENAI_API_BASE_URL,
        "api_key": DUMMY_KEY,
        "max_completion_tokens": 32000,
        "extra_body": {"store": False, "reasoning_effort": "high"},
    }


def _response(status: int) -> httpx.Response:
    return httpx.Response(
        status, request=httpx.Request("POST", f"{OPENAI_API_BASE_URL}/responses")
    )


def _auth_error() -> Exception:
    return litellm.AuthenticationError(
        message='OpenAIException - {"error":{"message":"Incorrect API key provided: '
        "sk-open***. You can find your API key at https://platform.openai.com/"
        'account/api-keys.","code":"invalid_api_key"}}',
        llm_provider="openai",
        model="gpt-6.2-sol",
    )


def _quota_error() -> Exception:
    return litellm.RateLimitError(
        message='OpenAIException - {"error":{"message":"You exceeded your current '
        'quota, please check your plan and billing details.","type":'
        '"insufficient_quota","code":"insufficient_quota"}}',
        llm_provider="openai",
        model="gpt-6.2-sol",
    )


def _model_access_error() -> Exception:
    return litellm.NotFoundError(
        message='OpenAIException - {"error":{"message":"The model `gpt-6.2-sol` '
        'does not exist or you do not have access to it.","code":'
        '"model_not_found"}}',
        model="gpt-6.2-sol",
        llm_provider="openai",
    )


def _bad_request(text: str = "Invalid value for 'input'.") -> Exception:
    return litellm.BadRequestError(
        message=f"OpenAIException - {text}",
        model="gpt-6.2-sol",
        llm_provider="openai",
    )


def _server_error() -> Exception:
    return litellm.InternalServerError(
        message='OpenAIException - {"error":{"message":"The server had an error '
        'while processing your request. Sorry about that!","type":"server_error"}}',
        llm_provider="openai",
        model="gpt-6.2-sol",
    )


def _output_cap_error() -> Exception:
    return litellm.APIConnectionError(
        message="OpenAIException - gpt-6.2-sol unable to complete request: "
        "max_output_tokens",
        llm_provider="openai",
        model="gpt-6.2-sol",
    )


# ── provider_errors.openai_error_kind ──────────────────────────────────


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (_auth_error(), "auth"),
        (
            litellm.PermissionDeniedError(
                message="OpenAIException - Your organization is not allowed here.",
                llm_provider="openai",
                model="gpt-6.2-sol",
                response=_response(403),
            ),
            "auth",
        ),
        (Exception("Incorrect API key provided: sk-xx"), "auth"),
        (Exception('{"code": "invalid_api_key"}'), "auth"),
        (_quota_error(), "quota"),
        (Exception("billing_hard_limit_reached"), "quota"),
        (_model_access_error(), "model_access"),
        (
            litellm.NotFoundError(
                message="OpenAIException - Project `proj_x` does not have access "
                'to model `gpt-6.2-sol` ("code":"model_not_found")',
                model="gpt-6.2-sol",
                llm_provider="openai",
            ),
            "model_access",
        ),
        (_server_error(), None),
        (_bad_request(), None),
        (Exception("boom"), None),
    ],
)
def test_openai_error_kind_classifies_provider_errors(error, kind):
    assert openai_error_kind(error) == kind


def test_openai_error_kind_ignores_the_echoed_request():
    error = _bad_request("Failed to convert the request." + _ECHOED_REQUEST)

    assert openai_error_kind(error) is None


def test_output_cap_error_is_recognised_only_for_the_bridge_text():
    assert is_output_cap_error(_output_cap_error()) is True
    assert is_output_cap_error(_server_error()) is False
    assert is_output_cap_error(Exception("max_output_tokens")) is False


# ── friendly errors ────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (_auth_error(), _OPENAI_AUTH_MESSAGE),
        (_quota_error(), _OPENAI_QUOTA_MESSAGE),
        (_model_access_error(), _OPENAI_MODEL_ACCESS_MESSAGE),
        (Exception("Incorrect API key provided: sk-xx"), _OPENAI_AUTH_MESSAGE),
    ],
    ids=["auth", "quota", "model-access", "no-llm-provider"],
)
def test_direct_openai_errors_use_openai_wording(error, expected):
    message = _friendly_error_message(error, user_plan="free", model_name=DIRECT_MODEL)

    assert message == expected
    assert "Hugging Face" not in message
    assert "huggingface.co" not in message


def test_openai_messages_have_the_exact_text():
    assert _OPENAI_AUTH_MESSAGE == (
        "OpenAI rejected the API key: OPENAI_API_KEY is missing or invalid. "
        "Set it in the repo .env and restart the backend."
    )
    assert _OPENAI_QUOTA_MESSAGE == (
        "Your OpenAI API quota is exhausted (insufficient_quota).\n\n"
        "Add credits or raise your usage limits in the OpenAI dashboard: "
        "https://platform.openai.com/settings/organization/billing/overview"
    )
    assert _OPENAI_MODEL_ACCESS_MESSAGE == (
        "Your OpenAI API key can't use this model (model_not_found). Check the "
        "project's model access in the OpenAI dashboard, or pick another model."
    )


@pytest.mark.parametrize(
    "error",
    [
        _bad_request("Invalid value for 'input'." + _ECHOED_REQUEST),
        Exception("Unauthorized: authentication failed"),
        Exception("402 Payment Required: monthly credits exhausted"),
        Exception("model not found"),
    ],
)
def test_other_direct_openai_errors_never_get_hf_wording(error):
    assert (
        _friendly_error_message(error, user_plan="free", model_name=DIRECT_MODEL)
        is None
    )


def test_output_cap_error_on_a_non_streaming_direct_call_is_explained():
    message = _friendly_error_message(_output_cap_error(), model_name=DIRECT_MODEL)

    assert message == _EMPTY_LENGTH_STOP_MESSAGE


def test_missing_openai_key_message_is_shown_verbatim():
    error = MissingProviderKeyError(MISSING_KEY_MESSAGE)

    assert (
        _friendly_error_message(error, model_name=DIRECT_MODEL) == MISSING_KEY_MESSAGE
    )


def test_router_gpt55_auth_error_keeps_hf_wording():
    message = _friendly_error_message(_auth_error(), model_name=ROUTER_MODEL)

    assert message is not None
    assert "Hugging Face token" in message
    assert "OPENAI_API_KEY" not in message


def test_router_gpt55_quota_error_keeps_hf_billing_wording():
    message = _friendly_error_message(
        _quota_error(), user_plan="free", model_name=ROUTER_MODEL
    )

    assert message is not None
    assert "https://huggingface.co/settings/billing" in message
    assert "OpenAI dashboard" not in message


def test_router_gpt55_model_access_error_keeps_hf_wording():
    message = _friendly_error_message(_model_access_error(), model_name=ROUTER_MODEL)

    assert message is not None
    assert message.startswith("Model not found.")


def test_anthropic_tagged_error_in_a_direct_openai_session_keeps_anthropic_wording():
    error = Exception("invalid x-api-key")
    error.llm_provider = "anthropic"

    message = _friendly_error_message(error, model_name=DIRECT_MODEL)

    assert message is not None
    assert "ANTHROPIC_API_KEY" in message


# ── retries ────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "error",
    [
        _quota_error(),
        _bad_request(),
        _bad_request("Invalid 'tools'. Request timed out 429 500" + _ECHOED_REQUEST),
        _auth_error(),
        _model_access_error(),
        _output_cap_error(),
    ],
    ids=["quota", "400", "400-echo", "auth", "model-access", "output-cap"],
)
def test_direct_openai_errors_that_repeat_are_not_retried(error):
    assert _retry_delay_for(error, 0, DIRECT_MODEL) is None


def test_router_429_insufficient_quota_keeps_the_rate_limit_schedule():
    # The OpenAI SDK (HF Router path) puts "Error code: 429" in the text.
    error = litellm.RateLimitError(
        message="OpenAIException - Error code: 429 - {'error': {'message': "
        "'You exceeded your current quota', 'code': 'insufficient_quota'}}",
        llm_provider="openai",
        model="openai/gpt-5.5:fal-ai",
    )

    assert (
        _retry_delay_for(error, 0, ROUTER_LITELLM_MODEL)
        == (_LLM_RATE_LIMIT_RETRY_DELAYS[0])
    )
    assert _retry_delay_for(error, 0) == _LLM_RATE_LIMIT_RETRY_DELAYS[0]
    assert _retry_delay_for(error, 0, DIRECT_MODEL) is None


@pytest.mark.parametrize(
    "error",
    [
        _server_error(),
        litellm.Timeout(
            message="Request timed out.", model="gpt-6.2-sol", llm_provider="openai"
        ),
        litellm.ServiceUnavailableError(
            message="OpenAIException - The engine is currently overloaded.",
            llm_provider="openai",
            model="gpt-6.2-sol",
        ),
        litellm.APIConnectionError(
            message="OpenAIException - Connection error.",
            llm_provider="openai",
            model="gpt-6.2-sol",
        ),
    ],
    ids=["500", "timeout", "503", "connection"],
)
def test_direct_openai_server_errors_and_timeouts_are_retried(error):
    delays = [
        _retry_delay_for(error, i, DIRECT_MODEL) for i in range(len(_LLM_RETRY_DELAYS))
    ]

    assert delays == _LLM_RETRY_DELAYS
    assert _retry_delay_for(error, len(_LLM_RETRY_DELAYS), DIRECT_MODEL) is None


def test_direct_openai_rate_limit_keeps_the_rate_limit_schedule():
    error = litellm.RateLimitError(
        message="OpenAIException - Rate limit reached for gpt-6.2-sol on tokens per min.",
        llm_provider="openai",
        model="gpt-6.2-sol",
    )

    assert _retry_delay_for(error, 0, DIRECT_MODEL) == _LLM_RATE_LIMIT_RETRY_DELAYS[0]


def test_router_500_without_status_text_keeps_head_behaviour():
    assert _retry_delay_for(_server_error(), 0, ROUTER_LITELLM_MODEL) is None


async def _noop_send_event(_event):
    return None


def _stub_session(model_name: str = DIRECT_MODEL) -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(model_name=model_name),
        is_cancelled=False,
        send_event=_noop_send_event,
    )


async def test_streaming_call_does_not_retry_a_direct_quota_error(monkeypatch):
    calls: list[dict] = []

    async def fail(**kwargs):
        calls.append(kwargs)
        raise _quota_error()

    monkeypatch.setattr(agent_loop, "acompletion", fail)

    with pytest.raises(litellm.RateLimitError):
        await _call_llm_streaming(
            _stub_session(), [Message(role="user", content="hi")], [], _direct_params()
        )

    assert len(calls) == 1


async def test_non_streaming_call_retries_a_direct_server_error(monkeypatch):
    calls: list[dict] = []

    async def flaky(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise _server_error()
        return _non_streaming_response(SimpleNamespace(content="ok", tool_calls=None))

    async def no_sleep(_session, _delay):
        return False

    monkeypatch.setattr(agent_loop, "acompletion", flaky)
    monkeypatch.setattr(agent_loop, "_sleep_for_retry_or_cancel", no_sleep)

    result = await _call_llm_non_streaming(
        _stub_session(), [Message(role="user", content="hi")], [], _direct_params()
    )

    assert result.content == "ok"
    assert len(calls) == 2


# ── non-streaming output cap ───────────────────────────────────────────


def _non_streaming_response(message) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="stop")],
        usage=SimpleNamespace(total_tokens=9),
    )


@pytest.mark.parametrize(
    ("llm_params", "expected"),
    [
        (_direct_params(), 16_000),
        ({**_direct_params(), "max_completion_tokens": 8_000}, 8_000),
        (
            {k: v for k, v in _direct_params().items() if k != "max_completion_tokens"},
            16_000,
        ),
    ],
    ids=["capped", "smaller-kept", "missing-set"],
)
def test_non_streaming_cap_applies_to_max_completion_tokens(llm_params, expected):
    capped = _with_non_streaming_max_tokens(llm_params)

    assert capped["max_completion_tokens"] == expected
    assert "max_tokens" not in capped
    assert _DIRECT_OPENAI_NON_STREAMING_MAX_COMPLETION_TOKENS == 16_000


def test_non_streaming_cap_leaves_router_gpt55_untouched():
    params = {"model": ROUTER_LITELLM_MODEL, "api_key": DUMMY_HF_TOKEN}

    assert _with_non_streaming_max_tokens(params) is params


async def test_non_streaming_direct_call_sends_the_capped_budget(monkeypatch):
    captured: list[dict] = []

    async def fake_acompletion(**kwargs):
        captured.append(kwargs)
        return _non_streaming_response(SimpleNamespace(content="ok", tool_calls=None))

    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)

    await _call_llm_non_streaming(
        _stub_session(), [Message(role="user", content="hi")], [], _direct_params()
    )

    (sent,) = captured
    assert sent["max_completion_tokens"] == 16_000
    assert "max_tokens" not in sent
    assert sent["extra_body"] == {"store": False, "reasoning_effort": "high"}
    assert "extra_headers" not in sent


# ── unproductive stops ─────────────────────────────────────────────────


def _result(content=None, finish_reason="stop", tool_calls=None, token_count=10):
    return LLMResult(
        content=content,
        tool_calls_acc=tool_calls or {},
        token_count=token_count,
        finish_reason=finish_reason,
    )


_TOOL_CALL_ACC = {
    1: {
        "id": "call_1",
        "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "ls"}'},
    }
}


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (_result(finish_reason="length"), _EMPTY_LENGTH_STOP_MESSAGE),
        (_result(finish_reason="content_filter"), _CONTENT_FILTER_STOP_MESSAGE),
        (_result("Partial.", "content_filter"), _CONTENT_FILTER_STOP_MESSAGE),
        # The bridge's signature of a dropped ``response.incomplete``.
        (_result(finish_reason="stop", token_count=0), _EMPTY_LENGTH_STOP_MESSAGE),
        (_result("Partial answer.", "length"), None),
        (_result(finish_reason="length", tool_calls=_TOOL_CALL_ACC), None),
        (_result("Done.", "stop"), None),
        (_result(finish_reason="stop", token_count=12), None),
        (_result(finish_reason="tool_calls", tool_calls=_TOOL_CALL_ACC), None),
    ],
    ids=[
        "empty-length",
        "content-filter",
        "content-filter-with-text",
        "empty-stop-without-usage",
        "length-with-text",
        "length-with-tool-call",
        "stop-with-text",
        "empty-stop-with-usage",
        "tool-calls",
    ],
)
def test_direct_openai_unproductive_stop_messages(result, expected):
    assert _unproductive_stop_message(result, DIRECT_MODEL) == expected


def test_content_filter_message_has_the_exact_text():
    assert _CONTENT_FILTER_STOP_MESSAGE == (
        "The model stopped without a reply (content_filter). Rephrase the "
        "request or switch models to continue."
    )


@pytest.mark.parametrize(
    "result",
    [
        _result(finish_reason="length"),
        _result(finish_reason="content_filter"),
        _result(finish_reason="stop", token_count=0),
    ],
)
def test_router_gpt55_unproductive_stops_keep_head_behaviour(result):
    assert _unproductive_stop_message(result, ROUTER_LITELLM_MODEL) is None


# ── compaction / restore summaries ─────────────────────────────────────


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


@pytest.mark.parametrize("requested", [2000, 4000, 27_200, 100_000])
async def test_direct_openai_summary_uses_low_effort_and_a_16k_budget(
    monkeypatch, requested
):
    captured: list[dict] = []
    monkeypatch.setattr(
        context_manager, "acompletion", _fake_summary_acompletion(captured)
    )
    history = [
        Message(role="user", content="build it"),
        Message(
            role="assistant",
            content="checking",
            reasoning_content="private reasoning",
        ),
    ]

    summary, tokens = await summarize_messages(
        history, model_name=DIRECT_MODEL, max_tokens=requested
    )

    (sent,) = captured
    assert (summary, tokens) == ("summary", 3)
    assert sent["model"] == DIRECT_MODEL
    assert sent["api_base"] == OPENAI_API_BASE_URL
    assert sent["extra_body"] == {"store": False, "reasoning_effort": "low"}
    assert sent["max_completion_tokens"] == 16_000
    assert "max_tokens" not in sent
    assert "extra_headers" not in sent
    assert all(getattr(m, "reasoning_content", None) is None for m in sent["messages"])
    assert history[1].reasoning_content == "private reasoning"


async def test_router_gpt55_summary_keeps_high_effort_and_requested_tokens(
    monkeypatch,
):
    captured: list[dict] = []
    monkeypatch.setattr(
        context_manager, "acompletion", _fake_summary_acompletion(captured)
    )

    await summarize_messages(
        [Message(role="assistant", content="earlier reply")],
        model_name=ROUTER_MODEL,
        hf_token=DUMMY_HF_TOKEN,
        max_tokens=50_000,
    )

    (sent,) = captured
    assert sent["max_completion_tokens"] == 50_000
    assert sent["extra_body"]["reasoning_effort"] == "high"


# ── streamed tool loop with a fake acompletion ─────────────────────────


def _tool_call_delta(index, *, call_id=None, name=None, arguments=None):
    return SimpleNamespace(
        index=index,
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _chunk(*, content=None, tool_calls=None, finish_reason=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                delta=SimpleNamespace(
                    content=content,
                    tool_calls=tool_calls,
                    reasoning_content=None,
                    thinking_blocks=None,
                ),
                finish_reason=finish_reason,
            )
        ],
    )


def _usage_chunk(total_tokens: int = 1200):
    return SimpleNamespace(choices=[], usage=SimpleNamespace(total_tokens=total_tokens))


def _bridge_tool_call_stream():
    # The bridge numbers tool calls by output_index (reasoning item is 0).
    yield _chunk(content="")
    yield _chunk(tool_calls=[_tool_call_delta(1, call_id="call_1", name="bash")])
    yield _chunk(tool_calls=[_tool_call_delta(1, arguments='{"command": ')])
    yield _chunk(tool_calls=[_tool_call_delta(1, arguments='"ls"}')])
    yield _chunk(finish_reason="tool_calls")
    yield _usage_chunk()


def _bridge_text_stream(text: str):
    yield _chunk(content=text)
    yield _chunk(finish_reason="stop")
    yield _usage_chunk()


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


def _make_session(model_name: str = DIRECT_MODEL) -> Session:
    config = Config.model_validate({"model_name": model_name, "save_sessions": False})
    return Session(
        asyncio.Queue(),
        config,
        tool_router=BashToolRouter(),
        stream=True,
        hf_token=DUMMY_HF_TOKEN,
        hf_username="tester",
    )


def _drain_events(session: Session) -> list:
    events = []
    while not session.event_queue.empty():
        events.append(session.event_queue.get_nowait())
    return events


def _assistant_with_thinking_state() -> Message:
    return Message(
        role="assistant",
        content="Earlier answer.",
        thinking_blocks=[{"type": "thinking", "thinking": "", "signature": "SIG"}],
        reasoning_content="earlier reasoning",
    )


async def test_streaming_tool_loop_accumulates_calls_records_model_and_scrubs_thinking(
    monkeypatch,
):
    session = _make_session()
    session.context_manager.add_message(Message(role="user", content="earlier"))
    session.context_manager.add_message(_assistant_with_thinking_state())
    sent: list[dict] = []
    streams = [_bridge_tool_call_stream(), _bridge_text_stream("Found file.txt.")]

    async def fake_acompletion(**kwargs):
        sent.append(kwargs)
        stream = streams.pop(0)

        async def iterate():
            for chunk in stream:
                yield chunk

        return iterate()

    monkeypatch.setattr(agent_loop, "acompletion", fake_acompletion)

    final = await Handlers.run_agent(session, "list files")

    assert final == "Found file.txt."
    assert session.tool_router.calls == [("bash", {"command": "ls"})]
    assert [call["model"] for call in sent] == [DIRECT_MODEL, DIRECT_MODEL]
    assert sent[0]["extra_body"] == {"store": False, "reasoning_effort": "high"}
    assert sent[0]["max_completion_tokens"] == 32000
    assert sent[0]["tool_choice"] == "auto"
    for call in sent:
        assert "extra_headers" not in call
        for message in call["messages"]:
            if getattr(message, "role", None) == "assistant":
                assert getattr(message, "thinking_blocks", None) is None
                assert getattr(message, "reasoning_content", None) is None
    tool_turn = sent[1]["messages"][-2]
    assert tool_turn.tool_calls[0].id == "call_1"
    assert tool_turn.tool_calls[0].function.arguments == '{"command": "ls"}'
    stored = session.context_manager.items[2]
    assert stored.thinking_blocks == [
        {"type": "thinking", "thinking": "", "signature": "SIG"}
    ]
    llm_calls = [e for e in _drain_events(session) if e.event_type == "llm_call"]
    assert [e.data["model"] for e in llm_calls] == [DIRECT_MODEL, DIRECT_MODEL]


async def test_empty_length_stop_ends_the_turn_with_a_visible_error(monkeypatch):
    session = _make_session()

    async def fake_call(_session, _messages, _tools, _llm_params):
        return _result(finish_reason="length")

    monkeypatch.setattr(agent_loop, "_call_llm_streaming", fake_call)

    final = await Handlers.run_agent(session, "think hard")

    errors = [
        e.data["error"] for e in _drain_events(session) if e.event_type == "error"
    ]
    assert final is None
    assert errors == [_EMPTY_LENGTH_STOP_MESSAGE]
    assert [m.role for m in session.context_manager.items] == ["system", "user"]


async def test_missing_openai_key_during_a_turn_becomes_an_error_event(monkeypatch):
    session = _make_session()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    await Handlers.run_agent(session, "hello")

    errors = [
        e.data["error"] for e in _drain_events(session) if e.event_type == "error"
    ]
    assert errors == [MISSING_KEY_MESSAGE]


def test_assistant_message_for_direct_openai_never_stores_thinking():
    result = LLMResult(
        content="Done.",
        tool_calls_acc={},
        token_count=5,
        finish_reason="stop",
        reasoning_content="private",
        thinking_blocks=[{"type": "thinking", "thinking": "", "signature": "SIG"}],
    )

    message = agent_loop._assistant_message_from_result(result, model_name=DIRECT_MODEL)

    assert message.content == "Done."
    assert getattr(message, "thinking_blocks", None) is None
    assert getattr(message, "reasoning_content", None) is None


def test_scrub_for_direct_openai_drops_thinking_and_reasoning():
    history = [
        Message(role="user", content="hi"),
        _assistant_with_thinking_state(),
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="call_1",
                    type="function",
                    function={"name": "bash", "arguments": "{}"},
                )
            ],
        ),
    ]

    scrubbed = agent_loop._scrub_reasoning_for_provider(history, DIRECT_MODEL)

    assert getattr(scrubbed[1], "thinking_blocks", None) is None
    assert getattr(scrubbed[1], "reasoning_content", None) is None
    assert scrubbed[2] is history[2]
    assert history[1].reasoning_content == "earlier reasoning"


def test_research_sub_call_for_direct_openai_is_capped_like_the_main_loop():
    from agent.tools import research_tool

    params = _resolve_llm_params(DIRECT_MODEL, DUMMY_HF_TOKEN, reasoning_effort="high")

    capped = research_tool._research_llm_params(params)

    assert capped["max_completion_tokens"] == 16_000
    assert "max_tokens" not in capped
    assert research_tool._get_research_model(GPT_55_DIRECT_MODEL_ID) == (
        GPT_55_DIRECT_MODEL_ID
    )


# ── CLI /model switch ──────────────────────────────────────────────────


class _SilentConsole:
    def __init__(self):
        self.lines: list[str] = []

    def print(self, *args, **_kwargs):
        self.lines.append(" ".join(str(arg) for arg in args))


@pytest.mark.parametrize(
    "model_id",
    [
        DIRECT_MODEL,
        GPT_55_DIRECT_MODEL_ID,
        "anthropic/claude-haiku-5-5",
        "deepseek/deepseek-v4-flash",
    ],
)
def test_cli_switch_skips_the_hf_catalog_for_direct_provider_ids(monkeypatch, model_id):
    from agent.core import hf_router_catalog, model_switcher

    def no_lookup(*_args, **_kwargs):
        raise AssertionError("direct-provider ids are not in the HF router catalog")

    monkeypatch.setattr(hf_router_catalog, "lookup", no_lookup)
    console = _SilentConsole()

    assert model_switcher._print_hf_routing_info(model_id, console) is True
    assert console.lines == []


def test_cli_switch_still_checks_the_catalog_for_router_gpt55(monkeypatch):
    from agent.core import hf_router_catalog, model_switcher

    lookups: list[str] = []

    def lookup(bare):
        lookups.append(bare)
        return None

    monkeypatch.setattr(hf_router_catalog, "lookup", lookup)
    monkeypatch.setattr(hf_router_catalog, "fuzzy_suggest", lambda _bare: [])

    assert model_switcher._print_hf_routing_info(ROUTER_MODEL, _SilentConsole()) is True
    assert lookups == ["openai/gpt-5.5"]


async def test_cli_switch_to_a_direct_openai_model_probes_and_commits(monkeypatch):
    from agent.core import effort_probe, model_switcher

    calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")])

    monkeypatch.setattr(effort_probe, "acompletion", fake_acompletion)
    config = SimpleNamespace(model_name=ROUTER_MODEL, reasoning_effort="max")
    session = SimpleNamespace(model_effective_effort={}, updated=[])
    session.update_model = session.updated.append
    console = _SilentConsole()

    await model_switcher.probe_and_switch_model(
        DIRECT_MODEL, config, session, console, hf_token=None
    )

    assert session.updated == [DIRECT_MODEL]
    assert session.model_effective_effort == {DIRECT_MODEL: "max"}
    (call,) = calls
    assert call["model"] == DIRECT_MODEL
    assert call["extra_body"] == {"store": False, "reasoning_effort": {"effort": "max"}}
    assert call["max_completion_tokens"] == 64
    assert "max_tokens" not in call
    assert "isn't in the HF router catalog" not in "\n".join(console.lines)
