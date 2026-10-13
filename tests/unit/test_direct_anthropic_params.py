"""Direct Anthropic route: model registration, LiteLLM kwargs, effort probe,
and an offline contract lock on the request LiteLLM actually sends.

No test here talks to a real LLM provider. The contract test patches
LiteLLM's async HTTP handler and returns a canned Anthropic response.
"""

import json as json_module
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import litellm
import pytest
from litellm import ChatCompletionMessageToolCall, Message
from litellm.llms.custom_httpx import http_handler
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

from agent.core import effort_probe, llm_params
from agent.core.direct_anthropic import (
    DIRECT_ANTHROPIC_LITELLM_MODELS,
    register_direct_anthropic_models,
)
from agent.core.llm_params import (
    MissingProviderKeyError,
    UnsupportedEffortError,
    _resolve_llm_params,
)
from agent.core.model_ids import (
    ANTHROPIC_API_BASE_URL,
    CLAUDE_OPUS_48_MODEL_ID,
    CLAUDE_OPUS_55_MODEL_ID,
    CLAUDE_SONNET_55_MODEL_ID,
    HF_ROUTER_BASE_URL,
)
from agent.core.prompt_caching import with_prompt_cache_params, with_prompt_caching

DUMMY_KEY = "sk-ant-test"
BOGUS_BASE_URL = "https://bogus.invalid"
MISSING_KEY_MESSAGE = (
    "Claude via the Anthropic API needs ANTHROPIC_API_KEY. "
    "Add it to the repo .env and restart the backend."
)
FORBIDDEN_DIRECT_KEYS = (
    "reasoning_effort",
    "thinking",
    "extra_body",
    "temperature",
    "top_p",
    "top_k",
)
DIRECT_IDS = (CLAUDE_OPUS_55_MODEL_ID, CLAUDE_SONNET_55_MODEL_ID)


@pytest.fixture
def anthropic_env(monkeypatch):
    """Dummy key, and an inherited base-URL override that must be ignored."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_KEY)
    monkeypatch.setenv("ANTHROPIC_BASE_URL", BOGUS_BASE_URL)
    monkeypatch.setenv("ANTHROPIC_API_BASE", BOGUS_BASE_URL)


def _fail_hf_token_lookup(*_args, **_kwargs):
    raise AssertionError("direct Anthropic must not resolve an HF token")


# ── Model registration ────────────────────────────────────────────────────


@pytest.mark.parametrize("model_id", DIRECT_IDS)
def test_registered_direct_models_expose_1m_context_and_128k_output(model_id):
    info = litellm.get_model_info(model_id)

    assert info["litellm_provider"] == "anthropic"
    assert info["max_input_tokens"] == 1_000_000
    assert info["max_output_tokens"] == 128_000
    assert info["supports_function_calling"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_assistant_prefill"] is False


@pytest.mark.parametrize("model_id", DIRECT_IDS)
def test_registered_direct_models_never_claim_reasoning_support(model_id):
    # supports_reasoning=True would let LiteLLM 1.83 turn a stray
    # reasoning_effort into thinking.type=enabled + budget_tokens (HTTP 400).
    info = litellm.get_model_info(model_id)

    assert info["supports_reasoning"] is False


@pytest.mark.parametrize(
    ("model_id", "expected_cost"),
    [
        # 100 in, 50 out, 1000 cache-read, 200 cache-write (5-minute TTL)
        (CLAUDE_OPUS_55_MODEL_ID, 100 * 4e-6 + 50 * 2e-5 + 1000 * 2e-7 + 200 * 5e-6),
        (
            CLAUDE_SONNET_55_MODEL_ID,
            100 * 2e-6 + 50 * 1e-5 + 1000 * 2e-7 + 200 * 2.5e-6,
        ),
    ],
)
def test_registered_direct_models_price_cache_reads_and_writes(model_id, expected_cost):
    # Anthropic usage folds cache tokens into prompt_tokens (100 + 1000 + 200).
    usage = Usage(
        prompt_tokens=1300,
        completion_tokens=50,
        total_tokens=1350,
        cache_read_input_tokens=1000,
        cache_creation_input_tokens=200,
        prompt_tokens_details=PromptTokensDetailsWrapper(
            cached_tokens=1000, cache_creation_tokens=200
        ),
    )

    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model_id, usage_object=usage
    )

    assert prompt_cost + completion_cost == pytest.approx(expected_cost)


def test_registration_overrides_a_stale_remote_map_entry(monkeypatch):
    # The remote LiteLLM map lists Sonnet 5.5 cache reads at $0.10/M and
    # supports_reasoning=True; registration must win regardless.
    stale = {
        **litellm.model_cost["claude-sonnet-5-5"],
        "cache_read_input_token_cost": 1e-7,
        "supports_reasoning": True,
    }
    monkeypatch.setitem(litellm.model_cost, "claude-sonnet-5-5", stale)

    register_direct_anthropic_models()

    info = litellm.get_model_info(CLAUDE_SONNET_55_MODEL_ID)
    assert info["cache_read_input_token_cost"] == pytest.approx(2e-7)
    assert info["supports_reasoning"] is False


def test_registration_data_is_keyed_by_bare_litellm_model_names():
    assert set(DIRECT_ANTHROPIC_LITELLM_MODELS) == {
        "claude-opus-5-5",
        "claude-sonnet-5-5",
    }


def _router_lookup(model_id: str):
    try:
        return litellm.get_model_info(model_id).get("max_input_tokens")
    except Exception as exc:  # unmapped ids raise; the type is the outcome
        return type(exc).__name__


@pytest.mark.parametrize(
    "router_id",
    [CLAUDE_OPUS_48_MODEL_ID, f"openai/{CLAUDE_OPUS_48_MODEL_ID}"],
)
def test_registration_leaves_router_claude_model_info_unchanged(router_id):
    before = _router_lookup(router_id)

    register_direct_anthropic_models()

    assert _router_lookup(router_id) == before
    assert before != 1_000_000


# ── _resolve_llm_params: direct branch ────────────────────────────────────


@pytest.mark.parametrize("model_id", DIRECT_IDS)
@pytest.mark.parametrize(
    ("effort", "wire_effort"),
    [("minimal", "low"), ("low", "low"), ("medium", "medium"), ("high", "high")],
)
@pytest.mark.parametrize("strict", [True, False])
def test_direct_params_send_supported_effort_as_output_config(
    anthropic_env, model_id, effort, wire_effort, strict
):
    params = _resolve_llm_params(model_id, reasoning_effort=effort, strict=strict)

    assert params == {
        "model": model_id,
        "api_base": ANTHROPIC_API_BASE_URL,
        "api_key": DUMMY_KEY,
        "max_tokens": 32000,
        "output_config": {"effort": wire_effort},
    }


@pytest.mark.parametrize("model_id", DIRECT_IDS)
def test_direct_params_omit_output_config_without_effort(anthropic_env, model_id):
    params = _resolve_llm_params(model_id, reasoning_effort=None)

    assert params == {
        "model": model_id,
        "api_base": ANTHROPIC_API_BASE_URL,
        "api_key": DUMMY_KEY,
        "max_tokens": 32000,
    }


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_direct_params_reject_xhigh_and_max_in_strict_mode(anthropic_env, effort):
    with pytest.raises(UnsupportedEffortError, match=effort):
        _resolve_llm_params(
            CLAUDE_OPUS_55_MODEL_ID, reasoning_effort=effort, strict=True
        )


@pytest.mark.parametrize("effort", ["xhigh", "max"])
def test_direct_params_clamp_xhigh_and_max_to_high_in_non_strict_mode(
    anthropic_env, effort
):
    params = _resolve_llm_params(
        CLAUDE_OPUS_55_MODEL_ID, reasoning_effort=effort, strict=False
    )

    assert params["output_config"] == {"effort": "high"}


def test_direct_params_reject_unknown_effort_in_strict_mode(anthropic_env):
    with pytest.raises(UnsupportedEffortError):
        _resolve_llm_params(
            CLAUDE_OPUS_55_MODEL_ID, reasoning_effort="turbo", strict=True
        )


def test_direct_params_drop_unknown_effort_in_non_strict_mode(anthropic_env):
    params = _resolve_llm_params(CLAUDE_OPUS_55_MODEL_ID, reasoning_effort="turbo")

    assert "output_config" not in params


@pytest.mark.parametrize("effort", [None, "minimal", "high", "xhigh", "max"])
def test_direct_params_never_emit_rejected_anthropic_fields(anthropic_env, effort):
    params = _resolve_llm_params(CLAUDE_OPUS_55_MODEL_ID, reasoning_effort=effort)

    for key in FORBIDDEN_DIRECT_KEYS:
        assert key not in params


def test_direct_params_strip_whitespace_around_the_key(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", f"  {DUMMY_KEY}\n")

    params = _resolve_llm_params(CLAUDE_OPUS_55_MODEL_ID)

    assert params["api_key"] == DUMMY_KEY


@pytest.mark.parametrize("raw_key", [None, "", "   ", "\n\t"])
def test_direct_params_raise_missing_provider_key_error_without_a_key(
    monkeypatch, raw_key
):
    if raw_key is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", raw_key)

    with pytest.raises(MissingProviderKeyError) as excinfo:
        _resolve_llm_params(CLAUDE_SONNET_55_MODEL_ID, reasoning_effort="high")

    assert str(excinfo.value) == MISSING_KEY_MESSAGE
    assert isinstance(excinfo.value, RuntimeError)


def test_direct_params_ignore_inherited_anthropic_base_url(anthropic_env):
    params = _resolve_llm_params(CLAUDE_OPUS_55_MODEL_ID, reasoning_effort="high")

    assert params["api_base"] == ANTHROPIC_API_BASE_URL


def test_direct_params_never_use_hf_tokens(anthropic_env, monkeypatch):
    monkeypatch.setattr(llm_params, "resolve_hf_router_token", _fail_hf_token_lookup)

    params = _resolve_llm_params(
        CLAUDE_OPUS_55_MODEL_ID, "hf_session_token", reasoning_effort="high"
    )

    assert params["api_key"] == DUMMY_KEY


def test_huggingface_prefixed_claude_55_still_routes_through_hf_router(
    anthropic_env,
):
    params = _resolve_llm_params(
        f"huggingface/{CLAUDE_OPUS_55_MODEL_ID}", "hf_session_token"
    )

    assert params["model"] == f"openai/{CLAUDE_OPUS_55_MODEL_ID}"
    assert params["api_base"] == HF_ROUTER_BASE_URL
    assert params["api_key"] == "hf_session_token"


def test_router_claude_ignores_anthropic_env(anthropic_env):
    params = _resolve_llm_params(
        CLAUDE_OPUS_48_MODEL_ID, "hf_session_token", reasoning_effort="high"
    )

    assert params == {
        "model": f"openai/{CLAUDE_OPUS_48_MODEL_ID}",
        "api_base": HF_ROUTER_BASE_URL,
        "api_key": "hf_session_token",
        "extra_body": {"reasoning_effort": "high"},
    }


# ── Effort probe cascade on the direct route ─────────────────────────────


@pytest.fixture
def probe_calls(monkeypatch):
    calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop")], usage=None
        )

    monkeypatch.setattr(effort_probe, "acompletion", fake_acompletion)
    return calls


@pytest.mark.parametrize("preference", ["max", "xhigh"])
async def test_probe_skips_xhigh_and_max_without_a_call_and_lands_on_high(
    anthropic_env, probe_calls, preference
):
    outcome = await effort_probe.probe_effort(
        CLAUDE_OPUS_55_MODEL_ID, preference, "hf_fake"
    )

    assert outcome.effective_effort == "high"
    assert outcome.attempts == 1
    assert outcome.note == f"{preference} not supported, using high"
    assert len(probe_calls) == 1
    assert probe_calls[0]["output_config"] == {"effort": "high"}
    assert probe_calls[0]["max_tokens"] == effort_probe._PROBE_MAX_TOKENS


async def test_probe_sends_minimal_preference_as_low_effort(anthropic_env, probe_calls):
    outcome = await effort_probe.probe_effort(
        CLAUDE_SONNET_55_MODEL_ID, "minimal", "hf_fake"
    )

    assert outcome.effective_effort == "minimal"
    assert len(probe_calls) == 1
    assert probe_calls[0]["output_config"] == {"effort": "low"}


async def test_probe_on_direct_route_sends_no_hf_session_header(
    anthropic_env, probe_calls
):
    session = SimpleNamespace(session_id="session-1")

    await effort_probe.probe_effort(CLAUDE_OPUS_55_MODEL_ID, "high", "hf_fake")
    await effort_probe.probe_effort(
        CLAUDE_OPUS_55_MODEL_ID, "high", "hf_fake", session=session
    )

    for call in probe_calls:
        assert "extra_headers" not in call
        assert "extra_body" not in call
        assert call["api_base"] == ANTHROPIC_API_BASE_URL
        assert call["model"] == CLAUDE_OPUS_55_MODEL_ID


async def test_probe_surfaces_missing_key_before_any_call(monkeypatch, probe_calls):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(MissingProviderKeyError):
        await effort_probe.probe_effort(CLAUDE_OPUS_55_MODEL_ID, "max", "hf_fake")

    assert probe_calls == []


# ── Contract lock: the request LiteLLM 1.83 actually sends ───────────────

_CANNED_ANTHROPIC_RESPONSE = {
    "id": "msg_contract",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [
        {"type": "thinking", "thinking": "", "signature": "SIG_RESPONSE"},
        {"type": "text", "text": "Listing the repo."},
        {
            "type": "tool_use",
            "id": "toolu_contract_2",
            "name": "bash",
            "input": {"command": "ls"},
        },
    ],
    "stop_reason": "tool_use",
    "stop_sequence": None,
    "usage": {
        "input_tokens": 100,
        "output_tokens": 50,
        "cache_read_input_tokens": 1000,
        "cache_creation_input_tokens": 200,
    },
}

_PRIOR_THINKING_BLOCK = {"type": "thinking", "thinking": "", "signature": "SIG_PRIOR"}

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read a file",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "bash",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {"command": {"type": "string"}},
            },
        },
    },
]


def _tool_loop_history() -> list:
    return [
        Message(role="system", content="You are ML Intern."),
        Message(role="user", content="What is in this repo?"),
        Message(
            role="assistant",
            content=None,
            tool_calls=[
                ChatCompletionMessageToolCall(
                    id="toolu_contract_1",
                    type="function",
                    function={"name": "bash", "arguments": '{"command": "pwd"}'},
                )
            ],
            thinking_blocks=[dict(_PRIOR_THINKING_BLOCK)],
        ),
        Message(
            role="tool",
            content="/workspace",
            tool_call_id="toolu_contract_1",
            name="bash",
        ),
    ]


def _count_cache_control(node) -> int:
    if isinstance(node, dict):
        own = 1 if "cache_control" in node else 0
        return own + sum(_count_cache_control(v) for v in node.values())
    if isinstance(node, list):
        return sum(_count_cache_control(v) for v in node)
    return 0


def _header(headers: dict, name: str):
    lowered = {str(k).lower(): v for k, v in headers.items()}
    return lowered.get(name.lower())


@pytest.fixture
def captured_requests(monkeypatch):
    """Patch LiteLLM's async HTTP client: capture the request, return canned JSON."""
    from litellm import anthropic_beta_headers_manager

    # Use the bundled beta-header allowlist: no GitHub fetch from this test.
    monkeypatch.setenv("LITELLM_LOCAL_ANTHROPIC_BETA_HEADERS", "True")
    monkeypatch.setattr(anthropic_beta_headers_manager, "_BETA_HEADERS_CONFIG", None)
    captured: list[dict] = []

    async def fake_post(self, url, data=None, json=None, headers=None, **kwargs):
        body = json
        if body is None:
            raw = data if data is not None else kwargs.get("content")
            body = json_module.loads(raw)
        captured.append({"url": str(url), "headers": dict(headers or {}), "body": body})
        return httpx.Response(
            200,
            content=json_module.dumps(_CANNED_ANTHROPIC_RESPONSE).encode(),
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(http_handler.AsyncHTTPHandler, "post", fake_post)
    return captured


# LiteLLM's background logging worker drops a pending success-callback
# coroutine when pytest-asyncio starts a new event loop per test.
_ignore_litellm_loop_change_warning = pytest.mark.filterwarnings(
    "ignore:coroutine 'Logging.async_success_handler' was never awaited:RuntimeWarning"
)


async def _send_contract_request():
    history = _tool_loop_history()
    params = _resolve_llm_params(CLAUDE_OPUS_55_MODEL_ID, reasoning_effort="high")
    params = with_prompt_cache_params(params, session_id="hf-session-123")
    messages, tools = with_prompt_caching(history, _TOOLS, params)
    response = await litellm.acompletion(
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=False,
        timeout=30,
        **params,
    )
    return history, response


@_ignore_litellm_loop_change_warning
async def test_contract_request_goes_to_api_anthropic_with_dummy_key(
    anthropic_env, captured_requests
):
    await _send_contract_request()

    assert len(captured_requests) == 1
    request = captured_requests[0]
    url = urlsplit(request["url"])
    assert url.hostname == "api.anthropic.com"
    assert url.path == "/v1/messages"
    assert _header(request["headers"], "x-api-key") == DUMMY_KEY
    assert _header(request["headers"], "X-HF-Session-id") is None


@_ignore_litellm_loop_change_warning
async def test_contract_request_body_matches_the_claude_55_api_surface(
    anthropic_env, captured_requests
):
    await _send_contract_request()

    body = captured_requests[0]["body"]
    assert body["model"] == "claude-opus-5-5"
    assert body["output_config"] == {"effort": "high"}
    assert body["max_tokens"] == 32000
    assert body["tool_choice"] == {"type": "auto"}
    for key in FORBIDDEN_DIRECT_KEYS:
        assert key not in body
    assert "prompt_cache_key" not in body


@_ignore_litellm_loop_change_warning
async def test_contract_request_uses_at_most_four_cache_breakpoints(
    anthropic_env, captured_requests
):
    await _send_contract_request()

    body = captured_requests[0]["body"]
    assert _count_cache_control(body) == 4
    assert "cache_control" in body["tools"][-1]
    assert "cache_control" in body["system"][-1]
    assert "cache_control" not in body["tools"][0]
    tool_result = body["messages"][-1]["content"][-1]
    assert tool_result["type"] == "tool_result"
    assert tool_result["cache_control"] == {"type": "ephemeral"}
    assert tool_result["content"] == "/workspace"


@_ignore_litellm_loop_change_warning
async def test_contract_request_echoes_signed_thinking_block_unchanged(
    anthropic_env, captured_requests
):
    history, _ = await _send_contract_request()

    body = captured_requests[0]["body"]
    assistant_turns = [m for m in body["messages"] if m["role"] == "assistant"]
    assert assistant_turns[-1]["content"][0] == _PRIOR_THINKING_BLOCK
    # The persisted history keeps plain string content and no markers.
    assert history[0].content == "You are ML Intern."
    assert history[3].content == "/workspace"


@_ignore_litellm_loop_change_warning
async def test_contract_response_exposes_signed_thinking_blocks_and_tool_call(
    anthropic_env, captured_requests
):
    _, response = await _send_contract_request()

    message = response.choices[0].message
    assert message.thinking_blocks[0]["signature"] == "SIG_RESPONSE"
    assert message.thinking_blocks[0]["thinking"] == ""
    assert message.tool_calls[0].function.name == "bash"
    assert response.choices[0].finish_reason == "tool_calls"
    assert litellm.completion_cost(completion_response=response) == pytest.approx(
        100 * 4e-6 + 50 * 2e-5 + 1000 * 2e-7 + 200 * 5e-6
    )


@_ignore_litellm_loop_change_warning
async def test_litellm_still_adds_effort_beta_header_for_direct_claude_55(
    anthropic_env, captured_requests
):
    # Tripwire, not an endorsement: LiteLLM 1.83 forces this legacy beta for
    # any non-4.6 model that sends output_config.effort. Whether Anthropic
    # accepts it on the 5.5 models is unverified offline. If a LiteLLM upgrade
    # changes this, re-check the direct route against the real API.
    await _send_contract_request()

    beta = _header(captured_requests[0]["headers"], "anthropic-beta") or ""
    assert "effort-2025-11-24" in beta
