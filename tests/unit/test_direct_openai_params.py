"""Direct OpenAI route: model ids, registration, LiteLLM kwargs, effort probe,
and an offline contract lock on what LiteLLM 1.83 sends and returns.

No test here talks to a real LLM provider. Every key is a dummy set with
monkeypatch (importing ``agent`` loads the repo .env, so each test pins the
provider variables itself). The wire tests patch ``httpx.AsyncClient.send``,
the layer under both LiteLLM's Responses API handler and the OpenAI SDK used
for the HF Router, return canned Responses API / Chat Completions streams, and
fail on any request nobody queued.
"""

import asyncio
import copy
import json
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import litellm
import pytest
from litellm import Message
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

from agent.config import Config
from agent.core import agent_loop, effort_probe, llm_params
from agent.core.agent_loop import Handlers, _call_llm_streaming
from agent.core.direct_openai import (
    DIRECT_OPENAI_LITELLM_MODELS,
    register_direct_openai_models,
)
from agent.core.llm_params import (
    MissingProviderKeyError,
    UnsupportedEffortError,
    _resolve_llm_params,
    with_output_token_cap,
)
from agent.core.model_ids import (
    ANTHROPIC_API_BASE_URL,
    CLAUDE_FABLE_52_MODEL_ID,
    CLAUDE_HAIKU_55_MODEL_ID,
    CLAUDE_OPUS_55_MODEL_ID,
    DIRECT_OPENAI_MODEL_IDS,
    DIRECT_PROVIDER_API_KEY_ENV,
    GPT_55_DIRECT_MODEL_ID,
    GPT_55_MODEL_ID,
    GPT_61_TERRA_MODEL_ID,
    GPT_62_ASTRA_MODEL_ID,
    GPT_62_SOL_MODEL_ID,
    GPT_6_LUNA_MODEL_ID,
    HF_ROUTER_BASE_URL,
    HOSTED_MODEL_IDS,
    OPENAI_API_BASE_URL,
    direct_provider_for,
    is_direct_openai_model_id,
)
from agent.core.prompt_caching import (
    HF_ROUTER_SESSION_ID_HEADER,
    with_prompt_cache_params,
)
from agent.core.session import Session, _get_max_tokens_safe

DUMMY_KEY = "sk-openai-test"
DUMMY_HF_TOKEN = "hf_test_dummy"
BOGUS_BASE_URL = "https://bogus.invalid/v1"
HF_TOKEN_PLACEHOLDER = "hf-token-missing"
MISSING_KEY_MESSAGE = (
    "GPT via the OpenAI API needs OPENAI_API_KEY. "
    "Add it to the repo .env and restart the backend."
)
DIRECT_IDS = (
    GPT_55_DIRECT_MODEL_ID,
    GPT_62_ASTRA_MODEL_ID,
    GPT_62_SOL_MODEL_ID,
    GPT_61_TERRA_MODEL_ID,
    GPT_6_LUNA_MODEL_ID,
)
GPT_6_IDS = DIRECT_IDS[1:]
# Keys the direct OpenAI kwargs must never carry at the top level.
FORBIDDEN_DIRECT_KEYS = (
    "max_tokens",
    "temperature",
    "top_p",
    "reasoning_effort",
    "store",
    "prompt_cache_key",
    "prompt_cache_retention",
    "allowed_openai_params",
    "extra_headers",
    "output_config",
    "thinking",
)


@pytest.fixture(autouse=True)
def provider_env(monkeypatch):
    """Dummy keys only; inherited OpenAI base-URL overrides must be ignored."""
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-ds-test")
    monkeypatch.setenv("HF_TOKEN", DUMMY_HF_TOKEN)
    monkeypatch.setenv("OPENAI_BASE_URL", BOGUS_BASE_URL)
    monkeypatch.setenv("OPENAI_API_BASE", BOGUS_BASE_URL)
    for name in ("OPENAI_ORGANIZATION", "OPENAI_ORG_ID", "OPENAI_PROJECT_ID"):
        monkeypatch.delenv(name, raising=False)


def _no_hf_token(monkeypatch) -> None:
    import huggingface_hub

    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.delenv("HUGGING_FACE_HUB_TOKEN", raising=False)
    monkeypatch.setattr(huggingface_hub, "get_token", lambda: None)


def _fail_hf_token_lookup(*_args, **_kwargs):
    raise AssertionError("direct OpenAI must not resolve an HF token")


def _expected_params(model_id: str, wire_effort=None) -> dict:
    extra_body: dict = {"store": False}
    if wire_effort is not None:
        extra_body["reasoning_effort"] = wire_effort
    return {
        "model": model_id,
        "api_base": OPENAI_API_BASE_URL,
        "api_key": DUMMY_KEY,
        "max_completion_tokens": 32000,
        "extra_body": extra_body,
    }


# ── (a) exact-membership allowlist ───────────────────────────────────────


def test_allowlist_holds_exactly_the_five_direct_ids():
    assert DIRECT_OPENAI_MODEL_IDS == frozenset(DIRECT_IDS)
    assert GPT_55_DIRECT_MODEL_ID == "openai/gpt-5.5"
    assert GPT_62_ASTRA_MODEL_ID == "openai/gpt-6.2-astra"
    assert GPT_62_SOL_MODEL_ID == "openai/gpt-6.2-sol"
    assert GPT_61_TERRA_MODEL_ID == "openai/gpt-6.1-terra"
    assert GPT_6_LUNA_MODEL_ID == "openai/gpt-6-luna"
    assert OPENAI_API_BASE_URL == "https://api.openai.com/v1"
    assert DIRECT_PROVIDER_API_KEY_ENV["openai"] == "OPENAI_API_KEY"


@pytest.mark.parametrize("model_id", DIRECT_IDS)
def test_direct_ids_match_and_name_the_openai_provider(model_id):
    assert is_direct_openai_model_id(model_id) is True
    assert direct_provider_for(model_id) == "openai"
    assert model_id not in HOSTED_MODEL_IDS


@pytest.mark.parametrize(
    "model_id",
    [
        GPT_55_MODEL_ID,
        "huggingface/openai/gpt-5.5",
        "openai/gpt-5.4",
        "openai/responses/gpt-6.2-sol",
        "openai/gpt-oss-120b:cerebras",
        "gpt-6.2-sol",
        "",
        None,
    ],
)
def test_router_prefixed_and_unknown_ids_are_not_direct_openai(model_id):
    assert is_direct_openai_model_id(model_id) is False
    assert direct_provider_for(model_id) is None


def test_router_gpt55_id_is_unchanged():
    assert GPT_55_MODEL_ID == "openai/gpt-5.5:fal-ai"
    assert GPT_55_MODEL_ID in HOSTED_MODEL_IDS


# ── (b) model registration ───────────────────────────────────────────────

# (input, output, cache read, cache write or None) and the >272K tier
# (input, output, cache read), per token.
PRICES = {
    GPT_55_DIRECT_MODEL_ID: ((5e-6, 3e-5, 5e-7, None), (1e-5, 4.5e-5, 1e-6)),
    GPT_62_ASTRA_MODEL_ID: ((8e-6, 4e-5, 4e-7, 1e-5), (1.6e-5, 6e-5, 8e-7)),
    GPT_62_SOL_MODEL_ID: ((2e-6, 1e-5, 1e-7, 2.5e-6), (4e-6, 1.5e-5, 2e-7)),
    GPT_61_TERRA_MODEL_ID: ((1e-6, 5e-6, 5e-8, 1.25e-6), (2e-6, 7.5e-6, 1e-7)),
    GPT_6_LUNA_MODEL_ID: ((1e-7, 5e-7, 1e-8, 1.25e-7), (2e-7, 7.5e-7, 2e-8)),
}
SUPPORTS_NONE_EFFORT = {
    "gpt-5.5": True,
    "gpt-6.2-astra": False,
    "gpt-6.2-sol": False,
    "gpt-6.1-terra": True,
    "gpt-6-luna": True,
}


def test_registration_data_is_keyed_by_bare_litellm_model_names():
    assert set(DIRECT_OPENAI_LITELLM_MODELS) == set(SUPPORTS_NONE_EFFORT)


@pytest.mark.parametrize("model_id", DIRECT_IDS)
def test_registered_models_always_use_the_native_responses_api(model_id):
    info = litellm.get_model_info(model_id)

    assert info["litellm_provider"] == "openai"
    assert info["mode"] == "responses"
    assert info["supports_native_streaming"] is True
    assert info["supports_tool_choice"] is True
    assert info["supports_function_calling"] is True
    assert info["supports_system_messages"] is True
    assert info["supports_prompt_caching"] is True
    assert info["supports_reasoning"] is True
    assert info["max_input_tokens"] == 272_000
    assert info["max_output_tokens"] == 128_000
    # get_model_info doesn't surface this flag; the map entry carries it.
    entry = litellm.model_cost[model_id.removeprefix("openai/")]
    assert entry["supports_parallel_function_calling"] is True


@pytest.mark.parametrize("name", sorted(SUPPORTS_NONE_EFFORT))
def test_registered_effort_flags_match_each_model(name):
    entry = litellm.model_cost[name]

    assert entry["supports_minimal_reasoning_effort"] is False
    assert entry["supports_xhigh_reasoning_effort"] is True
    assert entry["supports_none_reasoning_effort"] is SUPPORTS_NONE_EFFORT[name]
    assert (
        DIRECT_OPENAI_LITELLM_MODELS[name]["supports_none_reasoning_effort"]
        is (SUPPORTS_NONE_EFFORT[name])
    )


@pytest.mark.parametrize("model_id", DIRECT_IDS)
def test_registered_prices_match_the_table(model_id):
    (rate_in, rate_out, cache_read, cache_write), tier = PRICES[model_id]
    name = model_id.removeprefix("openai/")
    source = DIRECT_OPENAI_LITELLM_MODELS[name]
    entry = litellm.model_cost[name]

    expected = {
        "input_cost_per_token": rate_in,
        "output_cost_per_token": rate_out,
        "cache_read_input_token_cost": cache_read,
        "input_cost_per_token_above_272k_tokens": tier[0],
        "output_cost_per_token_above_272k_tokens": tier[1],
        "cache_read_input_token_cost_above_272k_tokens": tier[2],
    }
    if cache_write is not None:
        expected["cache_creation_input_token_cost"] = cache_write
    else:
        # GPT-5.5 has no cache-write premium, so no cache-write rate is set.
        assert "cache_creation_input_token_cost" not in source
    for key, value in expected.items():
        assert source[key] == pytest.approx(value), key
        assert entry[key] == pytest.approx(value), key


@pytest.mark.parametrize("model_id", DIRECT_IDS)
def test_cost_per_token_bills_cached_and_uncached_input_separately(model_id):
    (rate_in, rate_out, cache_read, _), _tier = PRICES[model_id]
    # OpenAI usage folds cached tokens into prompt_tokens (600 + 400 cached).
    usage = Usage(
        prompt_tokens=1000,
        completion_tokens=200,
        total_tokens=1200,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=400),
    )

    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model_id, usage_object=usage
    )

    assert prompt_cost + completion_cost == pytest.approx(
        600 * rate_in + 400 * cache_read + 200 * rate_out
    )


@pytest.mark.parametrize("model_id", DIRECT_IDS)
def test_cost_per_token_switches_to_the_long_context_tier_above_272k(model_id):
    _base, (tier_in, tier_out, tier_cache_read) = PRICES[model_id]
    usage = Usage(
        prompt_tokens=300_000,
        completion_tokens=1000,
        total_tokens=301_000,
        prompt_tokens_details=PromptTokensDetailsWrapper(cached_tokens=100_000),
    )

    prompt_cost, completion_cost = litellm.cost_per_token(
        model=model_id, usage_object=usage
    )

    assert prompt_cost + completion_cost == pytest.approx(
        200_000 * tier_in + 100_000 * tier_cache_read + 1000 * tier_out
    )


def test_registration_overrides_a_stale_remote_map_entry(monkeypatch):
    stale = {
        **litellm.model_cost["gpt-6.2-sol"],
        "mode": "chat",
        "supports_native_streaming": False,
        "input_cost_per_token": 9e-6,
        "max_input_tokens": 1_050_000,
    }
    monkeypatch.setitem(litellm.model_cost, "gpt-6.2-sol", stale)

    register_direct_openai_models()

    info = litellm.get_model_info(GPT_62_SOL_MODEL_ID)
    assert info["mode"] == "responses"
    assert info["supports_native_streaming"] is True
    assert info["input_cost_per_token"] == pytest.approx(2e-6)
    assert info["max_input_tokens"] == 272_000


def test_registration_passes_copies_to_litellm():
    register_direct_openai_models()

    for name, info in DIRECT_OPENAI_LITELLM_MODELS.items():
        assert litellm.model_cost[name] is not info


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        (GPT_62_SOL_MODEL_ID, 272_000),
        (GPT_55_DIRECT_MODEL_ID, 272_000),
        (CLAUDE_HAIKU_55_MODEL_ID, 100_000),
        (CLAUDE_FABLE_52_MODEL_ID, 1_000_000),
        # Documented side effect: the router id's bare name now resolves to
        # the registered "gpt-5.5" entry.
        (GPT_55_MODEL_ID, 272_000),
    ],
)
def test_context_ceiling_used_for_compaction(model_id, expected):
    assert _get_max_tokens_safe(model_id) == expected


# ── (c) _resolve_llm_params: direct OpenAI branch ────────────────────────


@pytest.mark.parametrize("model_id", DIRECT_IDS)
@pytest.mark.parametrize(
    ("effort", "wire_effort"),
    [
        ("minimal", "low"),
        ("low", "low"),
        ("medium", "medium"),
        ("high", "high"),
        ("xhigh", "xhigh"),
    ],
)
@pytest.mark.parametrize("strict", [True, False])
def test_direct_params_send_string_efforts_in_extra_body(
    model_id, effort, wire_effort, strict
):
    params = _resolve_llm_params(model_id, reasoning_effort=effort, strict=strict)

    assert params == _expected_params(model_id, wire_effort)


@pytest.mark.parametrize("model_id", GPT_6_IDS)
@pytest.mark.parametrize("strict", [True, False])
def test_gpt_6_models_send_max_as_a_dict(model_id, strict):
    # The bridge drops the string "max"; the dict passes through as-is.
    params = _resolve_llm_params(model_id, reasoning_effort="max", strict=strict)

    assert params == _expected_params(model_id, {"effort": "max"})


def test_gpt55_rejects_max_in_strict_mode():
    with pytest.raises(UnsupportedEffortError, match="max"):
        _resolve_llm_params(GPT_55_DIRECT_MODEL_ID, reasoning_effort="max", strict=True)


def test_gpt55_clamps_max_to_xhigh_in_non_strict_mode():
    params = _resolve_llm_params(GPT_55_DIRECT_MODEL_ID, reasoning_effort="max")

    assert params == _expected_params(GPT_55_DIRECT_MODEL_ID, "xhigh")


@pytest.mark.parametrize("model_id", DIRECT_IDS)
@pytest.mark.parametrize("effort", ["turbo", "none"])
def test_unknown_effort_raises_in_strict_mode(model_id, effort):
    with pytest.raises(UnsupportedEffortError):
        _resolve_llm_params(model_id, reasoning_effort=effort, strict=True)


@pytest.mark.parametrize("model_id", DIRECT_IDS)
@pytest.mark.parametrize("effort", ["turbo", "none"])
def test_unknown_effort_is_omitted_in_non_strict_mode(model_id, effort):
    params = _resolve_llm_params(model_id, reasoning_effort=effort)

    assert params == _expected_params(model_id)


@pytest.mark.parametrize("model_id", DIRECT_IDS)
def test_no_effort_sends_only_store_false(model_id):
    params = _resolve_llm_params(model_id, reasoning_effort=None)

    assert params == _expected_params(model_id)


@pytest.mark.parametrize("model_id", DIRECT_IDS)
@pytest.mark.parametrize("effort", [None, "minimal", "high", "xhigh", "max", "turbo"])
def test_direct_params_never_emit_rejected_fields(model_id, effort):
    params = _resolve_llm_params(model_id, reasoning_effort=effort)

    for key in FORBIDDEN_DIRECT_KEYS:
        assert key not in params
    assert set(params["extra_body"]) <= {"store", "reasoning_effort"}
    assert params["extra_body"].get("reasoning_effort") != "none"
    assert params["model"] == model_id


@pytest.mark.parametrize("model_id", DIRECT_IDS)
@pytest.mark.parametrize("raw_key", [None, "", "   ", "\n\t"])
def test_missing_key_raises_missing_provider_key_error(monkeypatch, model_id, raw_key):
    if raw_key is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", raw_key)

    with pytest.raises(MissingProviderKeyError) as excinfo:
        _resolve_llm_params(model_id, reasoning_effort="high")

    assert str(excinfo.value) == MISSING_KEY_MESSAGE
    assert isinstance(excinfo.value, RuntimeError)


def test_missing_key_is_reported_before_effort_validation(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingProviderKeyError):
        _resolve_llm_params(GPT_55_DIRECT_MODEL_ID, reasoning_effort="max", strict=True)


def test_direct_params_strip_whitespace_around_the_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", f"  {DUMMY_KEY}\n")

    params = _resolve_llm_params(GPT_62_SOL_MODEL_ID)

    assert params["api_key"] == DUMMY_KEY


def test_direct_params_ignore_inherited_openai_base_urls():
    params = _resolve_llm_params(GPT_62_SOL_MODEL_ID, reasoning_effort="high")

    assert params["api_base"] == OPENAI_API_BASE_URL


@pytest.mark.parametrize("model_id", DIRECT_IDS)
def test_direct_params_never_resolve_hf_tokens(monkeypatch, model_id):
    monkeypatch.setattr(llm_params, "resolve_hf_router_token", _fail_hf_token_lookup)

    params = _resolve_llm_params(model_id, "hf_session_token", reasoning_effort="high")

    assert params["api_key"] == DUMMY_KEY


def test_router_gpt55_still_routes_through_hf_router():
    params = _resolve_llm_params(
        GPT_55_MODEL_ID, "hf_session_token", reasoning_effort="high"
    )

    assert params == {
        "model": f"openai/{GPT_55_MODEL_ID}",
        "api_base": HF_ROUTER_BASE_URL,
        "api_key": "hf_session_token",
        "extra_body": {"reasoning_effort": "high"},
    }


@pytest.mark.parametrize(
    "model_id", ["huggingface/openai/gpt-5.5", "openai/gpt-oss-120b:cerebras"]
)
def test_other_openai_prefixed_ids_still_route_through_hf_router(model_id):
    params = _resolve_llm_params(model_id, "hf_session_token")

    assert params["model"] == f"openai/{model_id.removeprefix('huggingface/')}"
    assert params["api_base"] == HF_ROUTER_BASE_URL
    assert params["api_key"] == "hf_session_token"


# ── (d) HF Router never falls back to OPENAI_API_KEY ─────────────────────


@pytest.mark.parametrize(
    "model_id",
    [
        "moonshotai/Kimi-K2.7-Code:novita",
        GPT_55_MODEL_ID,
        "openai/gpt-oss-120b:cerebras",
    ],
)
def test_hf_router_without_a_token_uses_a_placeholder_key(monkeypatch, model_id):
    _no_hf_token(monkeypatch)

    params = _resolve_llm_params(model_id)

    assert params["api_key"] == HF_TOKEN_PLACEHOLDER
    assert params["api_base"] == HF_ROUTER_BASE_URL


# ── (e) with_output_token_cap ────────────────────────────────────────────


def test_output_cap_sets_max_completion_tokens_for_direct_openai():
    params = _resolve_llm_params(GPT_62_SOL_MODEL_ID, reasoning_effort="high")
    snapshot = copy.deepcopy(params)

    capped = with_output_token_cap(params, 64)

    assert capped == {**snapshot, "max_completion_tokens": 64}
    assert "max_tokens" not in capped
    assert params == snapshot
    assert capped is not params


def test_output_cap_drops_a_stray_max_tokens_for_direct_openai():
    params = {
        "model": GPT_6_LUNA_MODEL_ID,
        "max_tokens": 10,
        "max_completion_tokens": 9,
    }

    capped = with_output_token_cap(params, 64)

    assert capped == {"model": GPT_6_LUNA_MODEL_ID, "max_completion_tokens": 64}
    assert params == {
        "model": GPT_6_LUNA_MODEL_ID,
        "max_tokens": 10,
        "max_completion_tokens": 9,
    }


@pytest.mark.parametrize(
    "params",
    [
        {
            "model": CLAUDE_OPUS_55_MODEL_ID,
            "api_base": ANTHROPIC_API_BASE_URL,
            "max_tokens": 32000,
        },
        {"model": f"openai/{GPT_55_MODEL_ID}", "api_base": HF_ROUTER_BASE_URL},
        {"model": "deepseek/deepseek-v4-pro"},
    ],
    ids=["direct-anthropic", "router-gpt55", "deepseek"],
)
def test_output_cap_sets_max_tokens_on_every_other_route(params):
    snapshot = copy.deepcopy(params)

    capped = with_output_token_cap(params, 64)

    assert capped == {**snapshot, "max_tokens": 64}
    assert params == snapshot


# ── (f) wire contract: what LiteLLM 1.83 sends and returns ───────────────

_TOOLS = [
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
_RESPONSES_USAGE = {
    "input_tokens": 1000,
    "input_tokens_details": {"cached_tokens": 400},
    "output_tokens": 200,
    "output_tokens_details": {"reasoning_tokens": 50},
    "total_tokens": 1200,
}
_REASONING_ITEM = {
    "type": "reasoning",
    "id": "rs_1",
    "summary": [],
    "encrypted_content": "ENC_REASONING",
}
_TOOL_ARGUMENTS = '{"command": "ls"}'


def _sol_cost(prompt: int, cached: int, completion: int) -> float:
    (rate_in, rate_out, cache_read, _), _tier = PRICES[GPT_62_SOL_MODEL_ID]
    return (prompt - cached) * rate_in + cached * cache_read + completion * rate_out


def _response_object(status: str, output: list, usage=None, details=None) -> dict:
    response = {
        "id": "resp_test",
        "object": "response",
        "created_at": 1_700_000_000,
        "status": status,
        # OpenAI echoes a dated snapshot id; cost must still use the registry.
        "model": "gpt-6.2-sol-2026-09-30",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "text": {"format": {"type": "text"}},
        "error": None,
        "incomplete_details": details,
    }
    if usage is not None:
        response["usage"] = usage
    return response


def _sse_events(events: list[dict]) -> bytes:
    lines = []
    for sequence, event in enumerate(events):
        payload = {**event, "sequence_number": sequence}
        lines.append(f"event: {event['type']}\ndata: {json.dumps(payload)}\n\n")
    return "".join(lines).encode()


def _function_call_stream(call_id: str = "call_1") -> bytes:
    call = {
        "type": "function_call",
        "id": "fc_1",
        "call_id": call_id,
        "name": "bash",
        "arguments": _TOOL_ARGUMENTS,
        "status": "completed",
    }
    return _sse_events(
        [
            {
                "type": "response.created",
                "response": _response_object("in_progress", []),
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1", "summary": []},
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": _REASONING_ITEM,
            },
            {
                "type": "response.output_item.added",
                "output_index": 1,
                "item": {**call, "arguments": "", "status": "in_progress"},
            },
            {
                "type": "response.function_call_arguments.delta",
                "output_index": 1,
                "item_id": "fc_1",
                "delta": _TOOL_ARGUMENTS,
            },
            {
                "type": "response.function_call_arguments.done",
                "output_index": 1,
                "item_id": "fc_1",
                "arguments": _TOOL_ARGUMENTS,
            },
            {"type": "response.output_item.done", "output_index": 1, "item": call},
            {
                "type": "response.completed",
                "response": _response_object(
                    "completed", [_REASONING_ITEM, call], _RESPONSES_USAGE
                ),
            },
        ]
    )


def _text_stream(text: str) -> bytes:
    message = {
        "type": "message",
        "id": "msg_1",
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }
    part = {"type": "output_text", "text": "", "annotations": []}
    return _sse_events(
        [
            {
                "type": "response.created",
                "response": _response_object("in_progress", []),
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**message, "status": "in_progress", "content": []},
            },
            {
                "type": "response.content_part.added",
                "output_index": 0,
                "item_id": "msg_1",
                "content_index": 0,
                "part": part,
            },
            {
                "type": "response.output_text.delta",
                "output_index": 0,
                "item_id": "msg_1",
                "content_index": 0,
                "delta": text,
                "logprobs": [],
            },
            {
                "type": "response.output_text.done",
                "output_index": 0,
                "item_id": "msg_1",
                "content_index": 0,
                "text": text,
                "logprobs": [],
            },
            {"type": "response.output_item.done", "output_index": 0, "item": message},
            {
                "type": "response.completed",
                "response": _response_object("completed", [message], _RESPONSES_USAGE),
            },
        ]
    )


def _incomplete_stream() -> bytes:
    """The output cap was spent on reasoning before any reply was written."""
    return _sse_events(
        [
            {
                "type": "response.created",
                "response": _response_object("in_progress", []),
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {"type": "reasoning", "id": "rs_1", "summary": []},
            },
            {
                "type": "response.output_item.done",
                "output_index": 0,
                "item": _REASONING_ITEM,
            },
            {
                "type": "response.incomplete",
                "response": _response_object(
                    "incomplete",
                    [_REASONING_ITEM],
                    _RESPONSES_USAGE,
                    {"reason": "max_output_tokens"},
                ),
            },
        ]
    )


def _chat_stream(text: str, model: str) -> bytes:
    def chunk(delta, finish_reason=None):
        return {
            "id": "chatcmpl-1",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": model,
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }

    usage = {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": model,
        "choices": [],
        "usage": {"prompt_tokens": 50, "completion_tokens": 2, "total_tokens": 52},
    }
    payloads = [chunk({"role": "assistant", "content": text}), chunk({}, "stop"), usage]
    lines = [f"data: {json.dumps(payload)}\n\n" for payload in payloads]
    return ("".join(lines) + "data: [DONE]\n\n").encode()


class FakeProviderHttp:
    """Queue of canned streams keyed by URL; records every outgoing request."""

    def __init__(self):
        self.requests: list[dict] = []
        self._queue: list[tuple[str, bytes]] = []

    def queue(self, url: str, stream: bytes) -> None:
        self._queue.append((url, stream))

    def for_host(self, host: str) -> list[dict]:
        return [r for r in self.requests if urlsplit(r["url"]).hostname == host]

    async def send(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append(
            {
                "method": request.method,
                "url": url,
                "headers": {k.lower(): v for k, v in request.headers.items()},
                "raw": request.content,
                "body": json.loads(request.content),
            }
        )
        if not self._queue:
            raise AssertionError(f"Unexpected request to {url}")
        expected_url, stream = self._queue.pop(0)
        if url != expected_url:
            raise AssertionError(f"Expected a request to {expected_url}, got {url}")
        return httpx.Response(
            200,
            content=stream,
            headers={"content-type": "text/event-stream"},
            request=request,
        )


@pytest.fixture
def provider_http(monkeypatch):
    fake = FakeProviderHttp()

    async def send(_client, request, *_args, **_kwargs):
        return await fake.send(request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return fake


RESPONSES_URL = f"{OPENAI_API_BASE_URL}/responses"
ROUTER_CHAT_URL = f"{HF_ROUTER_BASE_URL}/chat/completions"

# LiteLLM's background logging worker drops a pending success-callback
# coroutine when pytest-asyncio starts a new event loop per test.
_ignore_litellm_loop_change_warning = pytest.mark.filterwarnings(
    "ignore:coroutine 'Logging.async_success_handler' was never awaited:RuntimeWarning"
)


def _recording_session(model_name: str = GPT_62_SOL_MODEL_ID) -> SimpleNamespace:
    events: list = []

    async def send_event(event):
        events.append(event)

    return SimpleNamespace(
        config=SimpleNamespace(model_name=model_name),
        session_id="session-1",
        is_cancelled=False,
        send_event=send_event,
        events=events,
    )


async def _stream_direct_turn(model_id: str, effort: str | None):
    session = _recording_session(model_id)
    params = _resolve_llm_params(model_id, DUMMY_HF_TOKEN, reasoning_effort=effort)
    result = await _call_llm_streaming(
        session,
        [
            Message(role="system", content="You are ML Intern."),
            Message(role="user", content="What is in this repo?"),
        ],
        _TOOLS,
        params,
    )
    return session, result


@_ignore_litellm_loop_change_warning
async def test_direct_request_posts_to_openai_responses_with_the_dummy_key(
    provider_http,
):
    provider_http.queue(RESPONSES_URL, _function_call_stream())

    await _stream_direct_turn(GPT_62_SOL_MODEL_ID, "high")

    (request,) = provider_http.requests
    assert request["method"] == "POST"
    assert request["url"] == RESPONSES_URL
    assert request["headers"]["authorization"] == f"Bearer {DUMMY_KEY}"
    assert HF_ROUTER_SESSION_ID_HEADER.lower() not in request["headers"]
    assert "openai-organization" not in request["headers"]


@_ignore_litellm_loop_change_warning
async def test_direct_request_body_matches_the_responses_api_surface(provider_http):
    provider_http.queue(RESPONSES_URL, _function_call_stream())

    await _stream_direct_turn(GPT_62_SOL_MODEL_ID, "high")

    body = provider_http.requests[0]["body"]
    assert body["model"] == "gpt-6.2-sol"
    assert body["reasoning"] == {"effort": "high"}
    assert body["stream"] is True
    assert body["tool_choice"] == "auto"
    assert body["max_output_tokens"] == 32000
    assert body["store"] is False
    assert body["instructions"] == "You are ML Intern."
    assert body["tools"][0]["name"] == "bash"
    for key in (
        "temperature",
        "top_p",
        "prompt_cache_key",
        "prompt_cache_retention",
        "max_tokens",
        "max_completion_tokens",
        "reasoning_effort",
        "messages",
    ):
        assert key not in body


@_ignore_litellm_loop_change_warning
async def test_streamed_tool_call_is_rebuilt_and_priced(provider_http):
    provider_http.queue(RESPONSES_URL, _function_call_stream())

    session, result = await _stream_direct_turn(GPT_62_SOL_MODEL_ID, "high")

    (tool_call,) = result.tool_calls_acc.values()
    assert tool_call["id"] == "call_1"
    assert tool_call["function"] == {"name": "bash", "arguments": _TOOL_ARGUMENTS}
    assert result.finish_reason == "tool_calls"
    assert result.usage["cost_usd"] == pytest.approx(_sol_cost(1000, 400, 200))
    assert result.usage["cost_usd"] > 0
    (llm_call,) = [e for e in session.events if e.event_type == "llm_call"]
    assert llm_call.data["model"] == GPT_62_SOL_MODEL_ID


@_ignore_litellm_loop_change_warning
async def test_astra_sends_max_effort_through_the_bridge(provider_http):
    provider_http.queue(RESPONSES_URL, _function_call_stream())

    await _stream_direct_turn(GPT_62_ASTRA_MODEL_ID, "max")

    body = provider_http.requests[0]["body"]
    assert body["model"] == "gpt-6.2-astra"
    assert body["reasoning"] == {"effort": "max"}


@_ignore_litellm_loop_change_warning
async def test_gpt55_without_effort_still_uses_the_responses_api(provider_http):
    provider_http.queue(RESPONSES_URL, _function_call_stream())

    await _stream_direct_turn(GPT_55_DIRECT_MODEL_ID, None)

    body = provider_http.requests[0]["body"]
    assert body["model"] == "gpt-5.5"
    assert "reasoning" not in body
    assert body["store"] is False
    assert body["max_output_tokens"] == 32000


_ROUTER_MESSAGES = [
    {"role": "system", "content": "You are ML Intern."},
    {"role": "user", "content": "What is in this repo?"},
]


async def _send_router_gpt55_request(provider_http, session_id: str = "sess-123"):
    provider_http.queue(ROUTER_CHAT_URL, _chat_stream("Hi.", GPT_55_MODEL_ID))
    params = _resolve_llm_params(
        GPT_55_MODEL_ID, DUMMY_HF_TOKEN, reasoning_effort="high"
    )
    params = with_prompt_cache_params(params, session_id=session_id)
    response = await litellm.acompletion(
        messages=_ROUTER_MESSAGES,
        tools=_TOOLS,
        tool_choice="auto",
        stream=True,
        stream_options={"include_usage": True},
        timeout=30,
        **params,
    )
    async for _chunk in response:
        pass
    return provider_http.requests[-1]


async def _router_request_without_gpt55_registration(provider_http):
    saved = litellm.model_cost.pop("gpt-5.5")
    litellm.utils._invalidate_model_cost_lowercase_map()
    try:
        return await _send_router_gpt55_request(provider_http)
    finally:
        litellm.model_cost["gpt-5.5"] = saved
        register_direct_openai_models()


@_ignore_litellm_loop_change_warning
async def test_router_gpt55_request_is_unchanged_by_the_direct_registration(
    provider_http,
):
    registered = await _send_router_gpt55_request(provider_http)
    unregistered = await _router_request_without_gpt55_registration(provider_http)

    assert registered["url"] == ROUTER_CHAT_URL
    assert registered["headers"]["authorization"] == f"Bearer {DUMMY_HF_TOKEN}"
    assert registered["headers"][HF_ROUTER_SESSION_ID_HEADER.lower()] == "sess-123"
    assert registered["body"] == {
        "messages": _ROUTER_MESSAGES,
        "model": GPT_55_MODEL_ID,
        "stream": True,
        "stream_options": {"include_usage": True},
        "tools": _TOOLS,
        "reasoning_effort": "high",
        "prompt_cache_key": "sess-123",
        "prompt_cache_retention": "24h",
    }
    assert registered["raw"] == unregistered["raw"]


@_ignore_litellm_loop_change_warning
async def test_router_call_without_hf_token_never_carries_the_openai_key(
    monkeypatch, provider_http
):
    _no_hf_token(monkeypatch)
    provider_http.queue(ROUTER_CHAT_URL, _chat_stream("Hi.", "zai-org/GLM-5.2:novita"))
    params = _resolve_llm_params("zai-org/GLM-5.2:novita", None)

    response = await litellm.acompletion(
        messages=_ROUTER_MESSAGES, stream=True, timeout=30, **params
    )
    async for _chunk in response:
        pass

    (request,) = provider_http.requests
    assert request["url"] == ROUTER_CHAT_URL
    assert request["headers"]["authorization"] == f"Bearer {HF_TOKEN_PLACEHOLDER}"


# End-to-end through the agent loop and the real bridge.


class BashToolRouter:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def get_tool_specs_for_llm(self):
        return _TOOLS

    async def call_tool(self, name, arguments, session=None, tool_call_id=None):
        self.calls.append((name, arguments))
        return "file.txt", True


def _make_session(model_name: str = GPT_62_SOL_MODEL_ID) -> Session:
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


@_ignore_litellm_loop_change_warning
async def test_two_turn_tool_loop_runs_through_the_bridge(provider_http):
    session = _make_session()
    provider_http.queue(RESPONSES_URL, _function_call_stream())
    provider_http.queue(RESPONSES_URL, _text_stream("Found file.txt."))

    final = await Handlers.run_agent(session, "list files")

    assert final == "Found file.txt."
    assert session.tool_router.calls == [("bash", {"command": "ls"})]
    second = provider_http.requests[1]["body"]
    item_types = [item.get("type") for item in second["input"]]
    assert "function_call" in item_types
    assert "reasoning" not in item_types
    output = next(i for i in second["input"] if i["type"] == "function_call_output")
    assert output["call_id"] == "call_1"
    assert "file.txt" in json.dumps(output["output"])
    assert second["store"] is False
    assert second["reasoning"] == {"effort": "high"}
    llm_calls = [e for e in _drain_events(session) if e.event_type == "llm_call"]
    assert [e.data["model"] for e in llm_calls] == [GPT_62_SOL_MODEL_ID] * 2
    assert all(e.data["cost_usd"] > 0 for e in llm_calls)


@_ignore_litellm_loop_change_warning
async def test_stream_cut_off_by_the_output_cap_surfaces_an_error(provider_http):
    # LiteLLM's bridge drops ``response.incomplete`` (and its usage): the
    # stream ends as an empty "stop". The turn must not end silently.
    session = _make_session()
    provider_http.queue(RESPONSES_URL, _incomplete_stream())

    final = await Handlers.run_agent(session, "think very hard")

    errors = [
        e.data["error"] for e in _drain_events(session) if e.event_type == "error"
    ]
    assert final is None
    assert len(errors) == 1
    assert "output token limit" in errors[0]
    assert [m.role for m in session.context_manager.items] == ["system", "user"]


# ── (g) effort probe on the direct route ─────────────────────────────────


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


async def test_probe_skips_max_on_gpt55_without_a_call_and_sends_xhigh(probe_calls):
    outcome = await effort_probe.probe_effort(GPT_55_DIRECT_MODEL_ID, "max", "hf_fake")

    assert outcome.effective_effort == "xhigh"
    assert outcome.attempts == 1
    assert outcome.note == "max not supported, using xhigh"
    (call,) = probe_calls
    assert call["extra_body"] == {"store": False, "reasoning_effort": "xhigh"}
    assert call["max_completion_tokens"] == effort_probe._PROBE_MAX_TOKENS == 64
    assert "max_tokens" not in call


@pytest.mark.parametrize("model_id", GPT_6_IDS)
async def test_probe_accepts_max_on_gpt_6_models(probe_calls, model_id):
    outcome = await effort_probe.probe_effort(model_id, "max", "hf_fake")

    assert outcome.effective_effort == "max"
    assert outcome.note is None
    (call,) = probe_calls
    assert call["extra_body"] == {"store": False, "reasoning_effort": {"effort": "max"}}


async def test_probe_surfaces_a_missing_key_before_any_call(monkeypatch, probe_calls):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    with pytest.raises(MissingProviderKeyError):
        await effort_probe.probe_effort(GPT_55_DIRECT_MODEL_ID, "max", "hf_fake")

    assert probe_calls == []


async def test_probe_on_direct_route_sends_no_hf_session_header(probe_calls):
    session = SimpleNamespace(session_id="session-1")

    await effort_probe.probe_effort(GPT_62_SOL_MODEL_ID, "high", "hf_fake")
    await effort_probe.probe_effort(
        GPT_62_SOL_MODEL_ID, "high", "hf_fake", session=session
    )

    for call in probe_calls:
        assert "extra_headers" not in call
        assert call["api_base"] == OPENAI_API_BASE_URL
        assert call["model"] == GPT_62_SOL_MODEL_ID
        assert call["api_key"] == DUMMY_KEY


async def test_probe_counts_a_reply_cut_off_by_the_cap_as_accepted(monkeypatch):
    # The bridge raises when OpenAI returns 200 "incomplete" with no text:
    # the effort was accepted, the 64-token probe budget just ran out.
    calls: list[dict] = []

    async def cut_off(**kwargs):
        calls.append(kwargs)
        raise litellm.APIConnectionError(
            message="OpenAIException - gpt-6.2-sol unable to complete request: "
            "max_output_tokens",
            llm_provider="openai",
            model="gpt-6.2-sol",
        )

    monkeypatch.setattr(effort_probe, "acompletion", cut_off)

    outcome = await effort_probe.probe_effort(GPT_62_SOL_MODEL_ID, "high", "hf_fake")

    assert outcome.effective_effort == "high"
    assert outcome.attempts == 1
    assert len(calls) == 1


async def test_probe_walks_down_after_openai_rejects_an_effort(monkeypatch):
    calls: list[dict] = []

    async def reject_xhigh(**kwargs):
        calls.append(kwargs)
        if kwargs["extra_body"].get("reasoning_effort") == "xhigh":
            raise litellm.BadRequestError(
                message="OpenAIException - Unsupported value: 'reasoning.effort' "
                "does not support 'xhigh' with this model.",
                model="gpt-5.5",
                llm_provider="openai",
            )
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")])

    monkeypatch.setattr(effort_probe, "acompletion", reject_xhigh)

    outcome = await effort_probe.probe_effort(GPT_55_DIRECT_MODEL_ID, "xhigh", "hf")

    assert outcome.effective_effort == "high"
    assert [c["extra_body"]["reasoning_effort"] for c in calls] == ["xhigh", "high"]


async def test_effort_heal_reprobes_and_rebuilds_direct_params(probe_calls):
    # A mid-turn "effort not accepted" 400 re-runs the cascade and rebuilds
    # the kwargs on the direct route with the level that works.
    error = litellm.BadRequestError(
        message="OpenAIException - Unsupported value: 'reasoning.effort' does "
        "not support 'max' with this model.",
        model="gpt-5.5",
        llm_provider="openai",
    )
    session = SimpleNamespace(
        config=SimpleNamespace(
            model_name=GPT_55_DIRECT_MODEL_ID, reasoning_effort="max"
        ),
        hf_token=DUMMY_HF_TOKEN,
        model_effective_effort={},
    )
    session.effective_effort_for = lambda model: session.model_effective_effort.get(
        model, session.config.reasoning_effort
    )
    assert agent_loop._is_effort_config_error(error) is True

    params = await agent_loop._heal_effort_and_rebuild_params(
        session, error, _expected_params(GPT_55_DIRECT_MODEL_ID, "xhigh")
    )

    assert session.model_effective_effort == {GPT_55_DIRECT_MODEL_ID: "xhigh"}
    assert params == _expected_params(GPT_55_DIRECT_MODEL_ID, "xhigh")
    assert [c["extra_body"]["reasoning_effort"] for c in probe_calls] == ["xhigh"]
