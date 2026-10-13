from litellm import Message
from types import SimpleNamespace

from agent.core.model_ids import HF_ROUTER_BASE_URL
from agent.core.prompt_caching import (
    HF_ROUTER_SESSION_ID_HEADER,
    router_session_id_for,
    with_prompt_cache_params,
    with_prompt_caching,
)

BILLING_SESSION_ID = "00000000-0000-4000-8000-000000000001"


def _anthropic_fal_params() -> dict:
    return {
        "model": "openai/anthropic/claude-opus-4.8:fal-ai",
        "api_base": HF_ROUTER_BASE_URL,
    }


def _gpt55_fal_params() -> dict:
    return {
        "model": "openai/openai/gpt-5.5:fal-ai",
        "api_base": HF_ROUTER_BASE_URL,
    }


def _kimi_novita_params() -> dict:
    return {
        "model": "openai/moonshotai/Kimi-K2.7-Code:novita",
        "api_base": HF_ROUTER_BASE_URL,
    }


def _gpt_oss_cerebras_params() -> dict:
    return {
        "model": "openai/openai/gpt-oss-120b:cerebras",
        "api_base": HF_ROUTER_BASE_URL,
    }


def test_prompt_caching_marks_system_prefix_and_tools_for_fal_router_model():
    messages = [
        Message(role="system", content="stable system prompt"),
        Message(role="user", content="current question"),
    ]
    tools = [
        {"type": "function", "function": {"name": "read"}},
        {"type": "function", "function": {"name": "write"}},
    ]

    cached_messages, cached_tools = with_prompt_caching(
        messages, tools, _anthropic_fal_params()
    )

    assert cached_tools is not tools
    assert cached_tools == [
        {"type": "function", "function": {"name": "read"}},
        {
            "type": "function",
            "function": {"name": "write"},
            "cache_control": {"type": "ephemeral"},
        },
    ]
    assert "cache_control" not in tools[-1]
    assert cached_messages is not messages
    assert cached_messages[0] == {
        "role": "system",
        "content": [
            {
                "type": "text",
                "text": "stable system prompt",
                "cache_control": {"type": "ephemeral"},
            }
        ],
    }
    assert messages[0].content == "stable system prompt"


def test_prompt_caching_marks_last_stable_user_before_current_message():
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "stable reference"},
        {"role": "assistant", "content": "previous answer"},
        {"role": "user", "content": "current question"},
    ]

    cached_messages, _ = with_prompt_caching(messages, None, _anthropic_fal_params())

    assert cached_messages[0]["content"] == "stable system"
    assert cached_messages[1]["content"] == [
        {
            "type": "text",
            "text": "stable reference",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert messages[1]["content"] == "stable reference"


def test_prompt_caching_marks_last_text_block_in_content_list():
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "stable part one"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.test/i.png"},
                },
                {"type": "text", "text": "stable part two"},
            ],
        },
        {"role": "user", "content": "current question"},
    ]

    cached_messages, _ = with_prompt_caching(messages, None, _anthropic_fal_params())

    assert cached_messages[0]["content"][0] == {
        "type": "text",
        "text": "stable part one",
    }
    assert cached_messages[0]["content"][1] == {
        "type": "image_url",
        "image_url": {"url": "https://example.test/i.png"},
    }
    assert cached_messages[0]["content"][2] == {
        "type": "text",
        "text": "stable part two",
        "cache_control": {"type": "ephemeral"},
    }
    assert "cache_control" not in messages[0]["content"][2]


def test_prompt_caching_marks_tools_without_message_prefix():
    messages = [{"role": "user", "content": "current question"}]
    tools = [{"type": "function", "function": {"name": "read"}}]

    cached_messages, cached_tools = with_prompt_caching(
        messages, tools, _anthropic_fal_params()
    )

    assert cached_messages is messages
    assert cached_tools == [
        {
            "type": "function",
            "function": {"name": "read"},
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert "cache_control" not in tools[0]


def test_prompt_caching_is_noop_for_non_fal_router_model():
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "current question"},
    ]
    llm_params = {
        "model": "openai/moonshotai/Kimi-K2.7-Code",
        "api_base": HF_ROUTER_BASE_URL,
    }

    cached_messages, cached_tools = with_prompt_caching(messages, None, llm_params)

    assert cached_messages is messages
    assert cached_tools is None


def test_prompt_caching_is_noop_for_gpt55_fal_router_model():
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "current question"},
    ]
    tools = [{"type": "function", "function": {"name": "read"}}]

    cached_messages, cached_tools = with_prompt_caching(
        messages, tools, _gpt55_fal_params()
    )

    assert cached_messages is messages
    assert cached_tools is tools


def test_prompt_caching_is_noop_for_non_router_fal_model():
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "current question"},
    ]
    llm_params = {
        "model": "openai/anthropic/claude-opus-4.8:fal-ai",
        "api_base": "http://localhost:8000/v1",
    }

    cached_messages, _ = with_prompt_caching(messages, None, llm_params)

    assert cached_messages is messages


def test_prompt_cache_params_add_session_id_header_for_fal_router_model():
    llm_params = _anthropic_fal_params()

    cached_params = with_prompt_cache_params(llm_params, session_id=BILLING_SESSION_ID)

    assert cached_params is not llm_params
    assert cached_params["extra_headers"] == {
        HF_ROUTER_SESSION_ID_HEADER: BILLING_SESSION_ID
    }
    assert cached_params["extra_body"] == {"cache_control": {"type": "ephemeral"}}
    assert "extra_headers" not in llm_params
    assert "extra_body" not in llm_params


def test_prompt_cache_params_adds_session_id_header_for_novita_router_model():
    llm_params = _kimi_novita_params()

    cached_params = with_prompt_cache_params(llm_params, session_id=BILLING_SESSION_ID)

    assert cached_params is not llm_params
    assert cached_params["extra_headers"] == {
        HF_ROUTER_SESSION_ID_HEADER: BILLING_SESSION_ID
    }
    assert "extra_body" not in cached_params
    assert "extra_headers" not in llm_params
    assert "extra_body" not in llm_params


def test_prompt_cache_params_adds_session_id_header_for_cerebras_router_model():
    llm_params = _gpt_oss_cerebras_params()

    cached_params = with_prompt_cache_params(llm_params, session_id=BILLING_SESSION_ID)

    assert cached_params is not llm_params
    assert cached_params["extra_headers"] == {
        HF_ROUTER_SESSION_ID_HEADER: BILLING_SESSION_ID
    }
    assert "extra_body" not in cached_params
    assert "extra_headers" not in llm_params
    assert "extra_body" not in llm_params


def test_prompt_cache_params_merges_existing_headers():
    llm_params = {
        **_kimi_novita_params(),
        "extra_headers": {"X-Existing": "1"},
    }

    cached_params = with_prompt_cache_params(llm_params, session_id=BILLING_SESSION_ID)

    assert cached_params["extra_headers"] == {
        "X-Existing": "1",
        HF_ROUTER_SESSION_ID_HEADER: BILLING_SESSION_ID,
    }
    assert llm_params["extra_headers"] == {"X-Existing": "1"}


def test_router_session_id_prefers_billing_window_id():
    assert (
        router_session_id_for(
            SimpleNamespace(
                session_id="session-1",
                inference_billing_session_id=BILLING_SESSION_ID,
            )
        )
        == BILLING_SESSION_ID
    )
    assert router_session_id_for(SimpleNamespace(session_id="session-1")) == "session-1"


def test_prompt_cache_params_adds_anthropic_cache_control_without_session_id():
    cached_params = with_prompt_cache_params(_anthropic_fal_params())

    assert cached_params["extra_body"] == {"cache_control": {"type": "ephemeral"}}


def test_prompt_cache_params_merges_gpt55_cache_hints():
    llm_params = {
        **_gpt55_fal_params(),
        "extra_body": {"reasoning_effort": "high"},
    }

    cached_params = with_prompt_cache_params(llm_params, session_id=BILLING_SESSION_ID)

    assert cached_params["extra_headers"] == {
        HF_ROUTER_SESSION_ID_HEADER: BILLING_SESSION_ID
    }
    assert cached_params["extra_body"] == {
        "reasoning_effort": "high",
        "prompt_cache_key": BILLING_SESSION_ID,
        "prompt_cache_retention": "24h",
    }
    assert llm_params["extra_body"] == {"reasoning_effort": "high"}


def test_prompt_cache_params_adds_gpt55_retention_without_session():
    cached_params = with_prompt_cache_params(_gpt55_fal_params())

    assert cached_params["extra_body"] == {"prompt_cache_retention": "24h"}


def test_prompt_cache_params_is_noop_for_non_router_model():
    llm_params = {
        "model": "openai/openai/gpt-5.5:fal-ai",
        "api_base": "http://localhost:8000/v1",
    }

    cached_params = with_prompt_cache_params(llm_params, session_id="session-1")

    assert cached_params is llm_params


# ── Direct Anthropic API (api.anthropic.com) ──────────────────────────────

_EPHEMERAL = {"type": "ephemeral"}


def _direct_anthropic_params() -> dict:
    return {
        "model": "anthropic/claude-opus-5-5",
        "api_base": "https://api.anthropic.com",
        "api_key": "sk-ant-test",
        "max_tokens": 32000,
        "output_config": {"effort": "high"},
    }


def _tool_call(call_id: str) -> dict:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "bash", "arguments": "{}"},
    }


def _cache_markers(messages: list, tools: list | None) -> int:
    def count(node) -> int:
        if isinstance(node, dict):
            return ("cache_control" in node) + sum(count(v) for v in node.values())
        if isinstance(node, list):
            return sum(count(v) for v in node)
        return 0

    as_dicts = [m if isinstance(m, dict) else m.model_dump() for m in messages]
    return count(as_dicts) + count(tools or [])


def test_direct_anthropic_prompt_cache_params_add_no_hf_hints():
    llm_params = _direct_anthropic_params()

    cached_params = with_prompt_cache_params(llm_params, session_id=BILLING_SESSION_ID)

    assert cached_params == llm_params
    assert "extra_body" not in cached_params
    assert "extra_headers" not in cached_params
    assert "prompt_cache_key" not in cached_params
    assert "cache_control" not in cached_params


def test_direct_anthropic_marks_last_tool_system_and_newest_message():
    messages = [
        Message(role="system", content="stable system prompt"),
        Message(role="user", content="current question"),
    ]
    tools = [
        {"type": "function", "function": {"name": "read"}},
        {"type": "function", "function": {"name": "write"}},
    ]

    cached_messages, cached_tools = with_prompt_caching(
        messages, tools, _direct_anthropic_params()
    )

    assert cached_tools[-1]["cache_control"] == _EPHEMERAL
    assert "cache_control" not in cached_tools[0]
    assert "cache_control" not in tools[-1]
    assert cached_messages[0]["content"] == [
        {"type": "text", "text": "stable system prompt", "cache_control": _EPHEMERAL}
    ]
    assert cached_messages[1]["content"] == [
        {"type": "text", "text": "current question", "cache_control": _EPHEMERAL}
    ]
    assert messages[0].content == "stable system prompt"
    assert messages[1].content == "current question"


def test_direct_anthropic_tool_loop_marks_newest_and_previous_tool_results():
    assistant_1 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call("toolu_1")],
        "thinking_blocks": [{"type": "thinking", "thinking": "", "signature": "S1"}],
    }
    assistant_2 = {
        "role": "assistant",
        "content": None,
        "tool_calls": [_tool_call("toolu_2")],
        "thinking_blocks": [{"type": "thinking", "thinking": "", "signature": "S2"}],
    }
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "do the task"},
        assistant_1,
        {"role": "tool", "tool_call_id": "toolu_1", "content": "first output"},
        assistant_2,
        {"role": "tool", "tool_call_id": "toolu_2", "content": "second output"},
    ]

    cached_messages, _ = with_prompt_caching(messages, None, _direct_anthropic_params())

    assert cached_messages[1] is messages[1]
    assert cached_messages[2] is assistant_1
    assert cached_messages[4] is assistant_2
    # Tool messages carry the marker at message level; LiteLLM puts it on the
    # Anthropic tool_result block and the content stays a plain string.
    assert cached_messages[3] == {
        "role": "tool",
        "tool_call_id": "toolu_1",
        "content": "first output",
        "cache_control": _EPHEMERAL,
    }
    assert cached_messages[5] == {
        "role": "tool",
        "tool_call_id": "toolu_2",
        "content": "second output",
        "cache_control": _EPHEMERAL,
    }
    assert "cache_control" not in messages[3]
    assert "cache_control" not in messages[5]


def test_direct_anthropic_new_user_turn_also_marks_last_message_of_prior_request():
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "first question"},
        {"role": "assistant", "content": "first answer"},
        {"role": "user", "content": "follow-up question"},
    ]

    cached_messages, _ = with_prompt_caching(messages, None, _direct_anthropic_params())

    assert cached_messages[1]["content"][-1]["cache_control"] == _EPHEMERAL
    assert cached_messages[2] is messages[2]
    assert cached_messages[3]["content"][-1]["cache_control"] == _EPHEMERAL


def test_direct_anthropic_skips_empty_tool_results():
    messages = [
        {"role": "user", "content": "do the task"},
        {"role": "assistant", "content": None, "tool_calls": [_tool_call("t1")]},
        {"role": "tool", "tool_call_id": "t1", "content": ""},
    ]

    cached_messages, _ = with_prompt_caching(messages, None, _direct_anthropic_params())

    assert cached_messages[2] is messages[2]
    assert cached_messages[0]["content"][-1]["cache_control"] == _EPHEMERAL


def test_direct_anthropic_never_exceeds_four_cache_breakpoints():
    messages = [{"role": "system", "content": "stable system"}]
    for turn in range(12):
        messages.append({"role": "user", "content": f"question {turn}"})
        messages.append(
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [_tool_call(f"t{turn}")],
            }
        )
        messages.append(
            {"role": "tool", "tool_call_id": f"t{turn}", "content": f"out {turn}"}
        )
    tools = [{"type": "function", "function": {"name": "bash"}}]

    for history_len in range(1, len(messages) + 1):
        history = messages[:history_len]
        cached_messages, cached_tools = with_prompt_caching(
            history, tools, _direct_anthropic_params()
        )
        assert _cache_markers(cached_messages, cached_tools) <= 4
    assert _cache_markers(messages, tools) == 0


def test_direct_anthropic_ignores_router_claude_shape():
    # The HF Router Claude keeps its own marker placement (newest message
    # stays dynamic) even though both ids start with "anthropic/".
    messages = [
        {"role": "system", "content": "stable system"},
        {"role": "user", "content": "current question"},
    ]

    cached_messages, _ = with_prompt_caching(messages, None, _anthropic_fal_params())

    assert cached_messages[1] is messages[1]
