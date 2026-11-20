"""Non-streaming replies that LiteLLM's Responses API bridge splits apart.

LiteLLM 1.83 turns a direct OpenAI reply with text and function calls into
two choices: the text first, every tool call in a last extra choice. These
tests run the real bridge on canned ``/v1/responses`` JSON (``httpx`` is
patched and the key is a dummy) and check every non-streaming reader: the
CLI agent loop, the research sub-agent and the compaction summary.
"""

import json
from types import SimpleNamespace

import httpx
import litellm
import pytest
from litellm import Message

from agent.context_manager.manager import summarize_messages
from agent.core.agent_loop import _call_llm_non_streaming
from agent.core.llm_params import _resolve_llm_params
from agent.core.llm_response import reply_finish_reason, reply_message
from agent.core.model_ids import (
    GPT_55_DIRECT_MODEL_ID,
    GPT_62_SOL_MODEL_ID,
    OPENAI_API_BASE_URL,
)
from agent.tools.research_tool import research_handler

DUMMY_KEY = "sk-openai-test"
RESPONSES_URL = f"{OPENAI_API_BASE_URL}/responses"
PREAMBLE = "I'll start by finding anchor papers."
_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "hf_papers",
            "description": "Search papers",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]
_USAGE = {
    "input_tokens": 1000,
    "input_tokens_details": {"cached_tokens": 0},
    "output_tokens": 200,
    "output_tokens_details": {"reasoning_tokens": 50},
    "total_tokens": 1200,
}

# LiteLLM's background logging worker drops a pending success-callback
# coroutine when pytest-asyncio starts a new event loop per test.
pytestmark = pytest.mark.filterwarnings(
    "ignore:coroutine 'Logging.async_success_handler' was never awaited:RuntimeWarning"
)


@pytest.fixture(autouse=True)
def provider_env(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", DUMMY_KEY)
    for name in (
        "OPENAI_BASE_URL",
        "OPENAI_API_BASE",
        "OPENAI_ORGANIZATION",
        "OPENAI_ORG_ID",
        "OPENAI_PROJECT_ID",
    ):
        monkeypatch.delenv(name, raising=False)


def _message_item(item_id: str, text: str) -> dict:
    return {
        "type": "message",
        "id": item_id,
        "role": "assistant",
        "status": "completed",
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _function_call_item(call_id: str, query: str) -> dict:
    return {
        "type": "function_call",
        "id": f"fc_{call_id}",
        "call_id": call_id,
        "name": "hf_papers",
        "arguments": json.dumps({"query": query}),
        "status": "completed",
    }


_PREAMBLE_AND_TWO_CALLS = [
    {"type": "reasoning", "id": "rs_1", "summary": []},
    _message_item("msg_1", PREAMBLE),
    _function_call_item("call_1", "GRPO"),
    _function_call_item("call_2", "DPO"),
]
_TWO_TEXT_ITEMS = [
    _message_item("msg_1", "Here is what I found:"),
    _message_item("msg_2", "FULL FINDINGS"),
]


def _response_body(output: list[dict]) -> dict:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1_700_000_000,
        "status": "completed",
        "model": "gpt-6.2-sol",
        "output": output,
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "text": {"format": {"type": "text"}},
        "error": None,
        "incomplete_details": None,
        "usage": _USAGE,
    }


class FakeResponsesApi:
    """Answers POST /v1/responses with queued output items, in order."""

    def __init__(self):
        self.requests: list[dict] = []
        self._outputs: list[list[dict]] = []

    def queue(self, output: list[dict]) -> None:
        self._outputs.append(output)

    async def send(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        self.requests.append({"url": url, "body": json.loads(request.content)})
        if url != RESPONSES_URL or not self._outputs:
            raise AssertionError(f"Unexpected request to {url}")
        body = _response_body(self._outputs.pop(0))
        return httpx.Response(200, json=body, request=request)


@pytest.fixture
def responses_api(monkeypatch):
    fake = FakeResponsesApi()

    async def send(_client, request, *_args, **_kwargs):
        return await fake.send(request)

    monkeypatch.setattr(httpx.AsyncClient, "send", send)
    return fake


async def _bridged_response(model_id: str = GPT_62_SOL_MODEL_ID):
    params = _resolve_llm_params(model_id, None, reasoning_effort="high")
    return await litellm.acompletion(
        messages=[{"role": "user", "content": "find recipes"}],
        tools=_TOOLS,
        tool_choice="auto",
        stream=False,
        timeout=30,
        **params,
    )


# ── reply_message / reply_finish_reason ───────────────────────────────────


@pytest.mark.parametrize("model_id", [GPT_62_SOL_MODEL_ID, GPT_55_DIRECT_MODEL_ID])
async def test_split_reply_is_merged_into_one_message(responses_api, model_id):
    responses_api.queue(_PREAMBLE_AND_TWO_CALLS)

    response = await _bridged_response(model_id)
    message = reply_message(response)

    # The bridge contract this helper exists for: text and calls are split.
    assert len(response.choices) == 2
    assert message.content == PREAMBLE
    assert [(c.id, c.function.name) for c in message.tool_calls] == [
        ("call_1", "hf_papers"),
        ("call_2", "hf_papers"),
    ]
    assert [json.loads(c.function.arguments) for c in message.tool_calls] == [
        {"query": "GRPO"},
        {"query": "DPO"},
    ]
    assert reply_finish_reason(response) == "tool_calls"


async def test_split_text_parts_are_joined_in_order(responses_api):
    responses_api.queue(_TWO_TEXT_ITEMS)

    response = await _bridged_response()

    assert reply_message(response).content == "Here is what I found:\n\nFULL FINDINGS"
    assert reply_message(response).tool_calls is None
    assert reply_finish_reason(response) == "stop"


async def test_merging_leaves_the_response_untouched(responses_api):
    responses_api.queue(_PREAMBLE_AND_TWO_CALLS)
    response = await _bridged_response()

    reply_message(response)

    assert len(response.choices) == 2
    assert response.choices[0].message.content == PREAMBLE
    assert response.choices[0].message.tool_calls is None
    assert len(response.choices[1].message.tool_calls) == 2


def test_single_choice_reply_is_returned_as_is():
    message = Message(role="assistant", content="partial", tool_calls=None)
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=message, finish_reason="length")]
    )

    assert reply_message(response) is message
    # A truncated single-choice reply keeps "length" (the truncation guard
    # in the agent loop depends on it), tool calls or not.
    assert reply_finish_reason(response) == "length"


def test_empty_choices_still_raise_like_choices_index():
    response = SimpleNamespace(choices=[])

    with pytest.raises(IndexError):
        reply_message(response)
    with pytest.raises(IndexError):
        reply_finish_reason(response)


# ── every non-streaming reader keeps the whole reply ────────────────────


def _loop_session() -> SimpleNamespace:
    async def send_event(_event):
        return None

    return SimpleNamespace(
        config=SimpleNamespace(model_name=GPT_62_SOL_MODEL_ID, reasoning_effort="high"),
        session_id="session-1",
        is_cancelled=False,
        send_event=send_event,
        model_effective_effort={},
        hf_token=None,
    )


async def test_non_streaming_agent_turn_keeps_every_tool_call(responses_api):
    responses_api.queue(_PREAMBLE_AND_TWO_CALLS)
    params = _resolve_llm_params(GPT_62_SOL_MODEL_ID, None, reasoning_effort="high")

    result = await _call_llm_non_streaming(
        _loop_session(),
        [Message(role="system", content="sys"), Message(role="user", content="go")],
        _TOOLS,
        params,
    )

    assert result.content == PREAMBLE
    assert result.finish_reason == "tool_calls"
    assert [tc["id"] for tc in result.tool_calls_acc.values()] == ["call_1", "call_2"]
    assert [tc["function"]["name"] for tc in result.tool_calls_acc.values()] == [
        "hf_papers",
        "hf_papers",
    ]


class _PapersRouter:
    def __init__(self):
        self.calls: list[tuple[str, dict]] = []

    def get_tool_specs_for_llm(self):
        return _TOOLS

    async def call_tool(self, name, args, session=None, tool_call_id=None):
        self.calls.append((name, args))
        return f"papers about {args['query']}", True


async def test_research_sub_agent_runs_the_calls_that_follow_a_preamble(
    responses_api,
):
    responses_api.queue(_PREAMBLE_AND_TWO_CALLS)
    responses_api.queue([_message_item("msg_2", "FINAL SUMMARY")])
    session = _loop_session()
    session.tool_router = _PapersRouter()

    output, ok = await research_handler(
        {"task": "find GRPO recipes"}, session=session, tool_call_id="tc-1"
    )

    assert (output, ok) == ("FINAL SUMMARY", True)
    assert session.tool_router.calls == [
        ("hf_papers", {"query": "GRPO"}),
        ("hf_papers", {"query": "DPO"}),
    ]
    second_input = responses_api.requests[1]["body"]["input"]
    outputs = [
        i["call_id"] for i in second_input if i["type"] == "function_call_output"
    ]
    assert outputs == ["call_1", "call_2"]


async def test_compaction_summary_keeps_every_text_part(responses_api):
    responses_api.queue(_TWO_TEXT_ITEMS)

    summary, completion_tokens = await summarize_messages(
        [
            Message(role="user", content="train"),
            Message(role="assistant", content="ok"),
        ],
        model_name=GPT_62_SOL_MODEL_ID,
        max_tokens=2000,
    )

    assert summary == "Here is what I found:\n\nFULL FINDINGS"
    assert completion_tokens == 200
