"""Backend wiring for the direct OpenAI model routes (OPENAI_API_KEY).

Covers the picker's OpenAI block, which direct entries ``/api/config/model``
shows for each set of keys, per-request API-key gating, lazy-restore model
validation, direct-provider spend attribution and the LLM health probe for an
OpenAI default. No test here makes a network call: every key is a dummy value
and LiteLLM's ``acompletion`` is replaced.
"""

import itertools
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import litellm
import pytest
from fastapi import HTTPException

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from agent.core import web_models  # noqa: E402
from agent.core.model_ids import (  # noqa: E402
    CLAUDE_FABLE_52_MODEL_ID,
    CLAUDE_HAIKU_55_MODEL_ID,
    CLAUDE_OPUS_48_MODEL_ID,
    CLAUDE_OPUS_55_MODEL_ID,
    CLAUDE_SONNET_55_MODEL_ID,
    DIRECT_OPENAI_MODEL_IDS,
    GLM_52_MODEL_ID,
    GPT_55_DIRECT_MODEL_ID,
    GPT_55_MODEL_ID,
    GPT_61_TERRA_MODEL_ID,
    GPT_62_ASTRA_MODEL_ID,
    GPT_62_SOL_MODEL_ID,
    GPT_6_LUNA_MODEL_ID,
    KIMI_K27_CODE_MODEL_ID,
    MINIMAX_M3_MODEL_ID,
)
from routes import agent  # noqa: E402
from session_manager import SessionManager  # noqa: E402
from usage import aggregate_usage_events  # noqa: E402

DEFAULT_MODEL_ENV = "ML_INTERN_DEFAULT_MODEL_ID"
ANTHROPIC_KEY_ENV = "ANTHROPIC_API_KEY"
OPENAI_KEY_ENV = "OPENAI_API_KEY"
DEEPSEEK_KEY_ENV = "DEEPSEEK_API_KEY"
PROVIDER_KEY_ENVS = (ANTHROPIC_KEY_ENV, OPENAI_KEY_ENV, DEEPSEEK_KEY_ENV)
DUMMY_KEYS = {
    ANTHROPIC_KEY_ENV: "sk-ant-test",
    OPENAI_KEY_ENV: "sk-openai-test",
    DEEPSEEK_KEY_ENV: "sk-deepseek-test",
}
DEEPSEEK_PRO_ID = "deepseek/deepseek-v4-pro"
DEEPSEEK_FLASH_ID = "deepseek/deepseek-v4-flash"
OPENAI_API_BASE = "https://api.openai.com/v1"
MISSING_KEY_VALUES = [None, "", "   "]

# Display order, as /api/config/model lists them.
HF_ROUTER_IDS = [
    CLAUDE_OPUS_48_MODEL_ID,
    GPT_55_MODEL_ID,
    KIMI_K27_CODE_MODEL_ID,
    MINIMAX_M3_MODEL_ID,
    GLM_52_MODEL_ID,
]
DIRECT_IDS_BY_KEY_ENV = {
    ANTHROPIC_KEY_ENV: [
        CLAUDE_OPUS_55_MODEL_ID,
        CLAUDE_FABLE_52_MODEL_ID,
        CLAUDE_SONNET_55_MODEL_ID,
        CLAUDE_HAIKU_55_MODEL_ID,
    ],
    OPENAI_KEY_ENV: [
        GPT_55_DIRECT_MODEL_ID,
        GPT_62_ASTRA_MODEL_ID,
        GPT_62_SOL_MODEL_ID,
        GPT_61_TERRA_MODEL_ID,
        GPT_6_LUNA_MODEL_ID,
    ],
    DEEPSEEK_KEY_ENV: [DEEPSEEK_PRO_ID, DEEPSEEK_FLASH_ID],
}
OPENAI_IDS = DIRECT_IDS_BY_KEY_ENV[OPENAI_KEY_ENV]
# The router GPT-5.5 followed by the direct OpenAI block.
OPENAI_PICKER_BLOCK = [
    {"id": "openai/gpt-5.5:fal-ai", "label": "GPT-5.5"},
    {"id": "openai/gpt-5.5", "label": "GPT-5.5 (OpenAI key)"},
    {"id": "openai/gpt-6.2-astra", "label": "GPT-6.2 Astra"},
    {"id": "openai/gpt-6.2-sol", "label": "GPT-6.2 Sol"},
    {"id": "openai/gpt-6.1-terra", "label": "GPT-6.1 Terra"},
    {"id": "openai/gpt-6-luna", "label": "GPT-6 Luna"},
]
KEY_ENV_SUBSETS = [
    subset
    for size in range(len(PROVIDER_KEY_ENVS) + 1)
    for subset in itertools.combinations(PROVIDER_KEY_ENVS, size)
]
GPT_62_SOL_MISSING_KEY_DETAIL = (
    "GPT-6.2 Sol needs OPENAI_API_KEY on the server. Add it to the repo .env "
    "and restart the backend."
)
PROBE_MESSAGES = [{"role": "user", "content": "hi"}]
RESTORE_MESSAGES = [{"role": "user", "content": "resume this"}]


class _Request:
    """Minimal stand-in for a FastAPI Request."""

    def __init__(self, headers: dict | None = None) -> None:
        self.headers = headers or {}
        self.cookies: dict = {}


class _JsonRequest(_Request):
    """A Request stand-in that also carries a JSON body."""

    def __init__(self, body: dict | None = None) -> None:
        super().__init__()
        self._body = body or {}

    async def json(self) -> dict:
        return self._body


@pytest.fixture
def no_provider_keys(monkeypatch):
    """No direct-provider key and no default override, whatever .env holds."""
    for env_name in PROVIDER_KEY_ENVS:
        monkeypatch.delenv(env_name, raising=False)
    monkeypatch.delenv(DEFAULT_MODEL_ENV, raising=False)


@pytest.fixture
def openai_key(monkeypatch, no_provider_keys):
    _set_keys(monkeypatch, [OPENAI_KEY_ENV])


@pytest.fixture
def other_provider_keys(monkeypatch, no_provider_keys):
    """Every key but OpenAI's is set, so a leaked key value would show."""
    _set_keys(monkeypatch, [ANTHROPIC_KEY_ENV, DEEPSEEK_KEY_ENV])


@pytest.fixture
def openai_default(monkeypatch, openai_key):
    monkeypatch.setenv(DEFAULT_MODEL_ENV, GPT_62_SOL_MODEL_ID)


@pytest.fixture
def no_hf_token(monkeypatch):
    monkeypatch.delenv("HF_TOKEN", raising=False)


@pytest.fixture
def probe_calls(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    async def fake_acompletion(**kwargs):
        calls.append(kwargs)

    monkeypatch.setattr(agent, "acompletion", fake_acompletion)
    return calls


def _set_keys(monkeypatch, env_names) -> None:
    for env_name in env_names:
        monkeypatch.setenv(env_name, DUMMY_KEYS[env_name])


def _set_openai_key(monkeypatch, value: str | None) -> None:
    if value is None:
        monkeypatch.delenv(OPENAI_KEY_ENV, raising=False)
    else:
        monkeypatch.setenv(OPENAI_KEY_ENV, value)


def _ids(models: list[dict]) -> list[str]:
    return [model["id"] for model in models]


def _expected_visible_ids(key_envs) -> list[str]:
    """Picker ids, in display order, that the given set keys make visible."""
    shown = set(HF_ROUTER_IDS).union(
        *(DIRECT_IDS_BY_KEY_ENV[env_name] for env_name in key_envs)
    )
    return [
        model_id for model_id in _ids(web_models.AVAILABLE_MODELS) if model_id in shown
    ]


def _assert_no_key_value(text: str) -> None:
    assert "sk-" not in text
    assert all(value not in text for value in DUMMY_KEYS.values())


def _install_session_access(monkeypatch) -> None:
    async def fake_check_session_access(session_id, user, request=None, **_):
        return SimpleNamespace(user_id=user["user_id"])

    monkeypatch.setattr(agent, "_check_session_access", fake_check_session_access)


def _install_model_updates(monkeypatch) -> list[tuple[str, str]]:
    updated: list[tuple[str, str]] = []

    async def fake_update_session_model(session_id, model_id):
        updated.append((session_id, model_id))

    monkeypatch.setattr(
        agent.session_manager, "update_session_model", fake_update_session_model
    )
    return updated


def _install_session_creation(monkeypatch) -> list[str]:
    created: list[str] = []

    async def fake_create_session(**kwargs):
        created.append(kwargs["model"])
        return "s1"

    async def fake_reset_usage_window(session_id):
        return None

    monkeypatch.setattr(agent.session_manager, "create_session", fake_create_session)
    monkeypatch.setattr(agent, "_reset_usage_window", fake_reset_usage_window)
    return created


def _install_summary_seeding(monkeypatch) -> list[str]:
    seeded: list[str] = []

    async def fake_seed_from_summary(session_id, messages):
        seeded.append(session_id)
        return len(messages)

    monkeypatch.setattr(
        agent.session_manager, "seed_from_summary", fake_seed_from_summary
    )
    return seeded


def _openai_quota_error(model: str) -> Exception:
    """What LiteLLM raises when an OpenAI-style API answers insufficient_quota."""
    return litellm.RateLimitError(
        message=(
            'OpenAIException - {"error":{"message":"You exceeded your current '
            'quota, please check your plan and billing details.","type":'
            '"insufficient_quota","param":null,"code":"insufficient_quota"}}'
        ),
        model=model,
        llm_provider="openai",
    )


def _output_cap_error() -> Exception:
    """What LiteLLM's Responses API bridge raises for a 200 "incomplete" reply
    whose output cap ran out during reasoning, before any text."""
    return litellm.APIConnectionError(
        message=(
            "OpenAIException - gpt-6.2-sol unable to complete request: "
            "max_output_tokens"
        ),
        model="gpt-6.2-sol",
        llm_provider="openai",
    )


def _capped_responses_reply() -> dict:
    """OpenAI's 200 reply when the probe's cap is spent on reasoning."""
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1700000000,
        "status": "incomplete",
        "incomplete_details": {"reason": "max_output_tokens"},
        "model": "gpt-6.2-sol",
        "output": [
            {"type": "reasoning", "id": "rs_1", "summary": [], "encrypted_content": "E"}
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "text": {"format": {"type": "text"}},
        "usage": {
            "input_tokens": 8,
            "input_tokens_details": {"cached_tokens": 0},
            "output_tokens": 64,
            "output_tokens_details": {"reasoning_tokens": 64},
            "total_tokens": 72,
        },
    }


@pytest.fixture
def capped_openai_http(monkeypatch) -> list[httpx.Request]:
    """Answer every LiteLLM HTTP request with OpenAI's capped 200 reply."""
    sent: list[httpx.Request] = []

    async def fake_send(self, request, *args, **kwargs):
        sent.append(request)
        return httpx.Response(200, json=_capped_responses_reply(), request=request)

    monkeypatch.setattr(httpx.AsyncClient, "send", fake_send)
    return sent


def _llm_call(model: str, cost_usd: float) -> dict:
    return {"event_type": "llm_call", "data": {"model": model, "cost_usd": cost_usd}}


# ── Picker ──────────────────────────────────────────────────────────────


def test_picker_lists_direct_openai_block_right_after_router_gpt_55():
    # Arrange
    models = web_models.AVAILABLE_MODELS
    start = _ids(models).index(GPT_55_MODEL_ID)

    # Act
    block = models[start : start + len(OPENAI_PICKER_BLOCK)]

    # Assert
    assert block == OPENAI_PICKER_BLOCK


def test_picker_offers_both_gpt_55_routes_under_distinct_ids():
    # Act
    gpt_55_entries = [
        model
        for model in web_models.AVAILABLE_MODELS
        if model["label"].startswith("GPT-5.5")
    ]

    # Assert
    assert _ids(gpt_55_entries) == [GPT_55_MODEL_ID, GPT_55_DIRECT_MODEL_ID]
    assert len(web_models.PICKER_MODEL_IDS) == len(web_models.AVAILABLE_MODELS)
    assert agent._model_label(GPT_55_MODEL_ID) == "GPT-5.5"
    assert agent._model_label(GPT_55_DIRECT_MODEL_ID) == "GPT-5.5 (OpenAI key)"


def test_picker_openai_entries_are_exactly_the_core_allowlist():
    picker_openai_ids = [
        model_id
        for model_id in _ids(web_models.AVAILABLE_MODELS)
        if model_id in DIRECT_OPENAI_MODEL_IDS
    ]

    assert picker_openai_ids == OPENAI_IDS
    assert set(OPENAI_IDS) == DIRECT_OPENAI_MODEL_IDS


# ── Key-based visibility ────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("key_envs", "expected_count"),
    [
        ((), 5),
        ((ANTHROPIC_KEY_ENV,), 9),
        ((OPENAI_KEY_ENV,), 10),
        ((DEEPSEEK_KEY_ENV,), 7),
        (PROVIDER_KEY_ENVS, 16),
    ],
)
async def test_visible_models_follow_the_configured_keys(
    monkeypatch, no_provider_keys, key_envs, expected_count
):
    # Arrange
    _set_keys(monkeypatch, key_envs)
    expected_ids = _expected_visible_ids(key_envs)

    # Act
    visible = web_models.visible_picker_models()
    response = await agent.get_model()

    # Assert
    assert len(visible) == expected_count
    assert _ids(visible) == expected_ids
    assert visible == [
        model for model in web_models.AVAILABLE_MODELS if model["id"] in expected_ids
    ]
    assert response["available"] == visible


def test_without_keys_only_hf_router_models_are_visible(no_provider_keys):
    assert _ids(web_models.visible_picker_models()) == HF_ROUTER_IDS


@pytest.mark.parametrize("blank_value", ["", "   ", "\t\n"])
def test_blank_openai_key_counts_as_missing(monkeypatch, no_provider_keys, blank_value):
    monkeypatch.setenv(OPENAI_KEY_ENV, blank_value)

    assert _ids(web_models.visible_picker_models()) == HF_ROUTER_IDS


def test_openai_key_set_after_import_shows_on_the_next_call(
    monkeypatch, no_provider_keys
):
    # Act
    before = _ids(web_models.visible_picker_models())
    monkeypatch.setenv(OPENAI_KEY_ENV, DUMMY_KEYS[OPENAI_KEY_ENV])
    with_key = _ids(web_models.visible_picker_models())
    monkeypatch.delenv(OPENAI_KEY_ENV)
    after_removal = _ids(web_models.visible_picker_models())

    # Assert
    assert before == HF_ROUTER_IDS
    assert with_key == _expected_visible_ids([OPENAI_KEY_ENV])
    assert after_removal == HF_ROUTER_IDS


def test_visible_picker_models_returns_fresh_copies(monkeypatch, no_provider_keys):
    # Arrange
    _set_keys(monkeypatch, PROVIDER_KEY_ENVS)
    pristine = [dict(model) for model in web_models.AVAILABLE_MODELS]

    # Act
    first = web_models.visible_picker_models()
    first[0]["label"] = "changed"
    first.append({"id": "extra", "label": "Extra"})
    second = web_models.visible_picker_models()

    # Assert
    assert second == pristine
    assert web_models.AVAILABLE_MODELS == pristine
    assert second is not web_models.AVAILABLE_MODELS
    assert all(
        copy is not original
        for copy, original in zip(second, web_models.AVAILABLE_MODELS)
    )


# ── /api/config/model ───────────────────────────────────────────────────


@pytest.mark.parametrize(
    "configured_default",
    [
        None,
        GPT_62_SOL_MODEL_ID,
        GPT_55_DIRECT_MODEL_ID,
        CLAUDE_HAIKU_55_MODEL_ID,
        DEEPSEEK_FLASH_ID,
    ],
)
@pytest.mark.parametrize("key_envs", KEY_ENV_SUBSETS)
async def test_config_model_current_is_always_an_available_model(
    monkeypatch, no_provider_keys, key_envs, configured_default
):
    # Arrange
    _set_keys(monkeypatch, key_envs)
    if configured_default is not None:
        monkeypatch.setenv(DEFAULT_MODEL_ENV, configured_default)

    # Act
    response = await agent.get_model()

    # Assert
    assert response["current"] in _ids(response["available"])
    assert response["available"] == web_models.visible_picker_models()


async def test_config_model_uses_openai_default_when_its_key_is_set(openai_default):
    response = await agent.get_model()

    assert response["current"] == GPT_62_SOL_MODEL_ID
    assert _ids(response["available"]) == _expected_visible_ids([OPENAI_KEY_ENV])


async def test_config_model_openai_default_without_its_key_falls_back_to_glm(
    monkeypatch, other_provider_keys
):
    # Arrange
    monkeypatch.setenv(DEFAULT_MODEL_ENV, GPT_62_SOL_MODEL_ID)

    # Act
    response = await agent.get_model()

    # Assert
    assert response["current"] == GLM_52_MODEL_ID
    assert _ids(response["available"]) == _expected_visible_ids(
        [ANTHROPIC_KEY_ENV, DEEPSEEK_KEY_ENV]
    )
    assert len(response["available"]) == 11


async def test_config_model_available_is_a_copy_of_the_picker(
    monkeypatch, no_provider_keys
):
    # Arrange
    _set_keys(monkeypatch, PROVIDER_KEY_ENVS)

    # Act
    response = await agent.get_model()

    # Assert
    assert response["available"] == web_models.AVAILABLE_MODELS
    assert response["available"] is not web_models.AVAILABLE_MODELS


# ── Routing and validation ──────────────────────────────────────────────


@pytest.mark.parametrize("model_id", OPENAI_IDS)
def test_direct_openai_ids_do_not_require_hf_router_token(model_id):
    assert agent._model_requires_hf_router_token(model_id) is False


@pytest.mark.parametrize(
    "model_id", [GPT_55_MODEL_ID, f"huggingface/{GPT_55_MODEL_ID}"]
)
def test_router_gpt_55_still_requires_hf_router_token(model_id):
    assert agent._model_requires_hf_router_token(model_id) is True


@pytest.mark.parametrize("model_id", OPENAI_IDS)
def test_validate_model_id_accepts_openai_ids_without_a_key(no_provider_keys, model_id):
    assert agent._validate_model_id(model_id) is None


@pytest.mark.parametrize(
    "model_id",
    [
        "openai/gpt-4o",
        "openai/gpt-5.4",
        "openai/responses/gpt-6.2-sol",
        "gpt-6.2-sol",
    ],
)
def test_validate_model_id_rejects_openai_ids_outside_the_picker(model_id):
    with pytest.raises(HTTPException) as exc_info:
        agent._validate_model_id(model_id)

    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == f"Unknown model: {model_id}"


@pytest.mark.parametrize("model_id", OPENAI_IDS)
def test_saved_openai_model_survives_lazy_restore_without_its_key(
    no_provider_keys, model_id
):
    assert SessionManager._model_from_saved_metadata(model_id) == model_id


# ── Model switch ────────────────────────────────────────────────────────


@pytest.mark.parametrize("missing_value", MISSING_KEY_VALUES)
async def test_switch_to_openai_model_without_key_is_rejected(
    monkeypatch, other_provider_keys, missing_value
):
    # Arrange
    _set_openai_key(monkeypatch, missing_value)
    _install_session_access(monkeypatch)
    updated = _install_model_updates(monkeypatch)

    # Act
    with pytest.raises(HTTPException) as exc_info:
        await agent.set_session_model(
            "s1",
            {"model": GPT_62_SOL_MODEL_ID},
            request=None,
            user={"user_id": "u1"},
        )

    # Assert
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == GPT_62_SOL_MISSING_KEY_DETAIL
    _assert_no_key_value(exc_info.value.detail)
    assert updated == []


@pytest.mark.parametrize("model_id", OPENAI_IDS)
async def test_switch_to_openai_model_with_key_is_allowed(
    monkeypatch, openai_key, model_id
):
    # Arrange
    _install_session_access(monkeypatch)
    updated = _install_model_updates(monkeypatch)

    # Act
    response = await agent.set_session_model(
        "s1",
        {"model": model_id},
        request=None,
        user={"user_id": "u1"},
    )

    # Assert
    assert response == {"session_id": "s1", "model": model_id}
    assert updated == [("s1", model_id)]


async def test_switch_to_router_gpt_55_does_not_need_openai_key(
    monkeypatch, no_provider_keys
):
    # Arrange
    _install_session_access(monkeypatch)
    updated = _install_model_updates(monkeypatch)

    # Act
    response = await agent.set_session_model(
        "s1",
        {"model": GPT_55_MODEL_ID},
        request=None,
        user={"user_id": "u1"},
    )

    # Assert
    assert response["model"] == GPT_55_MODEL_ID
    assert updated == [("s1", GPT_55_MODEL_ID)]


async def test_missing_openai_key_detail_uses_the_direct_gpt_55_label(
    monkeypatch, other_provider_keys
):
    # Arrange
    _install_session_access(monkeypatch)
    _install_model_updates(monkeypatch)

    # Act
    with pytest.raises(HTTPException) as exc_info:
        await agent.set_session_model(
            "s1",
            {"model": GPT_55_DIRECT_MODEL_ID},
            request=None,
            user={"user_id": "u1"},
        )

    # Assert
    assert exc_info.value.detail == (
        "GPT-5.5 (OpenAI key) needs OPENAI_API_KEY on the server. Add it to the "
        "repo .env and restart the backend."
    )


# ── Session create / restore-summary ────────────────────────────────────


@pytest.mark.parametrize("missing_value", MISSING_KEY_VALUES)
async def test_create_session_with_openai_model_without_key_is_rejected(
    monkeypatch, other_provider_keys, missing_value
):
    # Arrange
    _set_openai_key(monkeypatch, missing_value)
    created = _install_session_creation(monkeypatch)

    # Act
    with pytest.raises(HTTPException) as exc_info:
        await agent.create_session(
            _JsonRequest({"model": GPT_62_SOL_MODEL_ID}), {"user_id": "u1"}
        )

    # Assert
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == GPT_62_SOL_MISSING_KEY_DETAIL
    _assert_no_key_value(exc_info.value.detail)
    assert created == []


async def test_create_session_with_openai_model_and_key_is_allowed(
    monkeypatch, openai_key
):
    # Arrange
    created = _install_session_creation(monkeypatch)

    # Act
    response = await agent.create_session(
        _JsonRequest({"model": GPT_62_SOL_MODEL_ID}), {"user_id": "u1"}
    )

    # Assert
    assert response.model == GPT_62_SOL_MODEL_ID
    assert created == [GPT_62_SOL_MODEL_ID]


async def test_create_session_without_model_uses_openai_default_with_key(
    monkeypatch, openai_default
):
    # Arrange
    created = _install_session_creation(monkeypatch)

    # Act
    response = await agent.create_session(_JsonRequest(), {"user_id": "u1"})

    # Assert
    assert response.model == GPT_62_SOL_MODEL_ID
    assert created == [GPT_62_SOL_MODEL_ID]


async def test_create_session_without_model_falls_back_to_glm_without_openai_key(
    monkeypatch, no_provider_keys
):
    # Arrange: a missing key must not turn the configured default into a 400.
    monkeypatch.setenv(DEFAULT_MODEL_ENV, GPT_62_SOL_MODEL_ID)
    created = _install_session_creation(monkeypatch)

    # Act
    response = await agent.create_session(_JsonRequest(), {"user_id": "u1"})

    # Assert
    assert response.model == GLM_52_MODEL_ID
    assert created == [GLM_52_MODEL_ID]


@pytest.mark.parametrize("missing_value", MISSING_KEY_VALUES)
async def test_restore_summary_with_openai_model_without_key_is_rejected(
    monkeypatch, other_provider_keys, missing_value
):
    # Arrange
    _set_openai_key(monkeypatch, missing_value)
    created = _install_session_creation(monkeypatch)
    seeded = _install_summary_seeding(monkeypatch)

    # Act
    with pytest.raises(HTTPException) as exc_info:
        await agent.restore_session_summary(
            _JsonRequest(),
            {"messages": RESTORE_MESSAGES, "model": GPT_62_SOL_MODEL_ID},
            {"user_id": "u1"},
        )

    # Assert
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == GPT_62_SOL_MISSING_KEY_DETAIL
    _assert_no_key_value(exc_info.value.detail)
    assert created == []
    assert seeded == []


async def test_restore_summary_with_openai_model_and_key_is_allowed(
    monkeypatch, openai_key
):
    # Arrange
    created = _install_session_creation(monkeypatch)
    _install_session_access(monkeypatch)
    seeded = _install_summary_seeding(monkeypatch)

    # Act
    response = await agent.restore_session_summary(
        _JsonRequest(),
        {"messages": RESTORE_MESSAGES, "model": GPT_62_SOL_MODEL_ID},
        {"user_id": "u1"},
    )

    # Assert
    assert response.model == GPT_62_SOL_MODEL_ID
    assert created == [GPT_62_SOL_MODEL_ID]
    assert seeded == ["s1"]


# ── Usage attribution ───────────────────────────────────────────────────


def test_usage_counts_direct_openai_cost_but_not_router_gpt_55_cost():
    # Arrange
    events = [
        _llm_call(GPT_62_SOL_MODEL_ID, 2.5),
        _llm_call(GPT_55_DIRECT_MODEL_ID, 1.25),
        _llm_call(GPT_55_MODEL_ID, 4.0),
        _llm_call(f"openai/{GPT_55_MODEL_ID}", 0.5),
    ]

    # Act
    bucket = aggregate_usage_events(events, session_id="s1")

    # Assert
    assert bucket["direct_provider_inference_usd"] == 3.75
    assert bucket["inference_usd"] == 8.25


# ── LLM health probe ────────────────────────────────────────────────────


@pytest.mark.parametrize("model_id", OPENAI_IDS)
def test_health_probe_kwargs_for_openai_models(openai_key, model_id):
    # Act
    kwargs = agent._health_probe_kwargs(model_id, "hf-user-token")

    # Assert
    assert kwargs == {
        "model": model_id,
        "api_base": OPENAI_API_BASE,
        "api_key": DUMMY_KEYS[OPENAI_KEY_ENV],
        "max_completion_tokens": 64,
        "extra_body": {"store": False},
        "messages": PROBE_MESSAGES,
        "timeout": 10,
    }


async def test_llm_health_probes_openai_default_with_a_64_token_cap(
    openai_default, no_hf_token, probe_calls
):
    # Act
    response = await agent.llm_health_check(_Request(), {"user_id": "u1"})

    # Assert
    assert response.status == "ok"
    assert response.model == GPT_62_SOL_MODEL_ID
    assert probe_calls == [
        {
            "model": GPT_62_SOL_MODEL_ID,
            "api_base": OPENAI_API_BASE,
            "api_key": DUMMY_KEYS[OPENAI_KEY_ENV],
            "max_completion_tokens": 64,
            "extra_body": {"store": False},
            "messages": PROBE_MESSAGES,
            "timeout": 10,
        }
    ]
    assert "max_tokens" not in probe_calls[0]
    assert "reasoning_effort" not in probe_calls[0]


@pytest.mark.parametrize("missing_value", MISSING_KEY_VALUES)
async def test_llm_health_missing_openai_key_is_an_auth_error_without_a_call(
    monkeypatch, other_provider_keys, no_hf_token, probe_calls, missing_value
):
    # Arrange: the key vanished after the default was resolved.
    _set_openai_key(monkeypatch, missing_value)
    monkeypatch.setattr(agent, "resolve_default_model_id", lambda: GPT_62_SOL_MODEL_ID)

    # Act
    response = await agent.llm_health_check(_Request(), {"user_id": "u1"})

    # Assert
    assert response.status == "error"
    assert response.model == GPT_62_SOL_MODEL_ID
    assert response.error_type == "auth"
    assert "OPENAI_API_KEY" in response.error
    _assert_no_key_value(response.error)
    assert probe_calls == []


async def test_llm_health_openai_probe_cut_off_by_the_cap_is_ok(
    monkeypatch, openai_default, no_hf_token
):
    # Arrange: OpenAI accepted the probe but spent the 64-token cap reasoning.
    async def capped_acompletion(**_kwargs):
        raise _output_cap_error()

    monkeypatch.setattr(agent, "acompletion", capped_acompletion)

    # Act
    response = await agent.llm_health_check(_Request(), {"user_id": "u1"})

    # Assert
    assert response.status == "ok"
    assert response.model == GPT_62_SOL_MODEL_ID
    assert response.error is None
    assert response.error_type is None


async def test_llm_health_cap_cutoff_text_on_a_router_model_stays_an_error(
    monkeypatch, no_provider_keys
):
    # Arrange
    async def capped_acompletion(**_kwargs):
        raise _output_cap_error()

    monkeypatch.setattr(agent, "acompletion", capped_acompletion)

    # Act
    response = await agent.llm_health_check(
        _Request({"Authorization": "Bearer user-token"}), {"user_id": "u1"}
    )

    # Assert
    assert response.status == "error"
    assert response.model == GLM_52_MODEL_ID


async def test_llm_health_wire_probe_for_openai_default_survives_a_capped_reply(
    openai_default, no_hf_token, capped_openai_http
):
    # Act: the real LiteLLM Responses bridge, with only the HTTP send patched.
    response = await agent.llm_health_check(_Request(), {"user_id": "u1"})

    # Assert
    assert response.status == "ok"
    assert len(capped_openai_http) == 1
    request = capped_openai_http[0]
    assert (request.method, str(request.url)) == (
        "POST",
        f"{OPENAI_API_BASE}/responses",
    )
    body = json.loads(request.content)
    assert body["model"] == "gpt-6.2-sol"
    assert body["max_output_tokens"] == 64
    assert body["store"] is False
    assert "reasoning" not in body
    assert "max_tokens" not in body


async def test_llm_health_openai_quota_is_reported_as_credits(
    monkeypatch, openai_default, no_hf_token
):
    # Arrange
    async def failing_acompletion(**_kwargs):
        raise _openai_quota_error("gpt-6.2-sol")

    monkeypatch.setattr(agent, "acompletion", failing_acompletion)

    # Act
    response = await agent.llm_health_check(_Request(), {"user_id": "u1"})

    # Assert
    assert response.status == "error"
    assert response.model == GPT_62_SOL_MODEL_ID
    assert response.error_type == "credits"


async def test_llm_health_router_quota_429_stays_a_rate_limit(
    monkeypatch, no_provider_keys
):
    # Arrange: LiteLLM tags HF Router errors with llm_provider="openai" too.
    async def failing_acompletion(**_kwargs):
        raise _openai_quota_error(GLM_52_MODEL_ID)

    monkeypatch.setattr(agent, "acompletion", failing_acompletion)

    # Act
    response = await agent.llm_health_check(
        _Request({"Authorization": "Bearer user-token"}), {"user_id": "u1"}
    )

    # Assert
    assert response.status == "error"
    assert response.model == GLM_52_MODEL_ID
    assert response.error_type == "rate_limit"


@pytest.mark.parametrize(
    ("model_id", "expected"),
    [
        (GPT_62_SOL_MODEL_ID, "credits"),
        (GPT_55_DIRECT_MODEL_ID, "credits"),
        (GPT_55_MODEL_ID, "rate_limit"),
        (GLM_52_MODEL_ID, "rate_limit"),
        (None, "rate_limit"),
    ],
)
def test_health_error_type_treats_quota_as_credits_only_for_direct_openai(
    model_id, expected
):
    error = _openai_quota_error("gpt-6.2-sol")

    assert agent._health_error_type(error, model_id) == expected


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            litellm.RateLimitError(
                message=(
                    "OpenAIException - Rate limit reached for gpt-6.2-sol on "
                    "tokens per min (TPM): Limit 30000, Used 30000."
                ),
                model="gpt-6.2-sol",
                llm_provider="openai",
            ),
            "rate_limit",
        ),
        (
            litellm.AuthenticationError(
                message=(
                    "OpenAIException - Incorrect API key provided: sk-dummy***. "
                    "You can find your API key at "
                    "https://platform.openai.com/account/api-keys."
                ),
                model="gpt-6.2-sol",
                llm_provider="openai",
            ),
            "auth",
        ),
    ],
)
def test_health_error_type_keeps_other_openai_errors(error, expected):
    assert agent._health_error_type(error, GPT_62_SOL_MODEL_ID) == expected
