"""Backend wiring for the direct Anthropic (and direct DeepSeek) model routes.

Covers the picker entries, per-request API-key gating, lazy-restore model
validation, restore-summary token budget, and direct-provider spend
attribution. No test here makes a network call.
"""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from agent.context_manager import manager as context_manager_module  # noqa: E402
from agent.core.model_ids import (  # noqa: E402
    CLAUDE_OPUS_48_MODEL_ID,
    CLAUDE_OPUS_55_MODEL_ID,
    CLAUDE_SONNET_55_MODEL_ID,
    GLM_52_MODEL_ID,
)
from models import UsageBucket  # noqa: E402
from routes import agent  # noqa: E402
from session_manager import SessionManager  # noqa: E402
from usage import aggregate_usage_events  # noqa: E402

DEEPSEEK_PRO_ID = "deepseek/deepseek-v4-pro"
DEEPSEEK_FLASH_ID = "deepseek/deepseek-v4-flash"
HF_BILLING_PLUS_DIRECT_SOURCE = "hf_billing_plus_direct_provider"
DUMMY_ANTHROPIC_KEY = "sk-ant-test"
MISSING_KEY_MESSAGE = (
    "Claude via the Anthropic API needs ANTHROPIC_API_KEY. Add it to the repo "
    ".env and restart the backend."
)


class _JsonRequest:
    """Minimal stand-in for a FastAPI Request carrying a JSON body."""

    def __init__(self, body: dict | None = None) -> None:
        self._body = body or {}
        self.headers: dict = {}
        self.cookies: dict = {}

    async def json(self) -> dict:
        return self._body


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


def _llm_call(model: str, cost_usd: float) -> dict:
    return {"event_type": "llm_call", "data": {"model": model, "cost_usd": cost_usd}}


# ── Picker ──────────────────────────────────────────────────────────────


def test_picker_lists_direct_claude_55_models_right_after_router_opus_48():
    # Arrange
    ids = [model["id"] for model in agent.AVAILABLE_MODELS]
    models = {model["id"]: model for model in agent.AVAILABLE_MODELS}

    # Act
    router_index = ids.index(CLAUDE_OPUS_48_MODEL_ID)

    # Assert
    assert ids[router_index + 1] == CLAUDE_OPUS_55_MODEL_ID
    assert ids[router_index + 2] == CLAUDE_SONNET_55_MODEL_ID
    assert models[CLAUDE_OPUS_55_MODEL_ID] == {
        "id": CLAUDE_OPUS_55_MODEL_ID,
        "label": "Claude Opus 5.5",
    }
    assert models[CLAUDE_SONNET_55_MODEL_ID] == {
        "id": CLAUDE_SONNET_55_MODEL_ID,
        "label": "Claude Sonnet 5.5",
    }
    assert models[CLAUDE_OPUS_48_MODEL_ID]["label"] == "Claude Opus 4.8"
    assert agent._default_model() == GLM_52_MODEL_ID


@pytest.mark.parametrize(
    "model_id",
    [
        CLAUDE_OPUS_55_MODEL_ID,
        CLAUDE_SONNET_55_MODEL_ID,
        DEEPSEEK_PRO_ID,
        DEEPSEEK_FLASH_ID,
    ],
)
def test_direct_provider_ids_do_not_require_hf_router_token(model_id):
    assert agent._model_requires_hf_router_token(model_id) is False


@pytest.mark.parametrize(
    "model_id",
    [
        CLAUDE_OPUS_48_MODEL_ID,
        f"huggingface/{CLAUDE_OPUS_48_MODEL_ID}",
        GLM_52_MODEL_ID,
    ],
)
def test_router_ids_still_require_hf_router_token(model_id):
    assert agent._model_requires_hf_router_token(model_id) is True


# ── Model switch ────────────────────────────────────────────────────────


@pytest.mark.asyncio
@pytest.mark.parametrize("missing_value", [None, "", "   "])
async def test_switch_to_direct_claude_without_key_is_rejected_clearly(
    monkeypatch, missing_value
):
    # Arrange
    if missing_value is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", missing_value)
    _install_session_access(monkeypatch)
    updated = _install_model_updates(monkeypatch)

    # Act
    with pytest.raises(HTTPException) as exc_info:
        await agent.set_session_model(
            "s1",
            {"model": CLAUDE_OPUS_55_MODEL_ID},
            request=None,
            user={"user_id": "u1"},
        )

    # Assert
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == (
        "Claude Opus 5.5 needs ANTHROPIC_API_KEY on the server. Add it to the "
        "repo .env and restart the backend."
    )
    assert updated == []


@pytest.mark.asyncio
async def test_switch_to_direct_claude_with_key_is_allowed(monkeypatch):
    # Arrange
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_ANTHROPIC_KEY)
    _install_session_access(monkeypatch)
    updated = _install_model_updates(monkeypatch)

    # Act
    response = await agent.set_session_model(
        "s1",
        {"model": CLAUDE_SONNET_55_MODEL_ID},
        request=None,
        user={"user_id": "u1"},
    )

    # Assert
    assert response == {"session_id": "s1", "model": CLAUDE_SONNET_55_MODEL_ID}
    assert updated == [("s1", CLAUDE_SONNET_55_MODEL_ID)]


@pytest.mark.asyncio
async def test_missing_key_detail_names_env_var_but_no_key_values(monkeypatch):
    # Arrange
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")
    _install_session_access(monkeypatch)
    _install_model_updates(monkeypatch)

    # Act
    with pytest.raises(HTTPException) as exc_info:
        await agent.set_session_model(
            "s1",
            {"model": CLAUDE_OPUS_55_MODEL_ID},
            request=None,
            user={"user_id": "u1"},
        )

    # Assert
    assert "ANTHROPIC_API_KEY" in exc_info.value.detail
    assert "sk-" not in exc_info.value.detail


@pytest.mark.asyncio
async def test_switch_to_direct_deepseek_without_key_names_deepseek_key(monkeypatch):
    # Arrange
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    _install_session_access(monkeypatch)
    updated = _install_model_updates(monkeypatch)

    # Act
    with pytest.raises(HTTPException) as exc_info:
        await agent.set_session_model(
            "s1",
            {"model": DEEPSEEK_PRO_ID},
            request=None,
            user={"user_id": "u1"},
        )

    # Assert
    assert exc_info.value.status_code == 400
    assert "DeepSeek V4 Pro" in exc_info.value.detail
    assert "DEEPSEEK_API_KEY" in exc_info.value.detail
    assert updated == []


@pytest.mark.asyncio
async def test_switch_to_router_claude_does_not_need_anthropic_key(monkeypatch):
    # Arrange
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    _install_session_access(monkeypatch)
    updated = _install_model_updates(monkeypatch)

    # Act
    response = await agent.set_session_model(
        "s1",
        {"model": CLAUDE_OPUS_48_MODEL_ID},
        request=None,
        user={"user_id": "u1"},
    )

    # Assert
    assert response["model"] == CLAUDE_OPUS_48_MODEL_ID
    assert updated == [("s1", CLAUDE_OPUS_48_MODEL_ID)]


# ── Session create / restore-summary ────────────────────────────────────


@pytest.mark.asyncio
async def test_create_session_with_direct_claude_without_key_is_rejected(
    monkeypatch,
):
    # Arrange
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    created = _install_session_creation(monkeypatch)

    # Act
    with pytest.raises(HTTPException) as exc_info:
        await agent.create_session(
            _JsonRequest({"model": CLAUDE_OPUS_55_MODEL_ID}), {"user_id": "u1"}
        )

    # Assert
    assert exc_info.value.status_code == 400
    assert "Claude Opus 5.5" in exc_info.value.detail
    assert "ANTHROPIC_API_KEY" in exc_info.value.detail
    assert created == []


@pytest.mark.asyncio
async def test_create_session_with_direct_claude_and_key_uses_direct_id(
    monkeypatch,
):
    # Arrange
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_ANTHROPIC_KEY)
    created = _install_session_creation(monkeypatch)

    # Act
    response = await agent.create_session(
        _JsonRequest({"model": CLAUDE_SONNET_55_MODEL_ID}), {"user_id": "u1"}
    )

    # Assert
    assert response.model == CLAUDE_SONNET_55_MODEL_ID
    assert created == [CLAUDE_SONNET_55_MODEL_ID]


@pytest.mark.asyncio
async def test_create_session_with_default_model_ignores_anthropic_key(monkeypatch):
    # Arrange
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    created = _install_session_creation(monkeypatch)

    # Act
    response = await agent.create_session(_JsonRequest(), {"user_id": "u1"})

    # Assert
    assert response.model == GLM_52_MODEL_ID
    assert created == [GLM_52_MODEL_ID]


@pytest.mark.asyncio
async def test_restore_summary_with_direct_claude_without_key_is_rejected(
    monkeypatch,
):
    # Arrange
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    created = _install_session_creation(monkeypatch)

    # Act
    with pytest.raises(HTTPException) as exc_info:
        await agent.restore_session_summary(
            _JsonRequest(),
            {
                "messages": [{"role": "user", "content": "resume this"}],
                "model": CLAUDE_OPUS_55_MODEL_ID,
            },
            {"user_id": "u1"},
        )

    # Assert
    assert exc_info.value.status_code == 400
    assert "ANTHROPIC_API_KEY" in exc_info.value.detail
    assert created == []


@pytest.mark.asyncio
async def test_restore_summary_maps_missing_provider_key_error_to_400(monkeypatch):
    # Arrange
    from agent.core.llm_params import MissingProviderKeyError

    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_ANTHROPIC_KEY)
    _install_session_creation(monkeypatch)
    _install_session_access(monkeypatch)

    async def fake_seed(session_id, messages):
        raise MissingProviderKeyError(MISSING_KEY_MESSAGE)

    monkeypatch.setattr(agent.session_manager, "seed_from_summary", fake_seed)

    # Act
    with pytest.raises(HTTPException) as exc_info:
        await agent.restore_session_summary(
            _JsonRequest(),
            {
                "messages": [{"role": "user", "content": "resume this"}],
                "model": CLAUDE_OPUS_55_MODEL_ID,
            },
            {"user_id": "u1"},
        )

    # Assert
    assert exc_info.value.status_code == 400
    assert exc_info.value.detail == MISSING_KEY_MESSAGE


# ── Restore-summary token budget ────────────────────────────────────────


def _manager_with_summary_session(model_name: str) -> SessionManager:
    manager = object.__new__(SessionManager)
    manager.sessions = {}
    session = SimpleNamespace(
        config=SimpleNamespace(model_name=model_name),
        hf_token="hf_fake",
        context_manager=SimpleNamespace(items=[]),
    )
    manager.sessions["s1"] = SimpleNamespace(
        session=session,
        tool_router=SimpleNamespace(get_tool_specs_for_llm=lambda: []),
        last_active_at=None,
    )

    async def fake_persist(agent_session, **_):
        return None

    manager.persist_session_snapshot = fake_persist  # type: ignore[method-assign]
    return manager


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("model_name", "expected_max_tokens"),
    [
        (CLAUDE_OPUS_55_MODEL_ID, 16_000),
        (CLAUDE_SONNET_55_MODEL_ID, 16_000),
        (CLAUDE_OPUS_48_MODEL_ID, 4000),
        (GLM_52_MODEL_ID, 4000),
    ],
)
async def test_seed_from_summary_budgets_tokens_for_always_thinking_claude(
    monkeypatch, model_name, expected_max_tokens
):
    # Arrange
    manager = _manager_with_summary_session(model_name)
    seen: list[dict] = []

    async def fake_summarize_messages(messages, **kwargs):
        seen.append(kwargs)
        return "summary text", 10

    monkeypatch.setattr(
        context_manager_module, "summarize_messages", fake_summarize_messages
    )

    # Act
    summarized = await manager.seed_from_summary(
        "s1", [{"role": "user", "content": "resume this"}]
    )

    # Assert
    assert summarized == 1
    assert seen[0]["model_name"] == model_name
    assert seen[0]["max_tokens"] == expected_max_tokens


@pytest.mark.asyncio
async def test_seed_from_summary_without_anthropic_key_fails_before_any_call(
    monkeypatch,
):
    # Arrange
    from agent.core.llm_params import MissingProviderKeyError

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    manager = _manager_with_summary_session(CLAUDE_OPUS_55_MODEL_ID)

    async def fail_acompletion(**kwargs):
        raise AssertionError("must not call LiteLLM without ANTHROPIC_API_KEY")

    monkeypatch.setattr(context_manager_module, "acompletion", fail_acompletion)

    # Act
    with pytest.raises(MissingProviderKeyError) as exc_info:
        await manager.seed_from_summary(
            "s1", [{"role": "user", "content": "resume this"}]
        )

    # Assert
    assert "ANTHROPIC_API_KEY" in str(exc_info.value)
    assert manager.sessions["s1"].session.context_manager.items == []


# ── Lazy restore of saved model ids ─────────────────────────────────────


@pytest.mark.parametrize(
    "model_id",
    [
        CLAUDE_OPUS_55_MODEL_ID,
        CLAUDE_SONNET_55_MODEL_ID,
        DEEPSEEK_PRO_ID,
        DEEPSEEK_FLASH_ID,
    ],
)
def test_saved_direct_provider_model_survives_lazy_restore(model_id):
    assert SessionManager._model_from_saved_metadata(model_id) == model_id


@pytest.mark.parametrize(
    "saved_model",
    [CLAUDE_OPUS_48_MODEL_ID, f"huggingface/{CLAUDE_OPUS_48_MODEL_ID}"],
)
def test_saved_router_claude_still_restores(saved_model):
    assert (
        SessionManager._model_from_saved_metadata(saved_model)
        == CLAUDE_OPUS_48_MODEL_ID
    )


# ── Usage attribution ───────────────────────────────────────────────────


def test_usage_bucket_splits_out_direct_provider_inference_cost():
    # Arrange
    events = [
        _llm_call(CLAUDE_OPUS_55_MODEL_ID, 12.0),
        _llm_call(DEEPSEEK_PRO_ID, 0.1234567),
        _llm_call("openai/zai-org/GLM-5.2:novita", 0.5),
        _llm_call(f"openai/{CLAUDE_OPUS_48_MODEL_ID}", 3.0),
        _llm_call(CLAUDE_OPUS_48_MODEL_ID, 1.0),
    ]

    # Act
    bucket = aggregate_usage_events(events, session_id="s1")

    # Assert
    assert bucket["direct_provider_inference_usd"] == 12.123457
    assert bucket["inference_usd"] == 16.623457
    assert UsageBucket(**bucket).direct_provider_inference_usd == 12.123457


def test_usage_bucket_without_llm_calls_has_zero_direct_provider_cost():
    bucket = aggregate_usage_events([], session_id="s1")

    assert bucket["direct_provider_inference_usd"] == 0.0
    assert UsageBucket().direct_provider_inference_usd == 0.0


def test_usage_threshold_spend_adds_direct_provider_cost_to_hf_billing():
    # Arrange
    bucket = aggregate_usage_events(
        [
            _llm_call(CLAUDE_OPUS_55_MODEL_ID, 12.0),
            _llm_call("openai/zai-org/GLM-5.2:novita", 0.5),
            _llm_call(f"openai/{CLAUDE_OPUS_48_MODEL_ID}", 3.0),
        ],
        session_id="s1",
    )
    response = {
        "hf_account": {"current_session": {"inference_providers_usd": 3.5}},
        "session": bucket,
    }

    # Act
    spend, source = SessionManager._usage_spend_from_response(response)

    # Assert
    assert spend == 15.5
    assert source == HF_BILLING_PLUS_DIRECT_SOURCE


def test_usage_threshold_spend_labels_blended_hf_and_direct_deepseek_spend():
    response = {
        "hf_account": {"current_session": {"inference_providers_usd": 1.0}},
        "session": aggregate_usage_events(
            [_llm_call(DEEPSEEK_PRO_ID, 0.25)], session_id="s1"
        ),
    }

    spend, source = SessionManager._usage_spend_from_response(response)

    assert spend == 1.25
    assert source == HF_BILLING_PLUS_DIRECT_SOURCE


def test_usage_threshold_spend_without_direct_cost_keeps_hf_billing_source():
    # Arrange: router-only calls, so the direct subset aggregates to 0.0.
    bucket = aggregate_usage_events(
        [
            _llm_call("openai/zai-org/GLM-5.2:novita", 0.5),
            _llm_call(f"openai/{CLAUDE_OPUS_48_MODEL_ID}", 3.0),
        ],
        session_id="s1",
    )
    bucket["hf_jobs_estimated_usd"] = 1.25
    response = {
        "hf_account": {"current_session": {"inference_providers_usd": 3.5}},
        "session": bucket,
    }

    # Act
    spend, source = SessionManager._usage_spend_from_response(response)

    # Assert
    assert bucket["direct_provider_inference_usd"] == 0.0
    assert spend == 4.75
    assert source == "hf_billing_current_session"


@pytest.mark.parametrize("direct_value", [None, True, -2.0, "not-a-number"])
def test_usage_threshold_spend_ignores_non_positive_direct_cost(direct_value):
    response = {
        "hf_account": {"current_session": {"inference_providers_usd": 2.0}},
        "session": {"direct_provider_inference_usd": direct_value},
    }

    spend, source = SessionManager._usage_spend_from_response(response)

    assert spend == 2.0
    assert source == "hf_billing_current_session"


def test_usage_threshold_telemetry_spend_counts_direct_cost_once():
    # Without HF billing, inference_usd (and so total_usd) already includes
    # the direct-provider subset; it must not be added a second time.
    bucket = aggregate_usage_events(
        [
            _llm_call(CLAUDE_OPUS_55_MODEL_ID, 12.0),
            _llm_call("openai/zai-org/GLM-5.2:novita", 0.5),
        ],
        session_id="s1",
    )
    response = {"hf_account": {"current_session": None}, "session": bucket}

    spend, source = SessionManager._usage_spend_from_response(response)

    assert bucket["direct_provider_inference_usd"] == 12.0
    assert spend == 12.5
    assert source == "app_telemetry_session"
