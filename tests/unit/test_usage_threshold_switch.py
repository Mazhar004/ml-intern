"""ML_INTERN_USAGE_THRESHOLD_APPROVALS=0 turns off the $5 / $10 / $20 … pauses.

A personal deployment running on the operator's own API keys doesn't want the
agent to stop and ask for approval every time session spend doubles.
"""

import pytest

from agent.core.session_persistence import NoopSessionStore
from agent.core.usage_thresholds import (
    USAGE_THRESHOLD_APPROVALS_ENV,
    usage_threshold_approvals_enabled,
)
from test_session_manager_persistence import (
    _manager_with_store,
    _runtime_agent_session,
)


@pytest.mark.parametrize("value", ["0", "false", "no", "off", " OFF "])
def test_usage_threshold_approvals_can_be_turned_off(monkeypatch, value):
    monkeypatch.setenv(USAGE_THRESHOLD_APPROVALS_ENV, value)

    assert usage_threshold_approvals_enabled() is False


@pytest.mark.parametrize("value", [None, "", "1", "true", "on"])
def test_usage_threshold_approvals_stay_on_by_default(monkeypatch, value):
    if value is None:
        monkeypatch.delenv(USAGE_THRESHOLD_APPROVALS_ENV, raising=False)
    else:
        monkeypatch.setenv(USAGE_THRESHOLD_APPROVALS_ENV, value)

    assert usage_threshold_approvals_enabled() is True


async def test_session_never_pauses_for_spend_when_approvals_are_off(monkeypatch):
    # Arrange: spend far past every threshold, approvals switched off
    monkeypatch.setenv(USAGE_THRESHOLD_APPROVALS_ENV, "0")
    manager = _manager_with_store(NoopSessionStore())
    agent_session = _runtime_agent_session("s1")
    manager.sessions["s1"] = agent_session
    billing_lookups = 0

    async def fake_current_session_usage_spend(_agent_session, **_kwargs):
        nonlocal billing_lookups
        billing_lookups += 1
        return 50.0, "app_telemetry_session"

    monkeypatch.setattr(
        manager, "_current_session_usage_spend", fake_current_session_usage_spend
    )
    manager._install_usage_threshold_checker(agent_session)

    # Act
    paused = await agent_session.session.usage_threshold_checker(
        {"continuation": "continue_agent", "history_size": 4, "force_check": True}
    )

    # Assert
    assert paused is False
    assert agent_session.session.pending_approval is None
    assert billing_lookups == 0
