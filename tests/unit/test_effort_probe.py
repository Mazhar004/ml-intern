from types import SimpleNamespace

import pytest

from agent.core import effort_probe
from agent.core.prompt_caching import HF_ROUTER_SESSION_ID_HEADER

BILLING_SESSION_ID = "00000000-0000-4000-8000-000000000001"


@pytest.mark.asyncio
async def test_probe_effort_sends_session_id_to_hf_router(monkeypatch):
    completions = []

    async def fake_acompletion(**kwargs):
        completions.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="stop")],
            usage=None,
        )

    monkeypatch.setattr(effort_probe, "acompletion", fake_acompletion)

    outcome = await effort_probe.probe_effort(
        "moonshotai/Kimi-K2.7-Code:novita",
        "high",
        "hf_fake",
        session=SimpleNamespace(
            session_id="session-1",
            inference_billing_session_id=BILLING_SESSION_ID,
        ),
    )

    assert outcome.effective_effort == "high"
    assert completions[0]["extra_body"] == {"reasoning_effort": "high"}
    assert completions[0]["extra_headers"] == {
        HF_ROUTER_SESSION_ID_HEADER: BILLING_SESSION_ID
    }


@pytest.mark.parametrize(
    "text",
    [
        # OpenAI Responses API wording (via LiteLLM's bridge).
        "litellm.BadRequestError: OpenAIException - Unsupported value: "
        "'reasoning.effort' does not support 'max' with this model.",
        "This model does not support reasoning effort 'xhigh'.",
        "Unsupported value: 'reasoning_effort' must be low, medium or high.",
    ],
)
def test_invalid_effort_matches_openai_wording(text):
    assert effort_probe._is_invalid_effort(Exception(text)) is True


@pytest.mark.parametrize(
    "text",
    [
        "Unsupported value: 'temperature' does not support 0.2 with this model.",
        "This model does not support tools.",
        "Thinking is not supported for this model (reasoning effort high).",
    ],
)
def test_invalid_effort_ignores_errors_that_are_not_about_effort(text):
    assert effort_probe._is_invalid_effort(Exception(text)) is False


@pytest.mark.asyncio
async def test_probe_effort_keeps_max_tokens_cap_for_router_models(monkeypatch):
    completions = []

    async def fake_acompletion(**kwargs):
        completions.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(finish_reason="stop")])

    monkeypatch.setattr(effort_probe, "acompletion", fake_acompletion)

    await effort_probe.probe_effort("zai-org/GLM-5.2:novita", "high", "hf_fake")

    assert completions[0]["max_tokens"] == effort_probe._PROBE_MAX_TOKENS
    assert "max_completion_tokens" not in completions[0]
