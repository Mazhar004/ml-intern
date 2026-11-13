"""Web default model: Claude Opus 5.5, or GLM 5.2 when its key is missing.

Covers the resolver in ``backend/web_models.py`` and the places the web app
shows or uses it: the picker, ``/api/config/model``, new sessions and the LLM
health probe. No test here makes a network call: the LiteLLM contract test
patches LiteLLM's async HTTP handler, and every key is a dummy value.
"""

import json as json_module
import sys
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import urlsplit

import httpx
import pytest
from litellm.llms.custom_httpx import http_handler

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_BACKEND_DIR = _REPO_ROOT / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

from agent.core import web_models  # noqa: E402
from agent.config import substitute_env_vars  # noqa: E402
from agent.core.model_ids import (  # noqa: E402
    ANTHROPIC_API_BASE_URL,
    CLAUDE_OPUS_48_MODEL_ID,
    CLAUDE_OPUS_55_MODEL_ID,
    CLAUDE_SONNET_55_MODEL_ID,
    GLM_52_MODEL_ID,
    GPT_55_MODEL_ID,
    HF_ROUTER_BASE_URL,
    KIMI_K27_CODE_MODEL_ID,
    MINIMAX_M3_MODEL_ID,
)
from routes import agent  # noqa: E402

DUMMY_ANTHROPIC_KEY = "sk-ant-test"
DEFAULT_MODEL_ENV = "ML_INTERN_DEFAULT_MODEL_ID"
DEEPSEEK_PRO_ID = "deepseek/deepseek-v4-pro"
DEEPSEEK_FLASH_ID = "deepseek/deepseek-v4-flash"
FRONTEND_CONFIG_PATH = _REPO_ROOT / "configs" / "frontend_agent_config.json"
PROBE_MESSAGES = [{"role": "user", "content": "hi"}]
FORBIDDEN_PROBE_KEYS = (
    "reasoning_effort",
    "output_config",
    "thinking",
    "extra_body",
    "temperature",
    "top_p",
    "top_k",
)
# What Claude Opus 5.5 returns when max_tokens stops it inside its thinking.
_CANNED_PROBE_RESPONSE = {
    "id": "msg_probe",
    "type": "message",
    "role": "assistant",
    "model": "claude-opus-5-5",
    "content": [{"type": "thinking", "thinking": "", "signature": "SIG_PROBE"}],
    "stop_reason": "max_tokens",
    "stop_sequence": None,
    "usage": {"input_tokens": 8, "output_tokens": 1},
}


class _Request:
    """Minimal stand-in for a FastAPI Request."""

    def __init__(self, headers: dict | None = None) -> None:
        self.headers = headers or {}
        self.cookies: dict = {}


@pytest.fixture
def anthropic_key(monkeypatch):
    """Dummy Anthropic key and no default override (ignores any local .env)."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_ANTHROPIC_KEY)
    monkeypatch.delenv(DEFAULT_MODEL_ENV, raising=False)


@pytest.fixture
def no_anthropic_key(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv(DEFAULT_MODEL_ENV, raising=False)


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


@pytest.fixture
def captured_requests(monkeypatch) -> list[dict]:
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
            content=json_module.dumps(_CANNED_PROBE_RESPONSE).encode(),
            headers={"content-type": "application/json"},
            request=httpx.Request("POST", str(url)),
        )

    monkeypatch.setattr(http_handler.AsyncHTTPHandler, "post", fake_post)
    return captured


def _header(headers: dict, name: str):
    lowered = {str(k).lower(): v for k, v in headers.items()}
    return lowered.get(name.lower())


# ── Resolver ────────────────────────────────────────────────────────────


def test_default_is_claude_opus_55_when_anthropic_key_is_set(anthropic_key):
    assert web_models.resolve_default_model_id() == CLAUDE_OPUS_55_MODEL_ID


@pytest.mark.parametrize("missing_value", [None, "", "   "])
def test_default_falls_back_to_glm_when_anthropic_key_is_missing(
    monkeypatch, missing_value
):
    # Arrange
    monkeypatch.delenv(DEFAULT_MODEL_ENV, raising=False)
    if missing_value is None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    else:
        monkeypatch.setenv("ANTHROPIC_API_KEY", missing_value)

    # Act
    model = web_models.resolve_default_model_id()

    # Assert
    assert model == GLM_52_MODEL_ID


def test_default_reads_the_key_at_call_time(monkeypatch, no_anthropic_key):
    # Act / Assert: the key appearing or disappearing changes the next answer.
    assert web_models.resolve_default_model_id() == GLM_52_MODEL_ID
    monkeypatch.setenv("ANTHROPIC_API_KEY", DUMMY_ANTHROPIC_KEY)
    assert web_models.resolve_default_model_id() == CLAUDE_OPUS_55_MODEL_ID
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    assert web_models.resolve_default_model_id() == GLM_52_MODEL_ID


@pytest.mark.parametrize(
    "configured",
    [GPT_55_MODEL_ID, f"  {GPT_55_MODEL_ID}  ", GLM_52_MODEL_ID],
)
def test_env_default_naming_a_picker_model_wins(monkeypatch, anthropic_key, configured):
    monkeypatch.setenv(DEFAULT_MODEL_ENV, configured)

    assert web_models.resolve_default_model_id() == configured.strip()


@pytest.mark.parametrize(
    "configured",
    [
        "",
        "unsupported/model",
        "deepseek-ai/DeepSeek-V4-Pro:novita",
        f"huggingface/{CLAUDE_OPUS_55_MODEL_ID}",
    ],
)
def test_env_default_not_in_the_picker_is_ignored(
    monkeypatch, anthropic_key, configured
):
    monkeypatch.setenv(DEFAULT_MODEL_ENV, configured)

    assert web_models.configured_default_model_id() == CLAUDE_OPUS_55_MODEL_ID
    assert web_models.resolve_default_model_id() == CLAUDE_OPUS_55_MODEL_ID


@pytest.mark.parametrize(
    ("configured", "key_env"),
    [
        (CLAUDE_SONNET_55_MODEL_ID, "ANTHROPIC_API_KEY"),
        (DEEPSEEK_PRO_ID, "DEEPSEEK_API_KEY"),
    ],
)
def test_env_default_direct_model_without_its_key_falls_back_to_glm(
    monkeypatch, configured, key_env
):
    # Arrange
    monkeypatch.setenv(DEFAULT_MODEL_ENV, configured)
    monkeypatch.delenv(key_env, raising=False)

    # Act
    resolved = web_models.resolve_default_model_id()

    # Assert
    assert web_models.configured_default_model_id() == configured
    assert resolved == GLM_52_MODEL_ID


def test_env_default_direct_model_with_its_key_is_used(monkeypatch):
    monkeypatch.setenv(DEFAULT_MODEL_ENV, DEEPSEEK_FLASH_ID)
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-deepseek-test")

    assert web_models.resolve_default_model_id() == DEEPSEEK_FLASH_ID


@pytest.mark.parametrize(
    "model_id",
    [CLAUDE_OPUS_55_MODEL_ID, CLAUDE_OPUS_48_MODEL_ID, GLM_52_MODEL_ID, None],
)
def test_missing_key_env_is_none_for_router_models_or_set_keys(anthropic_key, model_id):
    assert web_models.missing_direct_provider_key_env(model_id) is None


def test_missing_key_env_names_only_the_variable(no_anthropic_key):
    missing = web_models.missing_direct_provider_key_env(CLAUDE_OPUS_55_MODEL_ID)

    assert missing == "ANTHROPIC_API_KEY"


# ── Picker ──────────────────────────────────────────────────────────────


def test_picker_order_is_unchanged():
    ids = [model["id"] for model in agent.AVAILABLE_MODELS]

    assert ids == [
        CLAUDE_OPUS_48_MODEL_ID,
        CLAUDE_OPUS_55_MODEL_ID,
        CLAUDE_SONNET_55_MODEL_ID,
        GPT_55_MODEL_ID,
        KIMI_K27_CODE_MODEL_ID,
        MINIMAX_M3_MODEL_ID,
        GLM_52_MODEL_ID,
        DEEPSEEK_PRO_ID,
        DEEPSEEK_FLASH_ID,
    ]


def test_picker_recommends_claude_opus_55_and_keeps_glm_entry():
    # Arrange
    models = {model["id"]: model for model in agent.AVAILABLE_MODELS}

    # Act
    recommended = [m["id"] for m in agent.AVAILABLE_MODELS if m.get("recommended")]

    # Assert
    assert recommended == [CLAUDE_OPUS_55_MODEL_ID]
    assert models[CLAUDE_OPUS_55_MODEL_ID] == {
        "id": CLAUDE_OPUS_55_MODEL_ID,
        "label": "Claude Opus 5.5",
        "recommended": True,
    }
    assert models[GLM_52_MODEL_ID] == {"id": GLM_52_MODEL_ID, "label": "GLM 5.2"}


def test_routes_share_the_catalog_picker_list():
    assert agent.AVAILABLE_MODELS is web_models.AVAILABLE_MODELS


# ── /api/config/model ───────────────────────────────────────────────────


async def test_config_model_current_is_claude_with_key(monkeypatch, anthropic_key):
    monkeypatch.setattr(
        agent.session_manager, "config", SimpleNamespace(model_name=GLM_52_MODEL_ID)
    )

    response = await agent.get_model()

    assert response["current"] == CLAUDE_OPUS_55_MODEL_ID
    assert response["available"] is agent.AVAILABLE_MODELS


async def test_config_model_current_falls_back_to_glm_without_key(
    monkeypatch, no_anthropic_key
):
    # The config string alone must not decide "current".
    monkeypatch.setattr(
        agent.session_manager,
        "config",
        SimpleNamespace(model_name=CLAUDE_OPUS_55_MODEL_ID),
    )

    response = await agent.get_model()

    assert response["current"] == GLM_52_MODEL_ID


def test_frontend_config_fallback_is_claude_opus_55(monkeypatch):
    monkeypatch.delenv(DEFAULT_MODEL_ENV, raising=False)
    raw = json_module.loads(FRONTEND_CONFIG_PATH.read_text())

    assert substitute_env_vars(raw)["model_name"] == CLAUDE_OPUS_55_MODEL_ID


# ── New sessions ────────────────────────────────────────────────────────


def test_new_session_without_model_uses_claude_with_key(anthropic_key):
    assert agent._model_override_for_new_session(None) == CLAUDE_OPUS_55_MODEL_ID


def test_new_session_without_model_uses_glm_without_key(no_anthropic_key):
    assert agent._model_override_for_new_session(None) == GLM_52_MODEL_ID


@pytest.mark.parametrize("model_id", [GLM_52_MODEL_ID, CLAUDE_OPUS_48_MODEL_ID])
def test_explicit_new_session_model_is_honored_with_key(anthropic_key, model_id):
    assert agent._model_override_for_new_session(model_id) == model_id


# ── LLM health probe ────────────────────────────────────────────────────


async def test_llm_health_probes_claude_default_directly_without_hf_token(
    anthropic_key, no_hf_token, probe_calls
):
    # Act
    response = await agent.llm_health_check(_Request(), {"user_id": "u1"})

    # Assert
    assert response.status == "ok"
    assert response.model == CLAUDE_OPUS_55_MODEL_ID
    assert probe_calls == [
        {
            "model": CLAUDE_OPUS_55_MODEL_ID,
            "api_base": ANTHROPIC_API_BASE_URL,
            "api_key": DUMMY_ANTHROPIC_KEY,
            "max_tokens": 1,
            "messages": PROBE_MESSAGES,
            "timeout": 10,
        }
    ]
    for key in FORBIDDEN_PROBE_KEYS:
        assert key not in probe_calls[0]


_ignore_litellm_loop_change_warning = pytest.mark.filterwarnings(
    "ignore:coroutine 'Logging.async_success_handler' was never awaited:RuntimeWarning"
)


@_ignore_litellm_loop_change_warning
async def test_llm_health_probe_wire_request_for_claude_default(
    anthropic_key, no_hf_token, captured_requests
):
    # Act: the real LiteLLM request path, with only the HTTP send patched.
    response = await agent.llm_health_check(_Request(), {"user_id": "u1"})

    # Assert
    assert response.status == "ok"
    assert len(captured_requests) == 1
    request = captured_requests[0]
    url = urlsplit(request["url"])
    assert (url.hostname, url.path) == ("api.anthropic.com", "/v1/messages")
    assert _header(request["headers"], "x-api-key") == DUMMY_ANTHROPIC_KEY
    assert _header(request["headers"], "anthropic-beta") is None
    assert request["body"] == {
        "model": "claude-opus-5-5",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "max_tokens": 1,
    }


async def test_llm_health_missing_key_is_a_clear_error_not_an_exception(
    monkeypatch, no_anthropic_key, no_hf_token, probe_calls
):
    # Arrange: the key vanished after the default was resolved.
    monkeypatch.setattr(
        agent, "resolve_default_model_id", lambda: CLAUDE_OPUS_55_MODEL_ID
    )

    # Act
    response = await agent.llm_health_check(_Request(), {"user_id": "u1"})

    # Assert
    assert response.status == "error"
    assert response.model == CLAUDE_OPUS_55_MODEL_ID
    assert response.error_type == "auth"
    assert "ANTHROPIC_API_KEY" in response.error
    assert "sk-" not in response.error
    assert probe_calls == []


async def test_llm_health_without_key_probes_glm_with_unchanged_router_shape(
    no_anthropic_key, probe_calls
):
    # Act
    response = await agent.llm_health_check(
        _Request({"Authorization": "Bearer user-token"}), {"user_id": "u1"}
    )

    # Assert
    assert response.status == "ok"
    assert response.model == GLM_52_MODEL_ID
    assert probe_calls == [
        {
            "model": f"openai/{GLM_52_MODEL_ID}",
            "api_base": HF_ROUTER_BASE_URL,
            "api_key": "user-token",
            "extra_body": {"reasoning_effort": "high"},
            "max_tokens": 1,
            "messages": PROBE_MESSAGES,
            "timeout": 10,
        }
    ]


async def test_llm_health_without_key_or_hf_token_skips_glm_probe(
    no_anthropic_key, no_hf_token, probe_calls
):
    response = await agent.llm_health_check(_Request(), {"user_id": "u1"})

    assert response.status == "skipped"
    assert response.model == GLM_52_MODEL_ID
    assert probe_calls == []


# ── health-check error classification ──────────────────────────────────


def _anthropic_error(cls, error_type: str, message: str):
    return cls(
        message=(
            'AnthropicException - {"type":"error","error":{"type":"'
            + error_type
            + '","message":"'
            + message
            + '"}}'
        ),
        model="claude-opus-5-5",
        llm_provider="anthropic",
    )


def _health_errors():
    import litellm

    return [
        (
            _anthropic_error(
                litellm.BadRequestError,
                "invalid_request_error",
                "Your credit balance is too low to access the Anthropic API. "
                "Please go to Plans & Billing to upgrade or purchase credits.",
            ),
            "credits",
        ),
        (
            _anthropic_error(
                litellm.AuthenticationError, "authentication_error", "invalid x-api-key"
            ),
            "auth",
        ),
        (
            _anthropic_error(
                litellm.RateLimitError,
                "rate_limit_error",
                "Number of requests exceeded",
            ),
            "rate_limit",
        ),
        (
            Exception(
                "litellm.AuthenticationError: Error code: 401 - Invalid credentials "
                "in Authorization header"
            ),
            "auth",
        ),
        (Exception("Error code: 402 - You have no remaining credits"), "credits"),
    ]


@pytest.mark.parametrize("error, expected", _health_errors())
async def test_llm_health_classifies_provider_errors(
    monkeypatch, anthropic_key, error, expected
):
    async def failing_acompletion(**_kwargs):
        raise error

    monkeypatch.setattr(agent, "acompletion", failing_acompletion)

    response = await agent.llm_health_check(_Request(), {"user_id": "u1"})

    assert response.status == "error"
    assert response.error_type == expected


# ── configured default: env handling ───────────────────────────────────


def test_invalid_default_env_value_is_logged_once(monkeypatch, caplog):
    monkeypatch.setenv(DEFAULT_MODEL_ENV, "ollama/not-a-picker-model-for-log-test")

    with caplog.at_level("WARNING", logger=web_models.logger.name):
        for _ in range(3):
            web_models.configured_default_model_id()

    warnings = [r for r in caplog.records if "not a model picker id" in r.message]
    assert len(warnings) == 1


def test_huggingface_prefixed_default_env_value_is_accepted(monkeypatch):
    monkeypatch.setenv(DEFAULT_MODEL_ENV, "huggingface/zai-org/GLM-5.2:novita")

    assert web_models.configured_default_model_id() == GLM_52_MODEL_ID


# ── import layout ──────────────────────────────────────────────────────


def test_session_manager_imports_with_only_the_repo_root_on_the_path():
    import subprocess

    result = subprocess.run(
        [sys.executable, "-c", "import backend.session_manager"],
        cwd=_REPO_ROOT,
        env={"PATH": "/usr/bin:/bin", "PYTHONPATH": str(_REPO_ROOT)},
        capture_output=True,
        text=True,
        timeout=120,
    )

    assert result.returncode == 0, result.stderr[-800:]
