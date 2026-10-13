"""LiteLLM model-map entries for the direct Anthropic API models.

LiteLLM 1.83's bundled map has no Claude Opus 5.5 / Sonnet 5.5 entries, and
whether the remote map is fetched at import depends on network access. We
register both models explicitly so context size, cost tracking and request
shape are the same on every start.

``supports_reasoning`` is set to ``False`` on purpose: with ``True``, LiteLLM
1.83 maps a stray ``reasoning_effort`` to ``thinking.type=enabled`` plus
``budget_tokens``, which these always-thinking models reject with HTTP 400.
Effort goes through ``output_config.effort`` instead (see ``llm_params``).
"""

import litellm

_CONTEXT_WINDOW_TOKENS = 1_000_000
_MAX_OUTPUT_TOKENS = 128_000

_SHARED_MODEL_INFO = {
    "litellm_provider": "anthropic",
    "mode": "chat",
    "max_input_tokens": _CONTEXT_WINDOW_TOKENS,
    "max_output_tokens": _MAX_OUTPUT_TOKENS,
    "max_tokens": _MAX_OUTPUT_TOKENS,
    "supports_function_calling": True,
    "supports_tool_choice": True,
    "supports_prompt_caching": True,
    "supports_vision": True,
    "supports_assistant_prefill": False,
    "supports_reasoning": False,
}

# Per-token USD prices. Cache writes use the 5-minute TTL rate (1.25x input).
DIRECT_ANTHROPIC_LITELLM_MODELS: dict[str, dict] = {
    "claude-opus-5-5": {
        **_SHARED_MODEL_INFO,
        "input_cost_per_token": 4e-6,
        "output_cost_per_token": 2e-5,
        "cache_read_input_token_cost": 2e-7,
        "cache_creation_input_token_cost": 5e-6,
    },
    "claude-sonnet-5-5": {
        **_SHARED_MODEL_INFO,
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 1e-5,
        "cache_read_input_token_cost": 2e-7,
        "cache_creation_input_token_cost": 2.5e-6,
    },
}


def register_direct_anthropic_models() -> None:
    """Merge the direct Anthropic entries into LiteLLM's model map.

    Idempotent; overrides any remote-map values for the same keys. Copies are
    passed so LiteLLM never holds references to the module constants.
    """
    litellm.register_model(
        {name: dict(info) for name, info in DIRECT_ANTHROPIC_LITELLM_MODELS.items()}
    )
