"""LiteLLM model-map entries for the direct OpenAI API models.

LiteLLM 1.83's bundled map has none of these models, and whether the remote
map is fetched at import depends on network access. We register them
explicitly so request shape, context size and cost tracking are the same on
every start:

* ``mode: "responses"`` makes LiteLLM bridge every chat call to
  ``POST /v1/responses``. Chat Completions rejects tools for the GPT-6.x
  models, and GPT-5.5 only switches to the Responses API when a call has
  tools *and* an effort, so without it the route would depend on the call.
* ``supports_native_streaming`` must be True: otherwise LiteLLM fakes the
  stream on ``/responses`` and every tool call is lost.
* ``max_input_tokens`` is 272K although the real window is 1.05M. Above 272K
  input tokens OpenAI bills the whole request at the long-context rates
  (``*_above_272k_tokens``), so compaction keeps prompts under that line.

Keys are the bare model names: LiteLLM resolves ``openai/gpt-6.2-sol`` to
``gpt-6.2-sol``. Registering ``gpt-5.5`` also gives the HF Router GPT-5.5
(``openai/gpt-5.5:fal-ai``) a 272K compaction ceiling; its request body is
unchanged.
"""

import litellm

_MAX_INPUT_TOKENS = 272_000
_MAX_OUTPUT_TOKENS = 128_000

_SHARED_MODEL_INFO = {
    "litellm_provider": "openai",
    "mode": "responses",
    "max_input_tokens": _MAX_INPUT_TOKENS,
    "max_output_tokens": _MAX_OUTPUT_TOKENS,
    "max_tokens": _MAX_OUTPUT_TOKENS,
    "supports_native_streaming": True,
    "supports_function_calling": True,
    "supports_parallel_function_calling": True,
    "supports_tool_choice": True,
    "supports_system_messages": True,
    "supports_prompt_caching": True,
    "supports_reasoning": True,
    "supports_minimal_reasoning_effort": False,
    "supports_xhigh_reasoning_effort": True,
}

# Per-token USD prices. Cache writes on the GPT-6.x models cost 1.25x input;
# GPT-5.5 has no cache-write premium, so it sets no cache-write rate.
DIRECT_OPENAI_LITELLM_MODELS: dict[str, dict] = {
    "gpt-5.5": {
        **_SHARED_MODEL_INFO,
        "supports_none_reasoning_effort": True,
        "input_cost_per_token": 5e-6,
        "output_cost_per_token": 3e-5,
        "cache_read_input_token_cost": 5e-7,
        "input_cost_per_token_above_272k_tokens": 1e-5,
        "output_cost_per_token_above_272k_tokens": 4.5e-5,
        "cache_read_input_token_cost_above_272k_tokens": 1e-6,
    },
    "gpt-6.2-astra": {
        **_SHARED_MODEL_INFO,
        "supports_none_reasoning_effort": False,
        "input_cost_per_token": 8e-6,
        "output_cost_per_token": 4e-5,
        "cache_read_input_token_cost": 4e-7,
        "cache_creation_input_token_cost": 1e-5,
        "input_cost_per_token_above_272k_tokens": 1.6e-5,
        "output_cost_per_token_above_272k_tokens": 6e-5,
        "cache_read_input_token_cost_above_272k_tokens": 8e-7,
    },
    "gpt-6.2-sol": {
        **_SHARED_MODEL_INFO,
        "supports_none_reasoning_effort": False,
        "input_cost_per_token": 2e-6,
        "output_cost_per_token": 1e-5,
        "cache_read_input_token_cost": 1e-7,
        "cache_creation_input_token_cost": 2.5e-6,
        "input_cost_per_token_above_272k_tokens": 4e-6,
        "output_cost_per_token_above_272k_tokens": 1.5e-5,
        "cache_read_input_token_cost_above_272k_tokens": 2e-7,
    },
    "gpt-6.1-terra": {
        **_SHARED_MODEL_INFO,
        "supports_none_reasoning_effort": True,
        "input_cost_per_token": 1e-6,
        "output_cost_per_token": 5e-6,
        "cache_read_input_token_cost": 5e-8,
        "cache_creation_input_token_cost": 1.25e-6,
        "input_cost_per_token_above_272k_tokens": 2e-6,
        "output_cost_per_token_above_272k_tokens": 7.5e-6,
        "cache_read_input_token_cost_above_272k_tokens": 1e-7,
    },
    "gpt-6-luna": {
        **_SHARED_MODEL_INFO,
        "supports_none_reasoning_effort": True,
        "input_cost_per_token": 1e-7,
        "output_cost_per_token": 5e-7,
        "cache_read_input_token_cost": 1e-8,
        "cache_creation_input_token_cost": 1.25e-7,
        "input_cost_per_token_above_272k_tokens": 2e-7,
        "output_cost_per_token_above_272k_tokens": 7.5e-7,
        "cache_read_input_token_cost_above_272k_tokens": 2e-8,
    },
}


def register_direct_openai_models() -> None:
    """Merge the direct OpenAI entries into LiteLLM's model map.

    Idempotent; overrides any remote-map values for the same keys. Copies are
    passed so LiteLLM never holds references to the module constants.
    """
    litellm.register_model(
        {name: dict(info) for name, info in DIRECT_OPENAI_LITELLM_MODELS.items()}
    )
