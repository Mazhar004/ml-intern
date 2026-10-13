"""Canonical model ids for HF Router and direct-provider inference."""

HF_ROUTER_BASE_URL = "https://router.huggingface.co/v1"
ANTHROPIC_API_BASE_URL = "https://api.anthropic.com"

# Direct Anthropic API ids: LiteLLM's native ``anthropic/`` provider, billed to
# ANTHROPIC_API_KEY. The HF Router Claude id below also starts with
# ``anthropic/`` (it carries a ``:fal-ai`` suffix), so always match these by
# exact membership via ``is_direct_anthropic_model_id`` — never by prefix.
CLAUDE_OPUS_55_MODEL_ID = "anthropic/claude-opus-5-5"
CLAUDE_SONNET_55_MODEL_ID = "anthropic/claude-sonnet-5-5"
DIRECT_ANTHROPIC_MODEL_IDS = frozenset(
    {CLAUDE_OPUS_55_MODEL_ID, CLAUDE_SONNET_55_MODEL_ID}
)

# Env var holding the API key for each direct (non-HF-Router) provider.
DIRECT_PROVIDER_API_KEY_ENV = {
    "anthropic": "ANTHROPIC_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
}

# Keep these as verbatim HF Router ids; version punctuation differs by model.
CLAUDE_OPUS_48_MODEL_ID = "anthropic/claude-opus-4.8:fal-ai"
GPT_55_MODEL_ID = "openai/gpt-5.5:fal-ai"
KIMI_K27_CODE_MODEL_ID = "moonshotai/Kimi-K2.7-Code:novita"
MINIMAX_M3_MODEL_ID = "MiniMaxAI/MiniMax-M3:novita"
GLM_52_MODEL_ID = "zai-org/GLM-5.2:novita"
DEEPSEEK_V4_PRO_MODEL_ID = "deepseek-ai/DeepSeek-V4-Pro:novita"

HOSTED_MODEL_IDS = {
    CLAUDE_OPUS_48_MODEL_ID,
    GPT_55_MODEL_ID,
    KIMI_K27_CODE_MODEL_ID,
    MINIMAX_M3_MODEL_ID,
    GLM_52_MODEL_ID,
    DEEPSEEK_V4_PRO_MODEL_ID,
}


def strip_huggingface_model_prefix(model_id: str | None) -> str | None:
    """Return model ids without LiteLLM's optional ``huggingface/`` prefix."""
    if not model_id:
        return model_id
    return model_id.removeprefix("huggingface/")


def is_known_router_model_id(model_id: str | None) -> bool:
    normalized = strip_huggingface_model_prefix(model_id)
    return bool(normalized and normalized in HOSTED_MODEL_IDS)


def is_direct_anthropic_model_id(model_id: str | None) -> bool:
    """True only for ids routed to api.anthropic.com with ANTHROPIC_API_KEY.

    Matches the raw id: a ``huggingface/`` prefix explicitly asks for the HF
    Router, and router ids like ``anthropic/claude-opus-4.8:fal-ai`` never
    match.
    """
    return bool(model_id) and model_id in DIRECT_ANTHROPIC_MODEL_IDS


def direct_provider_for(model_id: str | None) -> str | None:
    """Return the direct provider name for a model id, or None for HF Router.

    The result is a key of ``DIRECT_PROVIDER_API_KEY_ENV``.
    """
    if is_direct_anthropic_model_id(model_id):
        return "anthropic"
    if model_id and model_id.startswith("deepseek/"):
        return "deepseek"
    return None
