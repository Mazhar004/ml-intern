"""
HF Agent - Main agent module
"""

import litellm

# Global LiteLLM behavior — set once at package import so both CLI and
# backend entries share the same config.
#   drop_params: quietly drop unsupported params rather than raising
#   suppress_debug_info: hide the noisy "Give Feedback" banner on errors
#   modify_params: let LiteLLM patch provider-specific schema requirements
#     for router-compatible request bodies when possible.
litellm.drop_params = True
litellm.suppress_debug_info = True
litellm.modify_params = True

# Register the direct Anthropic models (Claude Opus/Sonnet 5.5) so context
# size, pricing and capability flags don't depend on LiteLLM's remote
# model-map fetch at import.
from agent.core.direct_anthropic import register_direct_anthropic_models  # noqa: E402

register_direct_anthropic_models()

from agent.core.agent_loop import submission_loop  # noqa: E402

__all__ = ["submission_loop"]
