"""Classify errors from the direct OpenAI API (agent loop, probes, backend).

Only call these for a direct OpenAI model id: HF Router errors carry
``llm_provider == "openai"`` too, so the provider tag can't tell them apart.

LiteLLM appends the whole request to some local errors ("Received
Messages=[...]"). That echo is arbitrary conversation text (it may mention
quotas or API keys), so classification only reads what precedes it.
"""

from typing import Literal

from litellm.exceptions import AuthenticationError, PermissionDeniedError

OpenAIErrorKind = Literal["auth", "quota", "model_access"]

_LITELLM_REQUEST_ECHO_MARKER = "received messages="

_AUTH_PHRASES = ("incorrect api key provided", "invalid_api_key")
_QUOTA_PHRASES = (
    "insufficient_quota",
    "exceeded your current quota",
    "billing_hard_limit_reached",
)
_MODEL_ACCESS_PHRASES = (
    "model_not_found",
    "does not exist or you do not have access",
)
# LiteLLM's Responses API bridge raises this (as an APIConnectionError) when
# OpenAI answered 200 "incomplete" because the output cap ran out before any
# text or tool call was written.
_OUTPUT_CAP_PHRASE = "unable to complete request: max_output_tokens"


def error_head(error: BaseException) -> str:
    """Lowercased error text before LiteLLM's echo of the request."""
    return str(error).lower().partition(_LITELLM_REQUEST_ECHO_MARKER)[0]


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(phrase in text for phrase in phrases)


def _is_auth_rejection(error: BaseException) -> bool:
    return isinstance(error, (AuthenticationError, PermissionDeniedError)) or (
        getattr(error, "status_code", None) == 401
    )


def openai_error_kind(error: BaseException) -> OpenAIErrorKind | None:
    """Return "auth", "quota", "model_access", or None for anything else.

    The text is checked first: a 403 whose body says ``model_not_found``
    (a project without access to the model) is a model-access problem, not
    a bad key.
    """
    head = error_head(error)
    if _contains_any(head, _AUTH_PHRASES):
        return "auth"
    if _contains_any(head, _QUOTA_PHRASES):
        return "quota"
    if _contains_any(head, _MODEL_ACCESS_PHRASES):
        return "model_access"
    if _is_auth_rejection(error):
        return "auth"
    return None


def is_output_cap_error(error: BaseException) -> bool:
    """True when the bridge reported a reply cut off by the output cap.

    The request itself was accepted: for the effort probe the effort works,
    and for a real turn the model spent the whole output budget reasoning.
    """
    return _OUTPUT_CAP_PHRASE in error_head(error)
