"""Read a non-streaming LiteLLM response as one assistant reply.

LiteLLM 1.83's Responses API bridge (the direct OpenAI route) returns every
output text part as its own choice and puts all function calls in one more
choice at the end. A reply with a short preamble and tool calls therefore
arrives as two choices, and reading only ``choices[0]`` loses the tool calls.
Every other route answers with a single choice, which is returned unchanged.
"""

from typing import Any

from litellm import Message

_SPLIT_TEXT_SEPARATOR = "\n\n"
_TOOL_CALLS_FINISH_REASON = "tool_calls"


def _joined_text(parts: list[str | None]) -> str | None:
    return _SPLIT_TEXT_SEPARATOR.join(part for part in parts if part) or None


def reply_message(response: Any) -> Any:
    """The reply's assistant message, merged when the bridge split it.

    A single choice comes back as the same object. Split choices become one
    new Message: text parts joined in order, every tool call in order, and
    the reasoning summaries joined. Like ``choices[0]``, an empty choice list
    raises IndexError.
    """
    choices = response.choices
    if len(choices) <= 1:
        return choices[0].message
    messages = [choice.message for choice in choices]
    tool_calls = [call for message in messages for call in message.tool_calls or ()]
    return Message(
        role="assistant",
        content=_joined_text([message.content for message in messages]),
        tool_calls=tool_calls or None,
        reasoning_content=_joined_text(
            [getattr(message, "reasoning_content", None) for message in messages]
        ),
    )


def reply_finish_reason(response: Any) -> str | None:
    """The reply's finish reason: "tool_calls" when a split reply has any."""
    choices = response.choices
    if len(choices) <= 1:
        return choices[0].finish_reason
    if any(choice.message.tool_calls for choice in choices):
        return _TOOL_CALLS_FINISH_REASON
    return choices[-1].finish_reason
