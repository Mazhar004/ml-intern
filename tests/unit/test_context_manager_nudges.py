"""Undo / edit must count only the user turns the web UI shows.

The frontend hides user messages that start with "[SYSTEM:" (doom-loop
nudges, restore notices, tool results recorded as text after a direct-Claude
history repair), so the backend must skip them when counting user turns.
"""

from litellm import Message

from agent.context_manager.manager import ContextManager


def _manager(items: list[Message]) -> ContextManager:
    cm = ContextManager(tool_specs=[])
    cm.items = [cm.items[0], *items]
    return cm


def _contents(cm: ContextManager) -> list[str]:
    return [m.content for m in cm.items[1:]]


def test_undo_removes_through_the_last_visible_user_message():
    cm = _manager(
        [
            Message(role="user", content="first"),
            Message(role="assistant", content="answer one"),
            Message(role="user", content="list files"),
            Message(role="assistant", content="[called tool `bash` with arguments {}]"),
            Message(role="user", content="[SYSTEM: tool results recorded as text]\nx"),
            Message(role="assistant", content="Found x."),
        ]
    )

    assert cm.undo_last_turn() is True

    assert _contents(cm) == ["first", "answer one"]


def test_undo_without_nudges_still_removes_one_turn():
    cm = _manager(
        [
            Message(role="user", content="first"),
            Message(role="assistant", content="answer one"),
            Message(role="user", content="second"),
            Message(role="assistant", content="answer two"),
        ]
    )

    assert cm.undo_last_turn() is True

    assert _contents(cm) == ["first", "answer one"]


def test_undo_with_only_nudges_reports_no_user_turn():
    cm = _manager([Message(role="user", content="[SYSTEM: restored session]")])

    assert cm.undo_last_turn() is False

    assert _contents(cm) == ["[SYSTEM: restored session]"]


def test_truncate_counts_only_visible_user_messages():
    cm = _manager(
        [
            Message(role="user", content="[SYSTEM: restored session]"),
            Message(role="user", content="first"),
            Message(role="assistant", content="answer one"),
            Message(role="user", content="[SYSTEM: doom-loop nudge]"),
            Message(role="user", content="second"),
            Message(role="assistant", content="answer two"),
        ]
    )

    assert cm.truncate_to_user_message(1) is True

    assert _contents(cm) == [
        "[SYSTEM: restored session]",
        "first",
        "answer one",
        "[SYSTEM: doom-loop nudge]",
    ]
