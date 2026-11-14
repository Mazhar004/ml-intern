"""Tests for renaming chats and keeping the title durable.

Covers ``PATCH /api/session/{id}/title`` (validation, live + archived-only
sessions, ownership, 404s), the on-disk ``session_logs/`` writes, the restore
paths that read saved metadata back, and the rule that a manual name always
beats the auto-generated one.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from urllib.parse import quote

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from litellm import Message
from pydantic import ValidationError

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import session_log_files  # noqa: E402
from agent.core import session_resume  # noqa: E402
from agent.core.session import Session  # noqa: E402
from agent.core.session_persistence import NoopSessionStore  # noqa: E402
from models import SESSION_TITLE_MAX_CHARS, SessionTitleRequest  # noqa: E402
from routes import agent as agent_routes  # noqa: E402
from session_manager import AgentSession, SessionManager  # noqa: E402

ROUTER_MODEL = "openai/gpt-5.5:fal-ai"


# ── Fakes & helpers ──────────────────────────────────────────────────────


class RecordingStore(NoopSessionStore):
    """Session store that records writes and can serve one saved session."""

    enabled = True

    def __init__(self, metadata: dict[str, Any] | None = None) -> None:
        self.metadata = metadata
        self.updated_fields: list[tuple[str, dict[str, Any]]] = []
        self.snapshots: list[dict[str, Any]] = []

    async def load_session(self, session_id: str, **_: Any) -> dict[str, Any] | None:
        if self.metadata is None:
            return None
        return {"metadata": {**self.metadata, "_id": session_id}, "messages": []}

    async def update_session_fields(self, session_id: str, **fields: Any) -> None:
        self.updated_fields.append((session_id, fields))

    async def save_snapshot(self, **kwargs: Any) -> None:
        self.snapshots.append(kwargs)


def _runtime(session_id: str = "s1") -> SimpleNamespace:
    return SimpleNamespace(
        session_id=session_id,
        hf_token=None,
        sandbox=None,
        pending_approval=None,
        turn_count=0,
        usage_warning_next_threshold_usd=5.0,
        context_manager=SimpleNamespace(items=[]),
        config=SimpleNamespace(model_name="test-model"),
        notification_destinations=[],
    )


def _agent_session(
    session_id: str = "s1",
    *,
    user_id: str = "dev",
    runtime: Any = None,
    title: str | None = None,
    title_is_custom: bool = False,
) -> AgentSession:
    return AgentSession(
        session_id=session_id,
        session=runtime or _runtime(session_id),  # type: ignore[arg-type]
        tool_router=object(),  # type: ignore[arg-type]
        submission_queue=asyncio.Queue(),
        user_id=user_id,
        title=title,
        title_is_custom=title_is_custom,
    )


def _manager(store: NoopSessionStore) -> SessionManager:
    manager = object.__new__(SessionManager)
    manager.config = SimpleNamespace(model_name="test-model")
    manager.sessions = {}
    manager._lock = asyncio.Lock()
    manager.persistence_store = store
    manager._pending_creates = 0
    manager._reaper_task = None
    return manager


def _core_session(session_id: str) -> Session:
    """A real core Session with a stubbed context manager (no litellm lookups)."""
    from agent.context_manager.manager import ContextManager

    cm = ContextManager.__new__(ContextManager)
    cm.items = []
    cm.tool_specs = []
    cm.model_max_tokens = 200_000
    cm.running_context_usage = 0
    cm.compact_size = 0.1
    cm.untouched_messages = 5
    cm.hf_token = None
    cm.local_mode = True
    config = SimpleNamespace(
        model_name="test-model",
        save_sessions=True,
        session_dataset_repo="fake/repo",
        auto_save_interval=1,
        heartbeat_interval_s=60,
    )
    return Session(
        event_queue=asyncio.Queue(),
        config=config,  # type: ignore[arg-type]
        tool_router=None,
        context_manager=cm,
        local_mode=True,
        session_id=session_id,
    )


def _write_log(
    directory: Path,
    session_id: str,
    stamp: str,
    *,
    user_id: str | None = "dev",
    mtime: float | None = None,
    extra: dict[str, Any] | None = None,
) -> Path:
    payload = {
        "session_id": session_id,
        "user_id": user_id,
        "model_name": ROUTER_MODEL,
        "last_save_time": "2026-11-01T10:00:00",
        "messages": [
            {"role": "system", "content": "system prompt"},
            {"role": "user", "content": "Fine-tune a model on my dataset"},
            {"role": "assistant", "content": "On it."},
        ],
        "events": [{"event_type": "llm_call", "data": {"cost_usd": 0.01}}],
        **(extra or {}),
    }
    path = directory / f"session_{session_id}_{stamp}.json"
    path.write_text(json.dumps(payload, indent=2))
    if mtime is not None:
        os.utime(path, (mtime, mtime))
    return path


def _client(user_id: str = "dev") -> TestClient:
    app = FastAPI()
    app.include_router(agent_routes.router)
    app.dependency_overrides[agent_routes.get_current_user] = lambda: {
        "user_id": user_id
    }
    return TestClient(app)


@pytest.fixture
def log_dir(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "session_logs"
    directory.mkdir()
    monkeypatch.setattr(session_log_files, "SESSION_LOG_DIR", directory)
    return directory


@pytest.fixture
def live_manager(monkeypatch) -> SessionManager:
    """The route-level global manager, isolated from real sessions/stores."""
    manager = agent_routes.session_manager
    monkeypatch.setattr(manager, "sessions", {})
    monkeypatch.setattr(manager, "persistence_store", RecordingStore())
    return manager


# ── Validation ───────────────────────────────────────────────────────────


def test_title_request_collapses_whitespace_and_newlines():
    request = SessionTitleRequest(title="  Fine-tune\n\n  Llama\t on  my\r\ndata  ")

    assert request.title == "Fine-tune Llama on my data"


def test_title_request_strips_control_characters():
    request = SessionTitleRequest(title="Run\x00 \x1b[31m42\x07\x7f done")

    assert request.title == "Run [31m42 done"


@pytest.mark.parametrize("raw", ["", "   ", "\n\t\r\n", "\x00\x07\x1b"])
def test_title_request_rejects_titles_that_are_empty_after_cleanup(raw):
    with pytest.raises(ValidationError, match="must not be empty"):
        SessionTitleRequest(title=raw)


def test_title_request_rejects_titles_over_the_limit_instead_of_truncating():
    assert SESSION_TITLE_MAX_CHARS == 80
    exact = "x" * SESSION_TITLE_MAX_CHARS
    assert SessionTitleRequest(title=exact).title == exact
    # Length is checked after whitespace collapse.
    assert SessionTitleRequest(title=f"  {exact}  ").title == exact

    with pytest.raises(ValidationError, match="at most 80 characters"):
        SessionTitleRequest(title="x" * (SESSION_TITLE_MAX_CHARS + 1))


def test_patch_rejects_invalid_title_with_clear_422(live_manager, log_dir):
    live_manager.sessions["s1"] = _agent_session("s1", title="Before")

    response = _client().patch("/api/session/s1/title", json={"title": " \n "})

    assert response.status_code == 422
    assert "must not be empty" in json.dumps(response.json())
    assert live_manager.sessions["s1"].title == "Before"


# ── Live sessions ────────────────────────────────────────────────────────


async def test_set_custom_title_updates_runtime_store_and_session_info(log_dir):
    store = RecordingStore()
    manager = _manager(store)
    agent_session = _agent_session("s1", title="Auto title")
    manager.sessions["s1"] = agent_session

    stored = await manager.set_custom_session_title("s1", "My experiment")

    assert stored == "My experiment"
    assert agent_session.title == "My experiment"
    assert agent_session.title_is_custom is True
    # The runtime session carries the title so every later auto-save keeps it.
    assert agent_session.session.title == "My experiment"
    assert agent_session.session.title_is_custom is True
    assert store.updated_fields == [
        ("s1", {"title": "My experiment", "title_is_custom": True})
    ]
    info = manager.get_session_info("s1")
    assert info["title"] == "My experiment"
    assert info["title_is_custom"] is True


async def test_snapshot_persists_custom_title_flag():
    store = RecordingStore()
    manager = _manager(store)
    agent_session = _agent_session("s1", title="Mine", title_is_custom=True)

    await manager.persist_session_snapshot(agent_session)

    assert store.snapshots[0]["title"] == "Mine"
    assert store.snapshots[0]["title_is_custom"] is True


def test_patch_renames_live_session(live_manager, log_dir):
    live_manager.sessions["s1"] = _agent_session("s1", title="Auto title")

    response = _client().patch(
        "/api/session/s1/title", json={"title": "  Sweep\nresults  "}
    )

    assert response.status_code == 200
    assert response.json() == {"session_id": "s1", "title": "Sweep results"}
    assert live_manager.sessions["s1"].title == "Sweep results"
    assert live_manager.sessions["s1"].title_is_custom is True


def test_live_rename_is_written_to_the_log_and_kept_by_later_saves(
    live_manager, log_dir
):
    core = _core_session("live-1")
    live_manager.sessions["live-1"] = _agent_session("live-1", runtime=core)
    first_save = Path(core.save_trajectory_local(directory=str(log_dir)))
    assert json.loads(first_save.read_text())["title"] is None

    response = _client().patch(
        "/api/session/live-1/title", json={"title": "Durable name"}
    )
    assert response.status_code == 200

    # The existing log is patched immediately (survives a restart before
    # the next auto-save)...
    patched = json.loads(first_save.read_text())
    assert patched["title"] == "Durable name"
    assert patched["title_is_custom"] is True
    # ...and every later auto-save rewrites it from memory with the title.
    core.context_manager.items.append(Message(role="user", content="next turn"))
    second_save = Path(core.save_trajectory_local(directory=str(log_dir)))
    assert second_save == first_save
    resaved = json.loads(second_save.read_text())
    assert resaved["title"] == "Durable name"
    assert resaved["title_is_custom"] is True
    assert resaved["messages"][-1]["content"] == "next turn"


def test_other_users_live_session_is_not_renamed(monkeypatch, live_manager, log_dir):
    live_manager.sessions["s1"] = _agent_session("s1", user_id="alice", title="Hers")

    response = _client(user_id="bob").patch(
        "/api/session/s1/title", json={"title": "Mine now"}
    )

    assert response.status_code == 404
    assert live_manager.sessions["s1"].title == "Hers"
    assert live_manager.sessions["s1"].title_is_custom is False


# ── Archived-only sessions (on disk, not in memory) ──────────────────────


def test_patch_updates_newest_archived_log_atomically(
    monkeypatch, live_manager, log_dir
):
    older = _write_log(log_dir, "arch-1", "20261101_100000", mtime=1_000_000)
    newest = _write_log(
        log_dir,
        "arch-1",
        "20261102_100000",
        mtime=2_000_000,
        extra={"project": "oculomics", "total_cost_usd": 1.5},
    )
    other = _write_log(log_dir, "arch-2", "20261103_100000", mtime=3_000_000)
    older_bytes, other_bytes = older.read_bytes(), other.read_bytes()
    before = json.loads(newest.read_text())
    replaced: list[tuple[str, str]] = []
    real_replace = os.replace

    def spy_replace(src, dst):
        replaced.append((str(src), str(dst)))
        real_replace(src, dst)

    monkeypatch.setattr(session_log_files.os, "replace", spy_replace)

    response = _client().patch(
        "/api/session/arch-1/title", json={"title": "Archived run"}
    )

    assert response.status_code == 200
    assert response.json() == {"session_id": "arch-1", "title": "Archived run"}
    after = json.loads(newest.read_text())
    assert after == {**before, "title": "Archived run", "title_is_custom": True}
    # Atomic: staged in the same directory, then os.replace'd over the file.
    assert len(replaced) == 1
    src, dst = replaced[0]
    assert dst == str(newest)
    assert Path(src).parent == log_dir
    assert not Path(src).exists()
    # Nothing else touched, no temp files left behind, ordering preserved.
    assert older.read_bytes() == older_bytes
    assert other.read_bytes() == other_bytes
    assert sorted(p.name for p in log_dir.iterdir()) == sorted(
        [older.name, newest.name, other.name]
    )
    assert newest.stat().st_mtime == 2_000_000


def test_failed_archive_write_leaves_the_log_untouched(
    monkeypatch, live_manager, log_dir
):
    path = _write_log(log_dir, "arch-1", "20261101_100000")
    original = path.read_bytes()

    def failing_replace(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(session_log_files.os, "replace", failing_replace)

    response = _client().patch("/api/session/arch-1/title", json={"title": "New"})

    assert response.status_code == 500
    assert path.read_bytes() == original
    assert [p.name for p in log_dir.iterdir()] == [path.name]


def test_unknown_session_returns_404(live_manager, log_dir):
    _write_log(log_dir, "arch-1", "20261101_100000")

    response = _client().patch("/api/session/nope/title", json={"title": "New"})

    assert response.status_code == 404


def test_glob_characters_in_session_id_never_match_other_logs(live_manager, log_dir):
    path = _write_log(log_dir, "arch-1", "20261101_100000")
    original = path.read_bytes()

    for session_id in ("*", "arch-?", "[a]rch-1"):
        response = _client().patch(
            f"/api/session/{quote(session_id, safe='')}/title",
            json={"title": "Hijack"},
        )
        assert response.status_code == 404

    assert path.read_bytes() == original


def test_archived_rename_rejects_other_users_log(live_manager, log_dir):
    path = _write_log(log_dir, "arch-1", "20261101_100000", user_id="alice")
    original = path.read_bytes()

    response = _client(user_id="bob").patch(
        "/api/session/arch-1/title", json={"title": "Not yours"}
    )

    assert response.status_code == 403
    assert path.read_bytes() == original


def test_archived_rename_allows_the_owner(live_manager, log_dir):
    path = _write_log(log_dir, "arch-1", "20261101_100000", user_id="alice")

    response = _client(user_id="alice").patch(
        "/api/session/arch-1/title", json={"title": "Mine"}
    )

    assert response.status_code == 200
    assert json.loads(path.read_text())["title"] == "Mine"


async def test_archived_listing_returns_title_or_null(log_dir):
    _write_log(
        log_dir,
        "named",
        "20261101_100000",
        extra={"title": "Named run", "title_is_custom": True},
    )
    _write_log(log_dir, "untitled", "20261101_110000")

    rows = await agent_routes.list_archived_sessions(
        user={"user_id": "dev"},
        min_messages=0,
        include_dummy=True,
        requires_project=False,
    )

    by_id = {row["session_id"]: row for row in rows}
    assert by_id["named"]["title"] == "Named run"
    assert by_id["named"]["title_is_custom"] is True
    assert by_id["untitled"]["title"] is None
    assert by_id["untitled"]["title_is_custom"] is False


# ── Restore paths carry the title back ───────────────────────────────────


async def test_lazy_restore_carries_saved_custom_title():
    store = RecordingStore(
        metadata={
            "session_id": "persisted",
            "user_id": "owner",
            "model": "test-model",
            "title": "Saved name",
            "title_is_custom": True,
        }
    )
    manager = _manager(store)
    stop = asyncio.Event()

    def fake_create_session_sync(**_: Any):
        return object(), _runtime("persisted")

    async def fake_run_session(*_: Any) -> None:
        await stop.wait()

    manager._create_session_sync = fake_create_session_sync  # type: ignore[method-assign]
    manager._run_session = fake_run_session  # type: ignore[method-assign]

    try:
        restored = await manager.ensure_session_loaded(
            "persisted", user_id="owner", preload_sandbox=False
        )

        assert restored is not None
        assert restored.title == "Saved name"
        assert restored.title_is_custom is True
        assert restored.session.title == "Saved name"
        assert restored.session.title_is_custom is True
    finally:
        stop.set()
        if restored is not None and restored.task is not None:
            await restored.task


def test_import_carries_saved_title_into_new_session(
    monkeypatch, live_manager, log_dir
):
    _write_log(
        log_dir,
        "arch-1",
        "20261101_100000",
        extra={"title": "Imported name", "title_is_custom": True},
    )

    async def fake_create_session(**_: Any) -> str:
        live_manager.sessions["fresh-id"] = _agent_session("fresh-id")
        return "fresh-id"

    monkeypatch.setattr(live_manager, "create_session", fake_create_session)

    response = _client().post("/api/session/import", json={"session_id": "arch-1"})

    assert response.status_code == 200
    imported = live_manager.sessions["fresh-id"]
    assert imported.title == "Imported name"
    assert imported.title_is_custom is True
    assert imported.session.title == "Imported name"


def test_cli_resume_carries_saved_title(tmp_path):
    path = _write_log(
        tmp_path,
        "arch-1",
        "20261101_100000",
        extra={"title": "CLI name", "title_is_custom": True},
    )

    class FakeContext:
        items = [Message(role="system", content="system")]

        def _recompute_usage(self, model_name: str) -> None:
            pass

    session = SimpleNamespace(
        context_manager=FakeContext(),
        config=SimpleNamespace(model_name="test-model"),
        session_id="current",
        session_start_time="2026-01-01T00:00:00",
        user_id="dev",
        logged_events=[],
        _local_save_path=None,
        turn_count=0,
        last_auto_save_turn=0,
        pending_approval=None,
        title=None,
        title_is_custom=False,
        update_model=lambda _model: None,
    )

    session_resume.restore_session_from_log(session, path)

    assert session.title == "CLI name"
    assert session.title_is_custom is True


# ── Manual names win over auto-titles ────────────────────────────────────


async def test_auto_title_never_overwrites_a_custom_title():
    store = RecordingStore()
    manager = _manager(store)
    agent_session = _agent_session("s1", title="My name", title_is_custom=True)
    manager.sessions["s1"] = agent_session

    effective = await manager.update_session_title("s1", "Auto generated")

    assert effective == "My name"
    assert agent_session.title == "My name"
    assert agent_session.title_is_custom is True
    assert store.updated_fields == []


async def test_auto_title_still_applies_to_untitled_sessions():
    store = RecordingStore()
    manager = _manager(store)
    agent_session = _agent_session("s1")
    manager.sessions["s1"] = agent_session

    effective = await manager.update_session_title("s1", "Auto generated")

    assert effective == "Auto generated"
    assert agent_session.title == "Auto generated"
    assert agent_session.title_is_custom is False
    assert store.updated_fields == [("s1", {"title": "Auto generated"})]


async def test_generate_title_returns_custom_title_without_calling_the_llm(
    monkeypatch,
):
    async def fake_check_session_access(session_id, user):
        return SimpleNamespace(title="My name", title_is_custom=True)

    async def fail_acompletion(**_: Any):
        raise AssertionError("auto-title must not run for a renamed chat")

    async def fail_update_session_title(*_: Any):
        raise AssertionError("auto-title must not persist over a renamed chat")

    monkeypatch.setattr(
        agent_routes, "_check_session_access", fake_check_session_access
    )
    monkeypatch.setattr(agent_routes, "acompletion", fail_acompletion)
    monkeypatch.setattr(
        agent_routes.session_manager, "update_session_title", fail_update_session_title
    )

    response = await agent_routes.generate_title(
        agent_routes.SubmitRequest(session_id="s1", text="hello"),
        {"user_id": "dev"},
    )

    assert response == {"title": "My name"}
