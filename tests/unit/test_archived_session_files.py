"""Archived-session routes must only ever touch one session's own log files.

The delete route used to glob ``session_{session_id}_*.json`` with the raw
URL value, so ``*`` deleted every saved chat; the import route also fell back
to ``glob(id)`` and ``glob(*id*)``, loading whatever file matched.
"""

from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import quote

import pytest

_BACKEND_DIR = Path(__file__).resolve().parent.parent.parent / "backend"
if str(_BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(_BACKEND_DIR))

import session_log_files  # noqa: E402
from test_session_rename import _client, _write_log  # noqa: E402

UNSAFE_IDS = ["*", "a?b", "[ab]", "../abc", "abc*", "abc/def", ".."]


@pytest.fixture
def log_dir(tmp_path, monkeypatch) -> Path:
    directory = tmp_path / "session_logs"
    directory.mkdir()
    monkeypatch.setattr(session_log_files, "SESSION_LOG_DIR", directory)
    return directory


@pytest.fixture
def logs(log_dir):
    """Two logs for session "abc" plus neighbours whose names share its prefix."""
    return {
        "abc_old": _write_log(log_dir, "abc", "20261101_100000", mtime=1_000),
        "abc_new": _write_log(log_dir, "abc", "20261102_100000", mtime=2_000),
        "abcd": _write_log(log_dir, "abcd", "20261101_100000"),
        "abc_def": _write_log(log_dir, "abc_def", "20261101_100000"),
    }


def _surviving(directory) -> set[str]:
    return {path.name for path in directory.iterdir()}


# ── file lookup helpers ────────────────────────────────────────────────


def test_find_session_logs_returns_only_that_sessions_files(logs):
    found = session_log_files.find_session_logs("abc")

    assert sorted(path.name for path in found) == sorted(
        [logs["abc_old"].name, logs["abc_new"].name]
    )


@pytest.mark.parametrize("session_id", UNSAFE_IDS + [""])
def test_find_session_logs_rejects_glob_and_path_characters(logs, session_id):
    assert session_log_files.find_session_logs(session_id) == []


def test_resolve_session_log_accepts_an_id_or_an_exact_log_filename(logs):
    resolve = session_log_files.resolve_session_log

    assert resolve("abc") == logs["abc_new"]
    assert resolve(logs["abc_old"].name) == logs["abc_old"]
    assert resolve("session_missing_20261101_100000.json") is None
    assert resolve("notes.json") is None
    assert resolve("*") is None


# ── DELETE /api/sessions/archived/{session_id} ─────────────────────────


@pytest.mark.parametrize("session_id", UNSAFE_IDS)
def test_delete_with_unsafe_id_is_rejected_and_deletes_nothing(
    logs,
    log_dir,
    session_id,
):
    before = _surviving(log_dir)

    response = _client().delete(f"/api/sessions/archived/{quote(session_id, safe='')}")

    # ".." is normalised away by the URL layer (405 on /api/sessions); every
    # other unsafe id reaches the route and gets a 400.
    assert 400 <= response.status_code < 500
    assert _surviving(log_dir) == before


def test_delete_removes_only_the_requested_sessions_files(logs, log_dir):
    response = _client().delete("/api/sessions/archived/abc")

    assert response.status_code == 200
    assert sorted(response.json()["deleted"]) == sorted(
        [logs["abc_old"].name, logs["abc_new"].name]
    )
    assert _surviving(log_dir) == {logs["abcd"].name, logs["abc_def"].name}


def test_delete_of_unknown_session_is_a_no_op(logs, log_dir):
    before = _surviving(log_dir)

    response = _client().delete("/api/sessions/archived/nosuchsession")

    assert response.status_code == 200
    assert response.json() == {"deleted": []}
    assert _surviving(log_dir) == before


# ── POST /api/session/import ───────────────────────────────────────────


@pytest.mark.parametrize("session_id", UNSAFE_IDS)
def test_import_with_unsafe_id_is_rejected(logs, session_id):
    response = _client().post("/api/session/import", json={"session_id": session_id})

    assert response.status_code in (400, 404)


def test_import_never_falls_back_to_a_substring_match(log_dir):
    # Only "abcd" exists; the old *abc* fallback would have loaded it.
    _write_log(log_dir, "abcd", "20261101_100000")

    response = _client().post("/api/session/import", json={"session_id": "abc"})

    assert response.status_code == 404
