"""Helpers for the on-disk ``session_logs/`` JSON snapshots.

These files are the durable record when no Mongo store is configured: they
survive backend restarts and back ``GET /api/sessions/archived``. The core
``Session`` writes them (``agent/core/session.py``); this module finds them
and patches chat titles into them in place.
"""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# The backend runs from backend/, so this matches the core Session's default
# relative "session_logs" directory (and the docker-compose bind mount).
SESSION_LOG_DIR = Path(__file__).resolve().parent / "session_logs"

# Session ids are uuids (or similar); anything else could smuggle glob
# metacharacters into the filename lookup.
_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9_-]{1,128}$")
_LOG_TIMESTAMP = r"_\d{8}_\d{6}\.json"
_LOG_FILENAME = re.compile(rf"^session_[A-Za-z0-9_-]{{1,128}}{_LOG_TIMESTAMP}$")


def session_log_dir() -> Path:
    """Directory holding session_logs/ JSON snapshots (patchable in tests)."""
    return SESSION_LOG_DIR


def is_safe_session_id(session_id: str | None) -> bool:
    return bool(session_id) and bool(_SAFE_SESSION_ID.match(session_id))


def find_session_logs(session_id: str) -> list[Path]:
    """Every ``session_<id>_<YYYYMMDD>_<HHMMSS>.json`` for exactly this id.

    Names are matched with an exact regex, never a glob, so ids holding glob
    or path characters match nothing and ``abc`` never picks up ``abc_def``.
    """
    if not is_safe_session_id(session_id):
        return []
    log_dir = session_log_dir()
    if not log_dir.is_dir():
        return []
    name_pattern = re.compile(rf"^session_{re.escape(session_id)}{_LOG_TIMESTAMP}$")
    return [path for path in log_dir.iterdir() if name_pattern.match(path.name)]


def find_latest_session_log(session_id: str) -> Path | None:
    """Return the newest ``session_<id>_<timestamp>.json`` for an exact id."""
    matches = find_session_logs(session_id)
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def resolve_session_log(identifier: str) -> Path | None:
    """The log to import: the newest one for a session id, or an exact
    ``session_<id>_<timestamp>.json`` filename inside the log directory."""
    if _LOG_FILENAME.match(identifier):
        path = session_log_dir() / identifier
        return path if path.is_file() else None
    return find_latest_session_log(identifier)


def load_session_log(path: Path) -> dict[str, Any] | None:
    """Parse a session log; None when it is unreadable or not a JSON object."""
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        logger.warning("Could not read session log %s: %s", path.name, e)
        return None
    return data if isinstance(data, dict) else None


def saved_title_fields(data: dict[str, Any]) -> tuple[str | None, bool]:
    """Extract ``(title, title_is_custom)`` from saved session metadata."""
    title = data.get("title")
    if not isinstance(title, str) or not title.strip():
        return None, False
    return title, bool(data.get("title_is_custom"))


def write_session_log_title(
    path: Path, data: dict[str, Any], *, title: str, is_custom: bool
) -> None:
    """Atomically rewrite ``path`` with only the title fields changed.

    Stages a uniquely named temp file in the same directory (so it never
    collides with the core Session's own ``.tmp`` staging file) and
    ``os.replace``s it over the original. The original mode and mtime are
    kept so a rename doesn't reorder the archived list.
    """
    updated = {**data, "title": title, "title_is_custom": is_custom}
    original = path.stat()
    fd, tmp_name = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".title.tmp"
    )
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(updated, f, indent=2)
        os.chmod(tmp_name, original.st_mode & 0o777)
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    os.utime(path, ns=(original.st_atime_ns, original.st_mtime_ns))


def persist_title_to_latest_log(
    session_id: str, title: str, *, is_custom: bool
) -> bool:
    """Patch the title into a live session's newest log, if it has one.

    Best-effort: the runtime session also writes the title on every later
    auto-save, so a failure here only narrows the durability window.
    Returns True when a file was updated.
    """
    path = find_latest_session_log(session_id)
    if path is None:
        return False
    data = load_session_log(path)
    if data is None:
        return False
    try:
        write_session_log_title(path, data, title=title, is_custom=is_custom)
    except OSError as e:
        logger.warning("Could not write title into %s: %s", path.name, e)
        return False
    return True
