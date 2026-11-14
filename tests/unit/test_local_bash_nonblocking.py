"""The local bash tool must not block the backend's event loop.

In local mode every agent shell command runs inside the web backend process.
A long foreground command (``sleep 14400`` while waiting on a benchmark) used
to freeze every HTTP endpoint — health, chat updates, even Stop — until it
finished, because the handler called ``subprocess.run`` on the event loop.
"""

import asyncio

from agent.tools import local_tools

TICK_SECONDS = 0.05


async def _count_ticks_while(coro) -> tuple[object, int]:
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(TICK_SECONDS)
            ticks += 1

    task = asyncio.create_task(ticker())
    try:
        result = await coro
    finally:
        task.cancel()
    return result, ticks


async def test_bash_command_runs_without_blocking_the_event_loop(tmp_path):
    # Arrange: a command that takes about one second
    args = {"command": "sleep 1; echo done", "work_dir": str(tmp_path)}

    # Act
    (output, ok), ticks = await _count_ticks_while(local_tools._bash_handler(args))

    # Assert: other coroutines kept running while the command slept
    assert ok is True
    assert "done" in output
    assert ticks >= 10


async def test_bash_timeout_still_kills_the_command_without_blocking(tmp_path):
    # Arrange
    args = {"command": "sleep 5", "work_dir": str(tmp_path), "timeout": 1}

    # Act
    (output, ok), ticks = await _count_ticks_while(local_tools._bash_handler(args))

    # Assert
    assert ok is False
    assert "timed out after 1s" in output
    assert ticks >= 10
