#!/bin/bash
# Entrypoint for HF Spaces dev mode compatibility.
# Dev mode spawns CMD multiple times simultaneously on restart.
# Only the first instance can bind port 7860 — the rest must exit
# with code 0 so the dev mode daemon doesn't mark the app as crashed.
#
# IMPORTANT (data safety): we run uvicorn as a child and forward SIGTERM/
# SIGINT to it, then wait for it to exit, so FastAPI's lifespan shutdown
# runs to completion. That shutdown flushes every still-active session to
# session_logs/ (see backend/main.py: _flush_session_on_shutdown). If this
# script exec'd uvicorn or swallowed the signal, `docker compose down` /
# container recreate would SIGTERM *bash*, uvicorn would get SIGKILL'd after
# the grace period, the flush would never run, and in-flight sessions not yet
# caught by the 60s heartbeat would be lost. Keep this forwarding intact.

child=0
_forward_term() {
    if [ "$child" -ne 0 ]; then
        kill -TERM "$child" 2>/dev/null
    fi
}
trap _forward_term SIGTERM SIGINT

# Run uvicorn in the background so the trap above can fire while we wait.
uvicorn main:app --host 0.0.0.0 --port 7860 &
child=$!

# `wait` returns when a trapped signal arrives (after running the trap, which
# forwards SIGTERM to uvicorn) — but uvicorn is still draining its graceful
# shutdown at that point. Re-wait until the child has actually exited so the
# lifespan flush completes before this script returns.
wait "$child"
EXIT_CODE=$?
while kill -0 "$child" 2>/dev/null; do
    wait "$child"
    EXIT_CODE=$?
done

if [ $EXIT_CODE -ne 0 ] && [ $EXIT_CODE -ne 143 ] && [ $EXIT_CODE -ne 130 ]; then
    # Non-signal failure — most commonly a port-in-use conflict when HF Spaces
    # dev mode spawns a second instance. Exit cleanly so the daemon doesn't
    # mark the app as crashed.
    echo "uvicorn exited with code $EXIT_CODE, exiting gracefully."
fi
exit 0
