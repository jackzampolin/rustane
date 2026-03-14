#!/bin/bash
# stop-loop.sh — Stop the optimization loop
#
# Usage:
#   ./system/stop-loop.sh          # graceful: finish current iteration, then stop
#   ./system/stop-loop.sh --now    # immediate: kill claude + loop process

set -euo pipefail

STOPFILE="/tmp/rustane-opt-STOP"

if [[ "${1:-}" == "--now" ]]; then
    echo "Killing optimization loop immediately..."
    # Kill only the AGENT's claude process by PID file, not interactive sessions
    PIDFILE="/tmp/rustane-claude-alpha.pid"
    if [ -f "$PIDFILE" ]; then
        CPID=$(cat "$PIDFILE")
        kill "$CPID" 2>/dev/null && echo "  Killed agent claude (PID=$CPID)" || echo "  Agent PID $CPID not running"
        rm -f "$PIDFILE"
    else
        echo "  No agent PID file found"
    fi
    # Kill the loop script itself
    pkill -f "optimize-loop.sh" 2>/dev/null && echo "  Killed loop script" || echo "  No loop script found"
    # Also touch stop file in case anything respawns
    touch "$STOPFILE"
    echo "Done."
else
    echo "Requesting graceful stop (will finish current iteration)..."
    touch "$STOPFILE"
    echo "Stop file created: ${STOPFILE}"
    echo "The loop will exit after the current Claude invocation finishes."
    echo ""
    echo "To kill immediately instead: ./system/stop-loop.sh --now"
fi
