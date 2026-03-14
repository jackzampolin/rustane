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
    # Kill only the AGENT's claude process (matches on agent ID in prompt), not interactive sessions
    pkill -f "Your agent ID is:" 2>/dev/null && echo "  Killed agent claude process" || echo "  No agent claude process found"
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
