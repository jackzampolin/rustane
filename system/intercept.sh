#!/bin/bash
# intercept.sh — Control the optimization loop between iterations
#
# Usage:
#   ./system/intercept.sh pause           # Pause after current iteration finishes
#   ./system/intercept.sh resume          # Resume a paused loop
#   ./system/intercept.sh stop            # Stop after current iteration (graceful)
#   ./system/intercept.sh kill            # Kill immediately (same as stop-loop.sh --now)
#   ./system/intercept.sh inject "msg"    # Add instructions for next iteration only
#   ./system/intercept.sh status          # Show what's happening
#   ./system/intercept.sh inject          # Open $EDITOR to write instructions
#
# How it works:
#   The loop checks for control files between iterations:
#   - /tmp/rustane-opt-STOP    → stop the loop
#   - /tmp/rustane-opt-PAUSE   → pause until file is removed
#   - /tmp/rustane-opt-INJECT  → append contents to next prompt, then auto-delete

set -euo pipefail

STOPFILE="/tmp/rustane-opt-STOP"
PAUSEFILE="/tmp/rustane-opt-PAUSE"
INJECTFILE="/tmp/rustane-opt-INJECT"
GOSSIPFILE="/tmp/rustane-gossip.md"

ACTION="${1:-status}"

case "$ACTION" in
    pause)
        touch "$PAUSEFILE"
        echo "PAUSED — loop will pause after current iteration finishes."
        echo "  Resume with: ./system/intercept.sh resume"
        echo "  The loop checks every 5s for the pause file to be removed."
        ;;

    resume|unpause)
        rm -f "$PAUSEFILE"
        echo "RESUMED — loop will continue on next check (within 5s)."
        ;;

    stop)
        touch "$STOPFILE"
        echo "STOPPING — loop will exit after current iteration finishes."
        echo "  To kill immediately: ./system/intercept.sh kill"
        ;;

    kill|now)
        echo "Killing optimization loop immediately..."
        pkill -f "claude.*dangerously-skip-permissions" 2>/dev/null && echo "  Killed claude process" || echo "  No claude process found"
        pkill -f "optimize-loop.sh" 2>/dev/null && echo "  Killed loop script" || echo "  No loop script found"
        touch "$STOPFILE"
        rm -f "$PAUSEFILE"
        echo "Done."
        ;;

    inject)
        if [ -n "${2:-}" ]; then
            echo "$2" > "$INJECTFILE"
            echo "INJECTED — next iteration will include:"
            echo "  \"$2\""
            echo "  (auto-deleted after use)"
        else
            # Open editor for multi-line instructions
            TMPFILE=$(mktemp)
            echo "# Write instructions for the next iteration." > "$TMPFILE"
            echo "# Lines starting with # are removed." >> "$TMPFILE"
            echo "# Save and close to inject. Empty file = cancel." >> "$TMPFILE"
            ${EDITOR:-nano} "$TMPFILE"
            # Strip comments and check if anything remains
            CONTENT=$(grep -v '^#' "$TMPFILE" | sed '/^$/d')
            rm -f "$TMPFILE"
            if [ -n "$CONTENT" ]; then
                echo "$CONTENT" > "$INJECTFILE"
                echo "INJECTED — next iteration will include:"
                echo "$CONTENT"
            else
                echo "Cancelled — no instructions injected."
            fi
        fi
        ;;

    status|st)
        echo "=== Optimization Loop Control ==="
        echo ""
        # Process status
        if pgrep -qf "optimize-loop.sh"; then
            echo "Loop:    RUNNING"
        else
            echo "Loop:    NOT RUNNING"
        fi
        if pgrep -qf "claude.*dangerously-skip-permissions"; then
            CLAUDE_PID=$(pgrep -f "claude.*dangerously-skip-permissions" 2>/dev/null | head -1)
            CLAUDE_TIME=$(ps -o etime= -p "$CLAUDE_PID" 2>/dev/null | tr -d ' ')
            echo "Claude:  RUNNING (PID=$CLAUDE_PID, elapsed=$CLAUDE_TIME)"
        else
            echo "Claude:  idle"
        fi
        echo ""
        # Control files
        echo "Controls:"
        [ -f "$STOPFILE" ]   && echo "  STOP:   SET (will stop after current iteration)" || echo "  STOP:   clear"
        [ -f "$PAUSEFILE" ]  && echo "  PAUSE:  SET (loop is paused)" || echo "  PAUSE:  clear"
        if [ -f "$INJECTFILE" ]; then
            echo "  INJECT: SET — contents:"
            sed 's/^/    /' "$INJECTFILE"
        else
            echo "  INJECT: clear"
        fi
        echo ""
        # Recent gossip
        echo "--- Last 10 gossip entries ---"
        tail -10 "$GOSSIPFILE" 2>/dev/null || echo "(no gossip)"
        ;;

    *)
        echo "Usage: ./system/intercept.sh {pause|resume|stop|kill|inject [msg]|status}"
        exit 1
        ;;
esac
