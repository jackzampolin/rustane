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
        # Kill only the AGENT's claude process by PID file, not interactive sessions
        PIDFILE="/tmp/rustane-claude-alpha.pid"
        if [ -f "$PIDFILE" ]; then
            CPID=$(cat "$PIDFILE")
            kill "$CPID" 2>/dev/null && echo "  Killed agent claude (PID=$CPID)" || echo "  Agent PID $CPID not running"
            rm -f "$PIDFILE"
        else
            echo "  No agent PID file found"
        fi
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
        echo "=== Optimization Loop Status ==="
        echo ""
        # Agent status from status file
        STATUSFILE="/tmp/rustane-status-alpha"
        if [ -f "$STATUSFILE" ]; then
            echo "Agent:   $(cat "$STATUSFILE")"
        else
            echo "Agent:   (no status file)"
        fi
        # Process status
        PIDFILE="/tmp/rustane-claude-alpha.pid"
        if [ -f "$PIDFILE" ]; then
            CPID=$(cat "$PIDFILE")
            if kill -0 "$CPID" 2>/dev/null; then
                CLAUDE_TIME=$(ps -o etime= -p "$CPID" 2>/dev/null | tr -d ' ')
                echo "Claude:  RUNNING (PID=$CPID, elapsed=$CLAUDE_TIME)"
            else
                echo "Claude:  DEAD (stale PID=$CPID)"
            fi
        else
            echo "Claude:  idle (no PID file)"
        fi
        if pgrep -qf "optimize-loop.sh"; then
            echo "Loop:    RUNNING"
        else
            echo "Loop:    NOT RUNNING"
        fi
        echo ""
        # Control files
        echo "Controls:"
        [ -f "$STOPFILE" ]   && echo "  STOP:   SET" || echo "  STOP:   clear"
        [ -f "$PAUSEFILE" ]  && echo "  PAUSE:  SET" || echo "  PAUSE:  clear"
        [ -f "$INJECTFILE" ] && echo "  INJECT: SET — $(cat "$INJECTFILE" | head -1)" || echo "  INJECT: clear"
        echo ""
        # Recent gossip
        echo "--- Last 8 gossip entries ---"
        tail -8 "$GOSSIPFILE" 2>/dev/null || echo "(no gossip)"
        ;;

    *)
        echo "Usage: ./system/intercept.sh {pause|resume|stop|kill|inject [msg]|status}"
        exit 1
        ;;
esac
