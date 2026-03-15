#!/bin/bash
# dashboard.sh — Live terminal dashboard for rustane optimization system
#
# Usage:
#   ./system/dashboard.sh          # refreshes every 3s
#   ./system/dashboard.sh --once   # single render, no loop
#
# Looks like mactop/jtop but for your AI optimization pipeline.

REFRESH=3
ONCE=false
[[ "${1:-}" == "--once" ]] && ONCE=true

# --- Colors ---
RST="\033[0m"
BOLD="\033[1m"
DIM="\033[2m"
RED="\033[31m"
GREEN="\033[32m"
YELLOW="\033[33m"
BLUE="\033[34m"
MAGENTA="\033[35m"
CYAN="\033[36m"
WHITE="\033[37m"
BG_DARK="\033[48;5;234m"
BG_CARD="\033[48;5;236m"

PROJECT_DIR="/tmp/rustane-projects"
HW_LOCK="/tmp/rustane-hw-lock"
GOSSIP="/tmp/rustane-gossip.md"

render() {
    clear
    local COLS=$(tput cols)
    local NOW=$(date '+%H:%M:%S')

    # ── Header ──
    printf "${BG_DARK}${BOLD}${CYAN}"
    printf "  %-*s" $((COLS - 2)) "RUSTANE OPTIMIZATION DASHBOARD"
    printf "${RST}\n"
    printf "${DIM}  $(date '+%Y-%m-%d %H:%M:%S')  |  target: 89ms  |  current: $(get_current_ms)ms  |  refresh: ${REFRESH}s${RST}\n"
    printf "${DIM}  ─%.0s" $(seq 1 $((COLS - 4)))
    printf "${RST}\n"

    # ── Projects ──
    printf "\n${BOLD}${WHITE}  PROJECTS${RST}\n\n"

    local has_projects=false
    for f in "$PROJECT_DIR"/*.md; do
        [ -f "$f" ] || continue
        has_projects=true
        local name=$(basename "$f" .md)
        local status=$(grep "^status:" "$f" 2>/dev/null | head -1 | cut -d' ' -f2-)
        local branch=$(grep "^branch:" "$f" 2>/dev/null | head -1 | cut -d' ' -f2-)
        local desc=$(grep "^description:" "$f" 2>/dev/null | head -1 | cut -d' ' -f2-)

        # Status color
        local scol="${WHITE}"
        case "$status" in
            RESEARCH)       scol="${BLUE}" ;;
            PLANNING)       scol="${MAGENTA}" ;;
            IMPLEMENTING*)  scol="${YELLOW}" ;;
            TESTING)        scol="${CYAN}" ;;
            BENCHMARKING)   scol="${CYAN}" ;;
            REVIEW)         scol="${MAGENTA}" ;;
            READY_TO_MERGE) scol="${GREEN}" ;;
            DONE)           scol="${GREEN}" ;;
            ABANDONED)      scol="${RED}" ;;
        esac

        # Agent info
        local pid_file="/tmp/rustane-project-PID-$name"
        local status_file="/tmp/rustane-project-status-$name"
        local agent_status="idle"
        local agent_elapsed=""
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                agent_elapsed=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
                agent_status="running"
            else
                agent_status="dead"
            fi
        fi

        local phase=$(cat "$status_file" 2>/dev/null || echo "-")

        # Render project card
        printf "  ${BG_CARD}${BOLD}${scol} %-20s ${RST}" "$name"
        printf "${BG_CARD} ${scol}%-15s${RST}" "$status"
        if [ "$agent_status" = "running" ]; then
            printf "${BG_CARD} ${GREEN}● ${WHITE}PID %-6s ${DIM}%s${RST}" "$pid" "$agent_elapsed"
        elif [ "$agent_status" = "dead" ]; then
            printf "${BG_CARD} ${RED}✕ dead${RST}              "
        else
            printf "${BG_CARD} ${DIM}○ idle${RST}              "
        fi
        printf "\n"
        printf "  ${DIM}  branch: %-30s phase: %s${RST}\n" "$branch" "$phase"

        # Steps progress
        local done total
        done=$(grep -cE '^\d+\. \[x\]' "$f" 2>/dev/null) || done=0
        total=$(grep -cE '^\d+\. \[' "$f" 2>/dev/null) || total=0
        if [ "$total" -gt 0 ] 2>/dev/null; then
            printf "  ${DIM}  progress: ${GREEN}"
            for ((j=0; j<done; j++)); do printf '█'; done
            printf "${DIM}"
            for ((j=done; j<total; j++)); do printf '░'; done
            printf " ${WHITE}%d/%d${RST}\n" "$done" "$total"
        fi

        # Pause/inject indicators
        [ -f "/tmp/rustane-project-PAUSE-$name" ] && printf "  ${YELLOW}  ⏸  PAUSED${RST}\n"
        [ -f "/tmp/rustane-project-INJECT-$name" ] && printf "  ${BLUE}  💉 INJECT queued${RST}\n"

        printf "\n"
    done

    if ! $has_projects; then
        printf "  ${DIM}No active projects. Create one with:${RST}\n"
        printf "  ${DIM}./system/project.sh --new <name> --desc <description>${RST}\n\n"
    fi

    # ── Old Loop (if running) ──
    local loop_status="/tmp/rustane-status-alpha"
    if [ -f "$loop_status" ]; then
        printf "${BOLD}${WHITE}  OPTIMIZATION LOOP${RST}\n\n"
        local lstatus=$(cat "$loop_status" 2>/dev/null || echo "stopped")
        if tmux has-session -t rustane 2>/dev/null; then
            printf "  ${GREEN}● ${WHITE}%-20s ${DIM}%s${RST}\n" "loop: RUNNING" "$lstatus"
        else
            printf "  ${DIM}○ loop: STOPPED ${RST}\n"
        fi
        printf "\n"
    fi

    # ── Hardware ──
    printf "${BOLD}${WHITE}  HARDWARE${RST}\n\n"
    if [ -f "$HW_LOCK" ]; then
        local holder=$(cat "$HW_LOCK" 2>/dev/null || echo "unknown")
        printf "  ${YELLOW}● ANE/Metal LOCKED by: %s${RST}\n" "$holder"
    else
        printf "  ${GREEN}● ANE/Metal: available${RST}\n"
    fi

    # Check for active cargo/rustc
    local cargo_count=$(pgrep -f "cargo.*engine" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$cargo_count" -gt 0 ]; then
        printf "  ${CYAN}● cargo: %s process(es) active${RST}\n" "$cargo_count"
    fi
    printf "\n"

    # ── Performance ──
    printf "${BOLD}${WHITE}  PERFORMANCE TRAJECTORY${RST}\n\n"
    printf "  ${DIM}1260 ─────────────────────────────── start${RST}\n"
    printf "  ${DIM} 138 ──────── async+staging wins${RST}\n"
    printf "  ${GREEN} 102 ──── opus wins (current auto-max)${RST}\n"
    printf "  ${YELLOW}  89 ···· target (maderix Obj-C)${RST}\n"
    printf "\n"

    # ── Recent Gossip ──
    printf "${BOLD}${WHITE}  RECENT ACTIVITY${RST}\n\n"
    if [ -f "$GOSSIP" ]; then
        tail -6 "$GOSSIP" 2>/dev/null | while IFS= read -r line; do
            if echo "$line" | grep -q "IMPROVED"; then
                printf "  ${GREEN}%s${RST}\n" "$line"
            elif echo "$line" | grep -q "TIMEOUT\|ERROR\|WORSE"; then
                printf "  ${RED}%s${RST}\n" "$line"
            elif echo "$line" | grep -q "CLAIMED\|ITERATION"; then
                printf "  ${CYAN}%s${RST}\n" "$line"
            else
                printf "  ${DIM}%s${RST}\n" "$line"
            fi
        done
    else
        printf "  ${DIM}(no gossip)${RST}\n"
    fi
    printf "\n"

    # ── Controls ──
    printf "${DIM}  ─%.0s" $(seq 1 $((COLS - 4)))
    printf "${RST}\n"
    printf "${DIM}  q: quit  |  project --status  |  intercept.sh pause/inject/kill${RST}\n"
}

get_current_ms() {
    # Find the latest ms/step from any experiments.tsv
    local ms=""
    for tsv in /tmp/rustane-worktree-*/system/experiments.tsv /tmp/rustane-opt-*/system/experiments.tsv /Users/dan/Dev/rustane/system/experiments.tsv; do
        [ -f "$tsv" ] || continue
        local val=$(tail -20 "$tsv" 2>/dev/null | grep -v '^-' | grep -v 'PLANNED' | awk -F'\t' '$7 ~ /^[0-9]+$/ { ms=$7 } END { print ms }')
        if [ -n "$val" ]; then
            ms="$val"
        fi
    done
    echo "${ms:-???}"
}

# ── Main ──
if $ONCE; then
    render
    exit 0
fi

# Interactive loop with quit on 'q'
stty -echo 2>/dev/null || true
trap 'stty echo 2>/dev/null; exit 0' EXIT INT TERM

while true; do
    render
    # Non-blocking read with timeout
    if read -t "$REFRESH" -n 1 key 2>/dev/null; then
        case "$key" in
            q|Q) break ;;
        esac
    fi
done
