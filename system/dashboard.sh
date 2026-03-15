#!/bin/bash
# dashboard.sh — Live terminal dashboard for rustane optimization system
#
# Usage:
#   ./system/dashboard.sh          # live, refreshes every 3s
#   ./system/dashboard.sh --once   # single render

REFRESH=3
ONCE=false
[[ "${1:-}" == "--once" ]] && ONCE=true

PROJECT_DIR="/tmp/rustane-projects"
HW_LOCK="/tmp/rustane-hw-lock"
GOSSIP="/tmp/rustane-gossip.md"

# Colors
R="\033[0m" B="\033[1m" D="\033[2m"
RED="\033[31m" GRN="\033[32m" YLW="\033[33m"
BLU="\033[34m" MAG="\033[35m" CYN="\033[36m" WHT="\033[37m"

get_current_ms() {
    local ms=""
    for tsv in /Users/dan/Dev/rustane/system/experiments.tsv /tmp/rustane-worktree-*/system/experiments.tsv /tmp/rustane-opt-*/system/experiments.tsv; do
        [ -f "$tsv" ] || continue
        local val=$(tail -20 "$tsv" 2>/dev/null | grep -v '^-' | grep -v 'PLANNED' | awk -F'\t' '$7 ~ /^[0-9]+$/ { ms=$7 } END { print ms }')
        [ -n "$val" ] && ms="$val"
    done
    echo "${ms:-???}"
}

render() {
    local W=$(tput cols)
    local H=$(tput lines)
    local line=0
    local sep=$(printf '%*s' $((W - 4)) '' | tr ' ' '-')

    # Move cursor to top, don't clear (no flicker)
    tput cup 0 0

    # Header
    printf "${B}${CYN}  RUSTANE OPTIMIZATION DASHBOARD${R}%*s\n" $((W - 33)) ""
    printf "${D}  %s  |  target: 89ms  |  current: %sms${R}%*s\n" "$(date '+%H:%M:%S')" "$(get_current_ms)" $((W - 50)) ""
    printf "${D}  %s${R}\n" "$sep"
    line=3

    # Projects
    printf "\n${B}${WHT}  PROJECTS${R}%*s\n\n" $((W - 12)) ""
    line=$((line + 3))

    local has_projects=false
    for f in "$PROJECT_DIR"/*.md; do
        [ -f "$f" ] || continue
        [ $line -ge $((H - 15)) ] && break  # leave room for other sections
        has_projects=true
        local name=$(basename "$f" .md)
        local status=$(grep "^status:" "$f" 2>/dev/null | head -1 | cut -d' ' -f2-)
        local branch=$(grep "^branch:" "$f" 2>/dev/null | head -1 | cut -d' ' -f2-)

        local scol="${WHT}"
        case "$status" in
            RESEARCH)       scol="${BLU}" ;;
            PLANNING)       scol="${MAG}" ;;
            IMPLEMENTING*)  scol="${YLW}" ;;
            TESTING)        scol="${CYN}" ;;
            BENCHMARKING)   scol="${CYN}" ;;
            REVIEW*)        scol="${MAG}" ;;
            READY*|DONE)    scol="${GRN}" ;;
            ABANDONED)      scol="${RED}" ;;
        esac

        # Agent info
        local pid_file="/tmp/rustane-project-PID-$name"
        local status_file="/tmp/rustane-project-status-$name"
        local agent_info="${D}idle${R}"
        if [ -f "$pid_file" ]; then
            local pid=$(cat "$pid_file")
            if kill -0 "$pid" 2>/dev/null; then
                local elapsed=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
                agent_info="${GRN}PID $pid ${D}($elapsed)${R}"
            else
                agent_info="${RED}dead${R}"
            fi
        fi

        local phase=$(cat "$status_file" 2>/dev/null | cut -c1-$((W - 40)) || echo "-")

        printf "  ${B}${scol}%-18s${R}  %-14s  %b%*s\n" "$name" "$status" "$agent_info" 10 ""
        printf "  ${D}%-*s${R}\n" $((W - 4)) "  $branch | $phase"
        line=$((line + 2))

        # Progress
        local done total
        done=$(grep -cE '^\d+\. \[x\]' "$f" 2>/dev/null) || done=0
        total=$(grep -cE '^\d+\. \[' "$f" 2>/dev/null) || total=0
        if [ "$total" -gt 0 ] 2>/dev/null; then
            local bar=""
            for ((j=0; j<done; j++)); do bar+="#"; done
            for ((j=done; j<total; j++)); do bar+="."; done
            printf "  ${D}  [${GRN}%s${D}] %d/%d${R}%*s\n" "$bar" "$done" "$total" $((W - ${#bar} - 14)) ""
            line=$((line + 1))
        fi

        [ -f "/tmp/rustane-project-PAUSE-$name" ] && { printf "  ${YLW}  PAUSED${R}%*s\n" $((W - 12)) ""; line=$((line + 1)); }
        printf "%*s\n" "$W" ""
        line=$((line + 1))
    done

    if ! $has_projects; then
        printf "  ${D}No projects. ./system/project.sh --new <name> --desc <desc>${R}%*s\n\n" 10 ""
        line=$((line + 2))
    fi

    # Hardware
    printf "${D}  %s${R}\n" "$sep"
    printf "${B}${WHT}  HARDWARE${R}%*s\n" $((W - 12)) ""
    line=$((line + 2))

    if [ -f "$HW_LOCK" ]; then
        printf "  ${YLW}LOCKED by: %s${R}%*s\n" "$(cat "$HW_LOCK")" $((W - 25)) ""
    else
        printf "  ${GRN}ANE/Metal: available${R}%*s\n" $((W - 24)) ""
    fi

    local cargo_count=$(pgrep -f "cargo.*engine" 2>/dev/null | wc -l | tr -d ' ')
    [ "$cargo_count" -gt 0 ] && printf "  ${CYN}cargo: %s active${R}%*s\n" "$cargo_count" $((W - 20)) ""
    printf "%*s\n" "$W" ""
    line=$((line + 3))

    # Performance
    printf "${D}  %s${R}\n" "$sep"
    printf "${B}${WHT}  TRAJECTORY${R}  ${D}1260 -> 138 -> 102 -> ${GRN}$(get_current_ms)ms${R}  ${D}(target ${YLW}89${D})${R}%*s\n\n" 10 ""
    line=$((line + 3))

    # Gossip (fill remaining space)
    local gossip_lines=$((H - line - 3))
    [ $gossip_lines -lt 3 ] && gossip_lines=3
    [ $gossip_lines -gt 10 ] && gossip_lines=10

    printf "${D}  %s${R}\n" "$sep"
    printf "${B}${WHT}  RECENT ACTIVITY${R}%*s\n" $((W - 19)) ""
    line=$((line + 2))

    if [ -f "$GOSSIP" ]; then
        tail -$gossip_lines "$GOSSIP" 2>/dev/null | while IFS= read -r gline; do
            local truncated="${gline:0:$((W - 4))}"
            if echo "$gline" | grep -q "IMPROVED"; then
                printf "  ${GRN}%s${R}%*s\n" "$truncated" $((W - ${#truncated} - 4)) ""
            elif echo "$gline" | grep -q "TIMEOUT\|ERROR\|WORSE\|REVERTED"; then
                printf "  ${RED}%s${R}%*s\n" "$truncated" $((W - ${#truncated} - 4)) ""
            elif echo "$gline" | grep -q "CLAIMED\|ITERATION"; then
                printf "  ${CYN}%s${R}%*s\n" "$truncated" $((W - ${#truncated} - 4)) ""
            else
                printf "  ${D}%s${R}%*s\n" "$truncated" $((W - ${#truncated} - 4)) ""
            fi
        done
    fi

    # Footer
    tput cup $((H - 1)) 0
    printf "${D}  q: quit  |  ./system/project.sh --status  |  ./system/dashboard.sh --once${R}%*s" $((W - 76)) ""

    # Clear any leftover lines below content
    tput el
}

# Main
if $ONCE; then
    render
    echo ""
    exit 0
fi

# Hide cursor, restore on exit
tput civis 2>/dev/null
trap 'tput cnorm 2>/dev/null; tput sgr0; exit 0' EXIT INT TERM

# Initial clear, then overwrite in place
clear

while true; do
    render
    if read -t "$REFRESH" -rsn 1 key 2>/dev/null; then
        [[ "$key" == "q" || "$key" == "Q" ]] && break
    fi
done
