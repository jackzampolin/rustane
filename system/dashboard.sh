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

# Draw a box: box <row> <col> <width> <height> <title> <title_color>
box() {
    local r=$1 c=$2 w=$3 h=$4 title=$5 tcol=${6:-$WHT}
    # Top border
    tput cup $r $c
    printf "${D}+"
    printf '%*s' $((w - 2)) '' | tr ' ' '-'
    printf "+${R}"
    # Title overlay
    if [ -n "$title" ]; then
        tput cup $r $((c + 2))
        printf " ${B}${tcol}%s${R} " "$title"
    fi
    # Sides
    for ((i=1; i<h-1; i++)); do
        tput cup $((r + i)) $c
        printf "${D}|${R}"
        tput cup $((r + i)) $((c + w - 1))
        printf "${D}|${R}"
    done
    # Bottom border
    tput cup $((r + h - 1)) $c
    printf "${D}+"
    printf '%*s' $((w - 2)) '' | tr ' ' '-'
    printf "+${R}"
}

# Write text inside a box: btext <row> <col> <color> <text>
btext() {
    tput cup $1 $2
    printf "%b%s%b" "$3" "$4" "$R"
}

# Clear a line region: bclear <row> <col> <width>
bclear() {
    tput cup $1 $2
    printf "%*s" $3 ""
}

render() {
    local W=$(tput cols)
    local H=$(tput lines)

    # Layout: 2 columns
    local LW=$(( (W - 3) / 2 ))  # left width
    local RW=$(( W - LW - 3 ))   # right width
    local LC=1                     # left col
    local RC=$((LW + 2))          # right col

    tput cup 0 0

    # Header
    printf "${B}${CYN}  RUSTANE${R} ${D}%s | current: ${GRN}%sms${R} ${D}| target: ${YLW}89ms${R}" "$(date '+%H:%M:%S')" "$(get_current_ms)"
    printf "%*s\n" $((W - 55)) ""

    # ═══════════════════════════════════════
    # LEFT COLUMN: Projects + Trajectory
    # ═══════════════════════════════════════

    # Projects box
    local proj_h=3  # base height
    local proj_count=0
    for f in "$PROJECT_DIR"/*.md; do
        [ -f "$f" ] || continue
        # Skip finished projects
        local _pst=$(grep "^status:" "$f" 2>/dev/null | head -1 | cut -d' ' -f2-)
        [[ "$_pst" == "ABANDONED" || "$_pst" == "DONE" ]] && continue
        # Skip projects with dead agents and no progress (abandoned in practice)
        local _pid_f="/tmp/rustane-project-PID-$(basename "$f" .md)"
        local _stat_f="/tmp/rustane-project-status-$(basename "$f" .md)"
        if [ -f "$_stat_f" ] && grep -qi "abandon\|recommend ABANDON" "$_stat_f" 2>/dev/null; then continue; fi
        proj_count=$((proj_count + 1))
        proj_h=$((proj_h + 4))
    done
    [ $proj_count -eq 0 ] && proj_h=4

    box 2 $LC $LW $proj_h "PROJECTS" "$CYN"

    local row=3
    if [ $proj_count -eq 0 ]; then
        bclear $row $((LC + 2)) $((LW - 4))
        btext $row $((LC + 2)) "$D" "no active projects"
    else
        for f in "$PROJECT_DIR"/*.md; do
            [ -f "$f" ] || continue
            local name=$(basename "$f" .md)
            local status=$(grep "^status:" "$f" 2>/dev/null | head -1 | cut -d' ' -f2-)
            [[ "$status" == "ABANDONED" || "$status" == "DONE" ]] && continue
            local _sf="/tmp/rustane-project-status-$name"
            if [ -f "$_sf" ] && grep -qi "abandon" "$_sf" 2>/dev/null; then continue; fi
            local phase=$(cat "/tmp/rustane-project-status-$name" 2>/dev/null | cut -c1-$((LW - 6)) || echo "-")

            local scol="${WHT}"
            case "$status" in
                RESEARCH)       scol="${BLU}" ;;
                PLANNING)       scol="${MAG}" ;;
                IMPLEMENTING*)  scol="${YLW}" ;;
                TESTING|BENCH*) scol="${CYN}" ;;
                REVIEW*|READY*) scol="${GRN}" ;;
                DONE)           scol="${GRN}" ;;
                ABANDONED)      scol="${RED}" ;;
            esac

            # Agent info
            local agent_str="${D}idle${R}"
            local pid_file="/tmp/rustane-project-PID-$name"
            if [ -f "$pid_file" ]; then
                local pid=$(cat "$pid_file")
                if kill -0 "$pid" 2>/dev/null; then
                    local elapsed=$(ps -o etime= -p "$pid" 2>/dev/null | tr -d ' ')
                    agent_str="${GRN}PID $pid ${D}($elapsed)"
                else
                    agent_str="${RED}dead"
                fi
            fi

            bclear $row $((LC + 2)) $((LW - 4))
            btext $row $((LC + 2)) "${B}${scol}" "$(printf '%-18s' "$name")"
            # Agent PID inline
            tput cup $row $((LC + 22))
            printf "%b${R}" "$agent_str"

            # Breadcrumb trail: RSRCH > PLAN > IMPL > TEST > BNCH > REVW
            row=$((row + 1))
            bclear $row $((LC + 2)) $((LW - 4))
            local steps="RESEARCH PLANNING IMPLEMENTING TESTING BENCHMARKING REVIEW"
            local found=false
            local bc="  "
            for step in $steps; do
                local label="${step:0:4}"
                if [ "$found" = "false" ]; then
                    if [ "$step" = "$status" ] || [[ "$status" == "$step"* ]]; then
                        bc+="${B}${scol}[${label}]${R} > "
                        found=true
                    else
                        bc+="${GRN}${label}${R} > "
                    fi
                else
                    bc+="${D}${label}${R} > "
                fi
            done
            bc="${bc% > }"  # remove trailing >
            tput cup $row $((LC + 2))
            printf "%b" "$bc"

            # Phase detail
            row=$((row + 1))
            bclear $row $((LC + 2)) $((LW - 4))
            btext $row $((LC + 2)) "$D" "  $(echo "$phase" | cut -c1-$((LW - 8)))"

            row=$((row + 1))
        done
    fi

    # Trajectory box (below projects)
    local traj_top=$((2 + proj_h + 1))
    local traj_h=6
    box $traj_top $LC $LW $traj_h "TRAJECTORY" "$YLW"

    local ms=$(get_current_ms)
    local target=89
    local start=1260

    bclear $((traj_top + 1)) $((LC + 2)) $((LW - 4))
    tput cup $((traj_top + 1)) $((LC + 2))
    printf "%b1260 > 138 > %b%sms%b > %b89ms%b" "$D" "$B$GRN" "$ms" "$R" "$YLW" "$R"

    # Progress bar toward target
    local bar_w=$((LW - 16))
    local gap=$((start - target))
    local ms_num=${ms:-102}
    local progress=$((start - ms_num))
    local filled=$((progress * bar_w / gap))
    [ $filled -gt $bar_w ] && filled=$bar_w
    [ $filled -lt 0 ] && filled=0

    local bar_full="" bar_empty=""
    for ((j=0; j<filled; j++)); do bar_full+="#"; done
    for ((j=filled; j<bar_w; j++)); do bar_empty+="."; done

    bclear $((traj_top + 2)) $((LC + 2)) $((LW - 4))
    tput cup $((traj_top + 2)) $((LC + 2))
    printf "  %b%s%b%s%b" "$GRN" "$bar_full" "$D" "$bar_empty" "$R"

    bclear $((traj_top + 3)) $((LC + 2)) $((LW - 4))
    local pct=$(( progress * 100 / gap ))
    local remaining=$((ms_num - target))
    tput cup $((traj_top + 3)) $((LC + 2))
    printf "  %b%d%%%b to target  |  %b%dms%b remaining" "$GRN" "$pct" "$R" "$YLW" "$remaining" "$R"

    bclear $((traj_top + 4)) $((LC + 2)) $((LW - 4))

    # ═══════════════════════════════════════
    # RIGHT COLUMN: Hardware + Activity
    # ═══════════════════════════════════════

    # Hardware box
    local hw_h=6
    box 2 $RC $RW $hw_h "HARDWARE" "$GRN"

    bclear 3 $((RC + 2)) $((RW - 4))
    if [ -f "$HW_LOCK" ]; then
        btext 3 $((RC + 2)) "${YLW}" "ANE/Metal: LOCKED ($(cat "$HW_LOCK"))"
    else
        btext 3 $((RC + 2)) "${GRN}" "ANE/Metal: available"
    fi

    bclear 4 $((RC + 2)) $((RW - 4))
    local cargo_n=$(pgrep -f "cargo.*engine" 2>/dev/null | wc -l | tr -d ' ')
    if [ "$cargo_n" -gt 0 ]; then
        btext 4 $((RC + 2)) "${CYN}" "cargo: $cargo_n active"
    else
        btext 4 $((RC + 2)) "$D" "cargo: idle"
    fi

    bclear 5 $((RC + 2)) $((RW - 4))
    local claude_n=$(pgrep -f "claude.*dangerously" 2>/dev/null | grep -v grep | wc -l | tr -d ' ')
    btext 5 $((RC + 2)) "$D" "claude agents: $claude_n"

    bclear 6 $((RC + 2)) $((RW - 4))

    # Activity box (fills remaining right column)
    local act_top=$((2 + hw_h + 1))
    local act_h=$((H - act_top - 2))
    [ $act_h -lt 5 ] && act_h=5

    box $act_top $RC $RW $act_h "ACTIVITY" "$MAG"

    local act_lines=$((act_h - 2))
    local act_row=$((act_top + 1))

    # Read gossip into array (avoids subshell pipe problem)
    local -a glines=()
    if [ -f "$GOSSIP" ]; then
        while IFS= read -r gline; do
            glines+=("$gline")
        done < <(tail -$act_lines "$GOSSIP" 2>/dev/null)
    fi

    for gline in "${glines[@]}"; do
        [ $act_row -ge $((act_top + act_h - 1)) ] && break
        bclear $act_row $((RC + 2)) $((RW - 4))
        local trunc="${gline:0:$((RW - 6))}"
        if echo "$gline" | grep -q "IMPROVED"; then
            btext $act_row $((RC + 2)) "${GRN}" "$trunc"
        elif echo "$gline" | grep -q "TIMEOUT\|ERROR\|WORSE\|REVERTED"; then
            btext $act_row $((RC + 2)) "${RED}" "$trunc"
        elif echo "$gline" | grep -q "CLAIMED\|ITERATION\|RESEARCHING"; then
            btext $act_row $((RC + 2)) "${CYN}" "$trunc"
        else
            btext $act_row $((RC + 2)) "$D" "$trunc"
        fi
        act_row=$((act_row + 1))
    done

    # Clear remaining activity lines
    while [ $act_row -lt $((act_top + act_h - 1)) ]; do
        bclear $act_row $((RC + 2)) $((RW - 4))
        act_row=$((act_row + 1))
    done

    # Footer
    tput cup $((H - 1)) 0
    printf "${D}  q: quit | ./system/dashboard.sh | ./system/project.sh --status${R}%*s" $((W - 65)) ""
}

# Main
if $ONCE; then
    clear
    render
    echo ""
    exit 0
fi

tput civis 2>/dev/null
trap 'tput cnorm 2>/dev/null; tput sgr0; clear; exit 0' EXIT INT TERM

clear
while true; do
    render
    if read -t "$REFRESH" -rsn 1 key 2>/dev/null; then
        [[ "$key" == "q" || "$key" == "Q" ]] && break
    fi
done
