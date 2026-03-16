#!/bin/bash
# phase-runner.sh — Sequential phase state machine
#
# Usage:
#   ./system/phase-runner.sh                    # run current phase
#   ./system/phase-runner.sh --status           # show state
#   ./system/phase-runner.sh --phase 1          # force start phase 1
#   ./system/phase-runner.sh --substep IMPLEMENT # force substep
#
# Substep sequence per phase:
#   REFERENCE → RESEARCH → PLAN → IMPLEMENT → TEST → GATE
#   (GATE checks correctness + performance, blocks advancement)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STATE_FILE="/tmp/rustane-phase-state.env"
STATUS_FILE="/tmp/rustane-phase-status"
LOG_FILE="/tmp/rustane-phase-runner.log"
PAUSE_FILE="/tmp/rustane-phase-PAUSE"
INJECT_FILE="/tmp/rustane-phase-INJECT"
GOSSIP_FILE="/tmp/rustane-gossip.md"
HEARTBEAT_FILE="/tmp/rustane-phase-heartbeat"
ALERT_FILE="/tmp/rustane-phase-ALERT"
MODEL="claude-opus-4-6"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

gossip() { echo "[$(date '+%H:%M:%S')] [phase-runner] $*" >> "$GOSSIP_FILE"; }

heartbeat() {
    # Write heartbeat with current state so monitors can detect stalls
    cat > "$HEARTBEAT_FILE" << HEOF
phase=$PHASE
substep=$SUBSTEP
agent_status=$(cat "$STATUS_FILE" 2>/dev/null || echo "unknown")
agent_pid=$(cat /tmp/rustane-phase-agent.pid 2>/dev/null || echo "0")
timestamp=$(date +%s)
human_time=$(date '+%H:%M:%S')
sessions=$SESSIONS
retries=$RETRIES
HEOF
}

alert() {
    # Write alert file for monitors (cron/dashboard can read this)
    local level=$1
    shift
    echo "[$(date '+%H:%M:%S')] [$level] $*" >> "$ALERT_FILE"
    gossip "ALERT ($level): $*"
    log "ALERT ($level): $*"
}

check_inject() {
    # Check for injected instructions from operator
    if [ -f "$INJECT_FILE" ]; then
        local inject_content=$(cat "$INJECT_FILE")
        rm -f "$INJECT_FILE"
        log "INJECT: loaded operator instructions"
        gossip "INJECT: $(echo "$inject_content" | head -1 | cut -c1-80)"
        echo "$inject_content"
    fi
}

# Heartbeat watchdog: runs in background, writes heartbeat every 30s
start_heartbeat_loop() {
    (
        while true; do
            heartbeat
            sleep 30
        done
    ) &
    HEARTBEAT_PID=$!
}

stop_heartbeat_loop() {
    if [ -n "${HEARTBEAT_PID:-}" ]; then
        kill "$HEARTBEAT_PID" 2>/dev/null || true
        wait "$HEARTBEAT_PID" 2>/dev/null || true
    fi
}

trap 'stop_heartbeat_loop; log "Phase runner exiting"' EXIT

# --- State Management (simple env file, not TOML for now) ---
load_state() {
    if [ -f "$STATE_FILE" ]; then
        source "$STATE_FILE"
    else
        PHASE=1
        SUBSTEP="REFERENCE"
        SESSIONS=0
        RETRIES=0
        BASELINE_MS=195
    fi
}

save_state() {
    cat > "$STATE_FILE" << EOF
PHASE=$PHASE
SUBSTEP=$SUBSTEP
SESSIONS=$SESSIONS
RETRIES=$RETRIES
BASELINE_MS=$BASELINE_MS
EOF
}

# --- Parse Args ---
SHOW_STATUS=false
FORCE_PHASE=""
FORCE_SUBSTEP=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --status)   SHOW_STATUS=true; shift ;;
        --phase)    FORCE_PHASE="$2"; shift 2 ;;
        --substep)  FORCE_SUBSTEP="$2"; shift 2 ;;
        --model)    MODEL="$2"; shift 2 ;;
        *)          echo "Unknown: $1"; exit 1 ;;
    esac
done

load_state

if [ -n "$FORCE_PHASE" ]; then PHASE=$FORCE_PHASE; fi
if [ -n "$FORCE_SUBSTEP" ]; then SUBSTEP=$FORCE_SUBSTEP; fi

# --- Status ---
if $SHOW_STATUS; then
    echo "=== Phase Runner Status ==="
    echo "Phase:    $PHASE"
    echo "Substep:  $SUBSTEP"
    echo "Sessions: $SESSIONS"
    echo "Retries:  $RETRIES"
    echo "Baseline: ${BASELINE_MS}ms"
    echo ""
    echo "Agent: $(cat "$STATUS_FILE" 2>/dev/null || echo 'not running')"
    echo ""
    echo "Phase file: system/phases/phase${PHASE}.toml"
    exit 0
fi

save_state

# --- Substep Definitions ---
SUBSTEPS=("REFERENCE" "RESEARCH" "PLAN" "IMPLEMENT" "TEST" "GATE")

next_substep() {
    local current=$1
    local found=false
    for s in "${SUBSTEPS[@]}"; do
        if $found; then echo "$s"; return; fi
        if [ "$s" = "$current" ]; then found=true; fi
    done
    echo "DONE"
}

# --- Build Prompt for Each Substep ---
build_prompt() {
    local phase=$1
    local substep=$2
    local phase_file="$REPO_ROOT/system/phases/phase${phase}.toml"

    # Read phase description
    local desc=$(grep "^description" "$phase_file" 2>/dev/null | cut -d'"' -f2)

    local prompt="You are working on Phase $phase of the rustane ANE training engine optimization.

PHASE $phase: $desc

"

    case "$substep" in
        REFERENCE)
            prompt+="YOUR ROLE: Reference Comparison Agent (CURATOR)
You must study reference implementations and compare against our code.

READ THESE REFERENCE FILES:"
            # Extract reference files from phase TOML
            grep -A20 '\[reference\]' "$phase_file" | grep '\"/' | sed 's/.*\"\(.*\)\".*/\1/' | while read f; do
                prompt+="
  - $f"
            done
            prompt+="

READ OUR CODE:"
            grep -A20 '\[context\]' "$phase_file" | grep 'our_code' -A10 | grep '\"' | sed 's/.*\"\(.*\)\".*/\1/' | while read f; do
                prompt+="
  - $f"
            done
            prompt+="

ALSO READ: AGENTS.md (hardware facts and dead ends)

FOR EACH COMPARISON AREA, write:
  ## Area Name
  ### Reference approach (with specific code/line references)
  ### Our approach (with specific code/line references)
  ### Gap analysis (what to change and why)
  ### Theoretical minimum (hardware-limited floor with math)

Write your analysis to /tmp/rustane-phase${phase}-reference.md
Update status: echo 'REFERENCE: <area>' > $STATUS_FILE

When done, update status: echo 'REFERENCE: COMPLETE' > $STATUS_FILE"
            ;;

        RESEARCH)
            prompt+="YOUR ROLE: Executor Agent
Read the reference comparison at /tmp/rustane-phase${phase}-reference.md
Read AGENTS.md for hardware facts and dead ends.
Read the relevant source code listed in the reference comparison.

Understand what needs to change. Do NOT write code yet.
Write your understanding and approach to /tmp/rustane-phase${phase}-research.md

Update status: echo 'RESEARCH: <what>' > $STATUS_FILE"
            ;;

        PLAN)
            prompt+="YOUR ROLE: Executor Agent
Read /tmp/rustane-phase${phase}-reference.md (reference comparison)
Read /tmp/rustane-phase${phase}-research.md (your research)

Write a detailed implementation plan with:
- Exact files to modify
- Exact functions to change
- Step-by-step changes (each step < 30min)
- What to test after each step

Write plan to /tmp/rustane-phase${phase}-plan.md

Update status: echo 'PLANNING: <what>' > $STATUS_FILE"
            ;;

        IMPLEMENT)
            prompt+="YOUR ROLE: Executor Agent
Read /tmp/rustane-phase${phase}-plan.md (your implementation plan)
Read AGENTS.md for constraints.

IMPLEMENT the plan step by step.
- Commit after each sub-step (incremental, not one big commit)
- Push to origin after each commit
- Run cargo build after each change to verify it compiles
- Update status at each phase:
    echo 'CODING: <file>:<function>' > $STATUS_FILE
    echo 'COMPILING' > $STATUS_FILE
    echo 'TESTING: <test>' > $STATUS_FILE

IMPORTANT: Use bench_training_step_1024 for benchmarks (not 768).
Do NOT modify any gate test files.

When done implementing, update status: echo 'IMPLEMENT: COMPLETE' > $STATUS_FILE"
            ;;

        TEST)
            prompt+="YOUR ROLE: Test Agent
Run ALL required tests:
  cargo test -p engine --release
  cargo test -p engine --test phase3_kernels --release
  cargo test -p engine --test phase4_training --release

If any test FAILS: report which test and why, then fix the code.
Do NOT modify test assertions — fix the implementation.

Run benchmark:
  cargo test -p engine --test bench_step_time --release -- bench_training_step_1024 --ignored --nocapture
  Run 3 times, report median of steps 2-4.

Report: test results + benchmark numbers.
Update status: echo 'TESTING: <suite>' > $STATUS_FILE
When done: echo 'TEST: COMPLETE <ms/step>' > $STATUS_FILE"
            ;;

        GATE)
            # Gate is NOT an agent — it's a script
            log "Running gate check for phase $phase..."
            echo "GATE: running" > "$STATUS_FILE"

            # Run all required tests
            local all_pass=true
            cargo test -p engine --release 2>&1 | tee -a "$LOG_FILE" || all_pass=false
            cargo test -p engine --test phase3_kernels --release 2>&1 | tee -a "$LOG_FILE" || all_pass=false
            cargo test -p engine --test phase4_training --release 2>&1 | tee -a "$LOG_FILE" || all_pass=false

            if ! $all_pass; then
                log "GATE FAIL: tests failed"
                echo "GATE: FAIL (tests)" > "$STATUS_FILE"
                gossip "GATE FAIL: Phase $phase tests failed (retry $((RETRIES+1))/3)"
                RETRIES=$((RETRIES + 1))
                if [ $RETRIES -ge 3 ]; then
                    alert "ESCALATE" "3 gate failures on Phase $phase. Human review needed."
                    touch "$PAUSE_FILE"
                    SUBSTEP="IMPLEMENT"
                else
                    alert "WARN" "Gate retry $RETRIES/3 on Phase $phase"
                    SUBSTEP="IMPLEMENT"
                fi
                save_state
                return 1
            fi

            # Run benchmark
            local bench_out=$(cargo test -p engine --test bench_step_time --release -- bench_training_step_1024 --ignored --nocapture 2>&1)
            local avg_ms=$(echo "$bench_out" | grep -E "^[2-4]" | awk '{sum+=$2; n++} END {printf "%.0f", sum/n}')

            log "GATE: tests PASS, bench ${avg_ms}ms (baseline ${BASELINE_MS}ms)"
            echo "GATE: PASS ${avg_ms}ms" > "$STATUS_FILE"

            # Check improvement
            local min_improvement=$(grep "min_improvement_pct" "$phase_file" | grep -o '[0-9]*')
            local threshold=$((BASELINE_MS * (100 - min_improvement) / 100))

            if [ "$avg_ms" -gt "$threshold" ]; then
                log "GATE FAIL: ${avg_ms}ms > ${threshold}ms (need ${min_improvement}% improvement)"
                RETRIES=$((RETRIES + 1))
                SUBSTEP="IMPLEMENT"
                save_state
                return 1
            fi

            log "GATE PASS: ${avg_ms}ms <= ${threshold}ms"
            git tag "phase${phase}-pass" HEAD 2>/dev/null || true

            # Advance to next phase
            PHASE=$((PHASE + 1))
            SUBSTEP="REFERENCE"
            RETRIES=0
            BASELINE_MS=$avg_ms
            save_state

            log "Phase $((PHASE - 1)) complete. Pausing before Phase $PHASE."
            touch "$PAUSE_FILE"
            return 0
            ;;
    esac

    # For non-GATE substeps: run an agent
    echo "$substep: starting" > "$STATUS_FILE"
    log "Running $substep agent for phase $phase..."

    local worktree="/tmp/rustane-worktree-phase${phase}"
    local phase_branch="phase${phase}/work"
    cd "$REPO_ROOT"
    git fetch origin --quiet
    # Reuse existing worktree within same phase (preserves uncommitted work)
    if [ ! -d "$worktree" ]; then
        # Create phase-specific branch from v2/ane-training
        git branch -f "$phase_branch" v2/ane-training 2>/dev/null || git branch "$phase_branch" v2/ane-training 2>/dev/null || true
        git worktree add "$worktree" "$phase_branch" 2>/dev/null || {
            git worktree remove --force "$worktree" 2>/dev/null || rm -rf "$worktree"
            git branch -f "$phase_branch" v2/ane-training 2>/dev/null || true
            git worktree add "$worktree" "$phase_branch"
        }
    else
        # Pull latest into existing worktree
        cd "$worktree" && git merge origin/v2/ane-training --no-edit --quiet 2>/dev/null || true
        cd "$REPO_ROOT"
    fi

    # Copy dev/ context
    if [ -d "${REPO_ROOT}/dev" ]; then
        mkdir -p "${worktree}/dev"
        cp -a "${REPO_ROOT}/dev/plans" "${worktree}/dev/plans" 2>/dev/null || true
        cp -a "${REPO_ROOT}/dev/info" "${worktree}/dev/info" 2>/dev/null || true
        for f in CURRENT.md METHODOLOGY.md; do
            [ -f "${REPO_ROOT}/dev/$f" ] && cp "${REPO_ROOT}/dev/$f" "${worktree}/dev/$f"
        done
    fi

    cd "$worktree"

    # Append inject if present
    if [ -n "${INJECT_EXTRA:-}" ]; then
        prompt+="$INJECT_EXTRA"
    fi

    # Run agent with timeout
    local timeout=10800  # 3 hours
    if [ "$substep" = "REFERENCE" ] || [ "$substep" = "RESEARCH" ] || [ "$substep" = "PLAN" ]; then
        timeout=1800  # 30 min for research/planning
    fi

    # Run agent in background, log to file directly (NO FIFO — FIFO blocks watchdog)
    local agent_log="/tmp/rustane-phase-agent-output.log"
    : > "$agent_log"  # truncate

    claude -p \
        --dangerously-skip-permissions \
        --model "$MODEL" \
        --effort high \
        "$prompt" >> "$agent_log" 2>&1 &
    local agent_pid=$!
    echo "$agent_pid" > "/tmp/rustane-phase-agent.pid"
    log "Agent PID=$agent_pid (timeout: ${timeout}s, substep: $substep)"

    # Watchdog: independent process, kills by PID (not blocked by FIFO)
    (
        sleep "$timeout"
        if kill -0 "$agent_pid" 2>/dev/null; then
            echo "[$(date '+%H:%M:%S')] WATCHDOG: killing agent PID=$agent_pid after ${timeout}s" >> "$LOG_FILE"
            kill "$agent_pid" 2>/dev/null
            sleep 3
            kill -9 "$agent_pid" 2>/dev/null || true
        fi
    ) &
    local watchdog=$!

    # Wait for agent (or watchdog kill)
    wait "$agent_pid" 2>/dev/null
    local exit_code=$?

    # Cancel watchdog
    kill "$watchdog" 2>/dev/null || true
    wait "$watchdog" 2>/dev/null || true

    # Append agent output to main log
    cat "$agent_log" >> "$LOG_FILE"
    rm -f "/tmp/rustane-phase-agent.pid"

    log "Agent exited with code $exit_code"

    if [ $exit_code -ne 0 ]; then
        log "Agent failed or timed out"
        echo "$substep: FAILED (exit $exit_code)" > "$STATUS_FILE"
        gossip "Agent FAILED (exit $exit_code) at Phase $phase $substep"
        alert "WARN" "Agent failed at Phase $phase $substep (exit $exit_code)"
    fi

    # Validate substep output exists before advancing
    local advance=true
    case "$substep" in
        REFERENCE)
            if [ ! -f "/tmp/rustane-phase${phase}-reference.md" ]; then
                alert "WARN" "REFERENCE substep produced no output — retrying"
                advance=false
            fi
            ;;
        RESEARCH)
            if [ ! -f "/tmp/rustane-phase${phase}-research.md" ]; then
                alert "WARN" "RESEARCH substep produced no output — retrying"
                advance=false
            fi
            ;;
        PLAN)
            if [ ! -f "/tmp/rustane-phase${phase}-plan.md" ]; then
                alert "WARN" "PLAN substep produced no output — retrying"
                advance=false
            fi
            ;;
        IMPLEMENT)
            # Check if any commits were made
            local new_commits=$(cd "$worktree" && git log --oneline HEAD --not "phase${phase}-start" 2>/dev/null | wc -l | tr -d ' ')
            if [ "$new_commits" -eq 0 ] && [ $exit_code -ne 0 ]; then
                alert "WARN" "IMPLEMENT substep made no commits — retrying"
                advance=false
            fi
            ;;
    esac

    SESSIONS=$((SESSIONS + 1))

    if $advance; then
        SUBSTEP=$(next_substep "$substep")
        log "Advanced to substep: $SUBSTEP"
        gossip "Advanced to Phase $phase $SUBSTEP"
    else
        log "Substep $substep did not produce expected output — staying"
        gossip "Retrying Phase $phase $substep (no output)"
    fi

    save_state
}

# --- Main Loop ---
log "=== Phase Runner starting (Phase $PHASE, Substep $SUBSTEP) ==="
gossip "ONLINE — Phase $PHASE, Substep $SUBSTEP"
git tag "phase${PHASE}-start" HEAD 2>/dev/null || true
start_heartbeat_loop

# Auto-measure baseline at phase start (if not already set)
if [ "$SUBSTEP" = "REFERENCE" ] && [ "$BASELINE_MS" -eq 195 ]; then
    log "Measuring baseline before Phase $PHASE..."
    echo "BASELINE: measuring" > "$STATUS_FILE"
    BENCH_OUT=$(cargo test -p engine --test bench_step_time --release -- bench_training_step_1024 --ignored --nocapture 2>&1)
    BASELINE_MS=$(echo "$BENCH_OUT" | grep -E "^[2-4]" | awk '{sum+=$2; n++} END {printf "%.0f", sum/n}')
    log "Baseline measured: ${BASELINE_MS}ms"
    gossip "Baseline: ${BASELINE_MS}ms"
    save_state
fi

while [ "$PHASE" -le 5 ]; do
    # Check pause
    if [ -f "$PAUSE_FILE" ]; then
        log "PAUSED — remove $PAUSE_FILE to continue"
        echo "PAUSED" > "$STATUS_FILE"
        gossip "PAUSED — waiting for operator"
        while [ -f "$PAUSE_FILE" ]; do sleep 5; done
        log "RESUMED"
        gossip "RESUMED"
    fi

    # Check inject (applies to next agent session)
    INJECT_EXTRA=""
    INJECT_RESULT=$(check_inject)
    if [ -n "$INJECT_RESULT" ]; then
        INJECT_EXTRA="

ADDITIONAL INSTRUCTIONS FROM OPERATOR:
$INJECT_RESULT"
    fi

    log "--- Phase $PHASE, Substep $SUBSTEP (session $SESSIONS, retry $RETRIES) ---"
    gossip "Phase $PHASE, Substep $SUBSTEP starting"
    heartbeat

    if [ "$SUBSTEP" = "DONE" ]; then
        PHASE=$((PHASE + 1))
        SUBSTEP="REFERENCE"
        save_state
        continue
    fi

    build_prompt "$PHASE" "$SUBSTEP"

    # Check if agent stalled (no status update for 20+ minutes)
    if [ -f "$STATUS_FILE" ]; then
        STATUS_AGE=$(( $(date +%s) - $(stat -f%m "$STATUS_FILE" 2>/dev/null || echo "0") ))
        if [ "$STATUS_AGE" -gt 1200 ] && [ "$SUBSTEP" != "GATE" ]; then
            alert "WARN" "Agent status unchanged for $((STATUS_AGE/60))min — may be stuck"
        fi
    fi

    sleep 5
done

stop_heartbeat_loop
gossip "OFFLINE — all phases complete"
log "=== All phases complete ==="
