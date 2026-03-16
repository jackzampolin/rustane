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
MODEL="claude-opus-4-6"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$LOG_FILE"; }

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
                RETRIES=$((RETRIES + 1))
                if [ $RETRIES -ge 3 ]; then
                    log "ESCALATE: 3 gate failures. Pausing for human review."
                    touch "$PAUSE_FILE"
                    SUBSTEP="IMPLEMENT"
                else
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
    if [ -d "$worktree" ]; then
        cd "$REPO_ROOT"
        git worktree remove --force "$worktree" 2>/dev/null || rm -rf "$worktree"
    fi
    cd "$REPO_ROOT"
    git fetch origin --quiet
    git worktree add "$worktree" "v2/ane-training" 2>/dev/null || {
        git worktree remove --force "$worktree" 2>/dev/null || rm -rf "$worktree"
        git worktree add "$worktree" "v2/ane-training"
    }

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

    # Run agent with timeout
    local timeout=10800  # 3 hours
    if [ "$substep" = "REFERENCE" ] || [ "$substep" = "RESEARCH" ] || [ "$substep" = "PLAN" ]; then
        timeout=1800  # 30 min for research/planning
    fi

    local fifo="/tmp/rustane-phase-fifo"
    rm -f "$fifo"
    mkfifo "$fifo"
    tee -a "$LOG_FILE" < "$fifo" &
    local tee_pid=$!

    claude -p \
        --dangerously-skip-permissions \
        --model "$MODEL" \
        "$prompt" > "$fifo" 2>&1 &
    local agent_pid=$!
    echo "$agent_pid" > "/tmp/rustane-phase-agent.pid"
    log "Agent PID=$agent_pid (timeout: ${timeout}s, substep: $substep)"

    # Watchdog
    (sleep "$timeout" && kill "$agent_pid" 2>/dev/null) &
    local watchdog=$!

    wait "$agent_pid" 2>/dev/null
    local exit_code=$?
    wait "$tee_pid" 2>/dev/null || true
    rm -f "$fifo" "/tmp/rustane-phase-agent.pid"
    kill "$watchdog" 2>/dev/null || true

    log "Agent exited with code $exit_code"

    if [ $exit_code -ne 0 ]; then
        log "Agent failed or timed out"
        echo "$substep: FAILED" > "$STATUS_FILE"
    fi

    # Advance substep
    SUBSTEP=$(next_substep "$substep")
    SESSIONS=$((SESSIONS + 1))
    save_state
    log "Advanced to substep: $SUBSTEP"
}

# --- Main Loop ---
log "=== Phase Runner starting (Phase $PHASE, Substep $SUBSTEP) ==="
git tag "phase${PHASE}-start" HEAD 2>/dev/null || true

while [ "$PHASE" -le 5 ]; do
    # Check pause
    if [ -f "$PAUSE_FILE" ]; then
        log "PAUSED — remove $PAUSE_FILE to continue"
        echo "PAUSED" > "$STATUS_FILE"
        while [ -f "$PAUSE_FILE" ]; do sleep 5; done
        log "RESUMED"
    fi

    log "--- Phase $PHASE, Substep $SUBSTEP ---"

    if [ "$SUBSTEP" = "DONE" ]; then
        log "Phase $PHASE substeps complete, this shouldn't happen"
        PHASE=$((PHASE + 1))
        SUBSTEP="REFERENCE"
        save_state
        continue
    fi

    build_prompt "$PHASE" "$SUBSTEP"

    sleep 5
done

log "=== All phases complete ==="
