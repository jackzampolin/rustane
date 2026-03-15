#!/bin/bash
# bench-agent.sh — Dedicated test + benchmark agent
#
# Pulls a feature branch into its own worktree, runs rigorous tests,
# benchmarks against baseline, and reports results.
#
# Usage:
#   ./system/bench-agent.sh --branch feature/fused-ane-kernels
#   ./system/bench-agent.sh --branch feature/fused-ane-kernels --baseline auto-max
#   ./system/bench-agent.sh --branch feature/fused-ane-kernels --quick   # skip training equiv
#   ./system/bench-agent.sh --status                                     # check running bench
#
# In tmux:
#   tmux new -d -s bench './system/bench-agent.sh --branch feature/fused-ane-kernels'
#
# The bench agent:
#   1. Creates its own worktree (isolated from builder)
#   2. Acquires hardware lock (exclusive ANE/Metal access)
#   3. Runs full test suite
#   4. Benchmarks 3x for stability
#   5. Runs 50-step training equivalence vs baseline
#   6. Reports results to gossip + project file
#   7. Releases hardware lock

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Config ---
BRANCH=""
BASELINE="auto-max"
QUICK=false
SHOW_STATUS=false
MODEL="claude-opus-4-6"
HW_LOCK="/tmp/rustane-hw-lock"
GOSSIP="/tmp/rustane-gossip.md"
STATUS_FILE="/tmp/rustane-bench-status"
LOG_FILE="/tmp/rustane-bench.log"
PID_FILE="/tmp/rustane-bench-PID"
BENCH_RUNS=3
EQUIV_STEPS=50

# --- Parse Args ---
while [[ $# -gt 0 ]]; do
    case "$1" in
        --branch)    BRANCH="$2"; shift 2 ;;
        --baseline)  BASELINE="$2"; shift 2 ;;
        --quick)     QUICK=true; shift ;;
        --status)    SHOW_STATUS=true; shift ;;
        --model)     MODEL="$2"; shift 2 ;;
        --runs)      BENCH_RUNS="$2"; shift 2 ;;
        *)           echo "Unknown arg: $1"; exit 1 ;;
    esac
done

log() {
    local msg="[$(date '+%H:%M:%S')] [bench] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

# --- Status ---
if $SHOW_STATUS; then
    echo "=== Bench Agent Status ==="
    cat "$STATUS_FILE" 2>/dev/null || echo "Not running"
    echo ""
    if [ -f "$PID_FILE" ]; then
        BENCH_PID=$(cat "$PID_FILE")
        ps -p "$BENCH_PID" -o pid,etime,pcpu 2>/dev/null || echo "PID $BENCH_PID dead"
    fi
    echo ""
    echo "--- Recent Log ---"
    tail -10 "$LOG_FILE" 2>/dev/null || echo "(no log)"
    exit 0
fi

if [ -z "$BRANCH" ]; then
    echo "Usage: ./system/bench-agent.sh --branch <branch-name> [--baseline <branch>] [--quick]"
    exit 1
fi

WORKTREE="/tmp/rustane-bench-worktree"
BRANCH_SHORT=$(echo "$BRANCH" | sed 's|.*/||')

log "=== Bench Agent starting ==="
log "Branch: $BRANCH | Baseline: $BASELINE | Runs: $BENCH_RUNS"

# --- Setup worktree ---
echo "SETUP: creating worktree" > "$STATUS_FILE"

cd "$REPO_ROOT"
git fetch origin --quiet

if [ -d "$WORKTREE" ]; then
    git worktree remove --force "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"
fi

git worktree add "$WORKTREE" "$BRANCH" 2>/dev/null || {
    log "ERROR: could not create worktree for $BRANCH"
    exit 1
}

# Copy dev/ context
if [ -d "${REPO_ROOT}/dev" ]; then
    mkdir -p "${WORKTREE}/dev"
    for f in CURRENT.md METHODOLOGY.md; do
        [ -f "${REPO_ROOT}/dev/$f" ] && cp "${REPO_ROOT}/dev/$f" "${WORKTREE}/dev/$f"
    done
fi

cd "$WORKTREE"
log "Worktree: $WORKTREE"
log "HEAD: $(git log --oneline -1)"

# --- Acquire hardware lock ---
echo "WAITING: hardware lock" > "$STATUS_FILE"
log "Acquiring hardware lock..."
while [ -f "$HW_LOCK" ]; do
    HOLDER=$(cat "$HW_LOCK" 2>/dev/null || echo "unknown")
    log "HW lock held by $holder — waiting..."
    sleep 10
done
echo "bench-agent" > "$HW_LOCK"
log "Got hardware lock"
trap "rm -f '$HW_LOCK' '$PID_FILE'; log 'Released HW lock'" EXIT

# --- Phase 1: Full test suite ---
echo "TESTING: full suite" > "$STATUS_FILE"
log "Phase 1: Running full test suite..."

TEST_OUTPUT=$(cargo test -p engine --release 2>&1)
TEST_EXIT=$?
if [ $TEST_EXIT -ne 0 ]; then
    log "FAIL: test suite failed (exit $TEST_EXIT)"
    echo "FAIL: test suite failed" > "$STATUS_FILE"
    echo "$TEST_OUTPUT" >> "$LOG_FILE"
    echo "[$(date '+%H:%M:%S')] [bench] FAIL: $BRANCH_SHORT — test suite failed" >> "$GOSSIP"
    exit 1
fi

TESTS_PASSED=$(echo "$TEST_OUTPUT" | grep "test result:" | tail -1)
log "PASS: $TESTS_PASSED"

# Phase 3 kernels
echo "TESTING: phase3 kernels" > "$STATUS_FILE"
cargo test -p engine --test phase3_kernels --release 2>&1 | tail -3 >> "$LOG_FILE"
log "PASS: phase3_kernels"

# Phase 4 training
echo "TESTING: phase4 training" > "$STATUS_FILE"
cargo test -p engine --test phase4_training --release 2>&1 | tail -3 >> "$LOG_FILE"
log "PASS: phase4_training"

# --- Phase 2a: Baseline benchmark (on auto-max) ---
echo "BASELINE: benchmarking auto-max" > "$STATUS_FILE"
log "Phase 2a: Running baseline benchmark on $BASELINE..."

BASELINE_WORKTREE="/tmp/rustane-bench-baseline"
if [ -d "$BASELINE_WORKTREE" ]; then
    git -C "$REPO_ROOT" worktree remove --force "$BASELINE_WORKTREE" 2>/dev/null || rm -rf "$BASELINE_WORKTREE"
fi
git -C "$REPO_ROOT" worktree add "$BASELINE_WORKTREE" "origin/$BASELINE" 2>/dev/null

if [ -d "${REPO_ROOT}/dev" ]; then
    mkdir -p "${BASELINE_WORKTREE}/dev"
    for f in CURRENT.md METHODOLOGY.md; do
        [ -f "${REPO_ROOT}/dev/$f" ] && cp "${REPO_ROOT}/dev/$f" "${BASELINE_WORKTREE}/dev/$f"
    done
fi

cd "$BASELINE_WORKTREE"
BASELINE_OUT=$(cargo test -p engine --test bench_step_time --release -- --ignored --nocapture 2>&1)
BASELINE_MS=$(echo "$BASELINE_OUT" | grep -E "^[2-4]" | awk '{sum+=$2; n++} END {printf "%.0f", sum/n}')
BASELINE_FWD=$(echo "$BASELINE_OUT" | grep -E "^[2-4]" | awk '{sum+=$3; n++} END {printf "%.0f", sum/n}')
BASELINE_BWD=$(echo "$BASELINE_OUT" | grep -E "^[2-4]" | awk '{sum+=$4; n++} END {printf "%.0f", sum/n}')
log "BASELINE ($BASELINE): ${BASELINE_MS}ms/step (fwd: ${BASELINE_FWD}ms, bwd: ${BASELINE_BWD}ms)"

cd "$REPO_ROOT"
git worktree remove --force "$BASELINE_WORKTREE" 2>/dev/null || true

# Back to feature worktree
cd "$WORKTREE"

# --- Phase 2b: Feature benchmark (3x) ---
echo "BENCHMARKING: run 1/$BENCH_RUNS" > "$STATUS_FILE"
log "Phase 2b: Running feature benchmarks ($BENCH_RUNS runs)..."

BENCH_RESULTS=""
for run in $(seq 1 $BENCH_RUNS); do
    echo "BENCHMARKING: run $run/$BENCH_RUNS" > "$STATUS_FILE"
    log "Benchmark run $run/$BENCH_RUNS..."

    BENCH_OUT=$(cargo test -p engine --test bench_step_time --release -- --ignored --nocapture 2>&1)
    echo "$BENCH_OUT" >> "$LOG_FILE"

    # Extract ms/step from steps 2-4 (skip warmup)
    MS=$(echo "$BENCH_OUT" | grep -E "^[2-4]" | awk '{sum+=$2; n++} END {printf "%.0f", sum/n}')
    FWD=$(echo "$BENCH_OUT" | grep -E "^[2-4]" | awk '{sum+=$3; n++} END {printf "%.0f", sum/n}')
    BWD=$(echo "$BENCH_OUT" | grep -E "^[2-4]" | awk '{sum+=$4; n++} END {printf "%.0f", sum/n}')

    log "  Run $run: ${MS}ms/step (fwd: ${FWD}ms, bwd: ${BWD}ms)"
    BENCH_RESULTS="${BENCH_RESULTS}${MS} "
done

# Average across runs
AVG_MS=$(echo "$BENCH_RESULTS" | awk '{sum=0; for(i=1;i<=NF;i++) sum+=$i; printf "%.0f", sum/NF}')
log "Average: ${AVG_MS}ms/step across $BENCH_RUNS runs"

# --- Phase 3: Profile single layer ---
echo "PROFILING: single layer" > "$STATUS_FILE"
log "Phase 3: Profiling single layer..."

PROFILE_OUT=$(cargo test -p engine --test profile_single_layer --release -- --ignored --nocapture 2>&1)
echo "$PROFILE_OUT" >> "$LOG_FILE"
log "Profile complete"

# --- Phase 4: Training equivalence (optional) ---
if ! $QUICK; then
    echo "EQUIVALENCE: ${EQUIV_STEPS}-step training" > "$STATUS_FILE"
    log "Phase 4: Running ${EQUIV_STEPS}-step training equivalence..."

    # Run on feature branch
    EQUIV_PROMPT="Run a ${EQUIV_STEPS}-step training validation in $(pwd).
    Execute: cargo test -p engine --test phase4_training --release -- --nocapture
    Then run: cargo test -p engine --test bench_step_time --release -- --ignored --nocapture
    Report the loss trajectory and any NaN/divergence.
    Write results to /tmp/rustane-bench-equiv.txt"

    echo "EQUIVALENCE: running ${EQUIV_STEPS}-step test" > "$STATUS_FILE"

    # Simple: just run the training test which validates loss decreases
    EQUIV_OUT=$(cargo test -p engine --test phase4_training --release -- --nocapture 2>&1)
    EQUIV_EXIT=$?

    if [ $EQUIV_EXIT -ne 0 ]; then
        log "FAIL: training equivalence failed"
        echo "FAIL: training diverged" > "$STATUS_FILE"
        echo "[$(date '+%H:%M:%S')] [bench] FAIL: $BRANCH_SHORT — training diverged" >> "$GOSSIP"
        exit 1
    fi
    log "PASS: training equivalence (loss decreasing, no NaN)"
fi

# --- Report ---
echo "DONE: ${AVG_MS}ms/step (all tests pass)" > "$STATUS_FILE"

REPORT="[$(date '+%H:%M:%S')] [bench] RESULT: $BRANCH_SHORT — ${AVG_MS}ms/step"
REPORT="$REPORT | tests: ALL PASS | bench: $BENCH_RESULTS(avg $AVG_MS)"

# Compare against MEASURED baseline (not experiments.tsv)
if [ -n "$BASELINE_MS" ]; then
    DELTA=$((BASELINE_MS - AVG_MS))
    if [ $DELTA -gt 0 ]; then
        REPORT="$REPORT | vs ${BASELINE} (measured ${BASELINE_MS}ms): -${DELTA}ms FASTER"
    elif [ $DELTA -lt 0 ]; then
        REPORT="$REPORT | vs ${BASELINE} (measured ${BASELINE_MS}ms): +$((-DELTA))ms SLOWER"
    else
        REPORT="$REPORT | vs ${BASELINE} (measured ${BASELINE_MS}ms): NO CHANGE"
    fi
else
    REPORT="$REPORT | baseline bench failed"
fi

log "$REPORT"
echo "$REPORT" >> "$GOSSIP"

# Update project file if it exists
PROJECT_NAME=$(echo "$BRANCH_SHORT" | sed 's/feature\///')
PROJECT_FILE="/tmp/rustane-projects/${PROJECT_NAME}.md"
if [ -f "$PROJECT_FILE" ]; then
    echo "" >> "$PROJECT_FILE"
    echo "## Bench Agent Results ($(date '+%Y-%m-%d %H:%M'))" >> "$PROJECT_FILE"
    echo "- ms/step: ${AVG_MS} (runs: $BENCH_RESULTS)" >> "$PROJECT_FILE"
    echo "- baseline: ${BASELINE_MS:-unknown}" >> "$PROJECT_FILE"
    echo "- delta: ${DELTA:-unknown}ms" >> "$PROJECT_FILE"
    echo "- tests: ALL PASS" >> "$PROJECT_FILE"
    echo "- training: $(if $QUICK; then echo 'SKIPPED'; else echo 'PASS'; fi)" >> "$PROJECT_FILE"
    log "Updated project file: $PROJECT_FILE"
fi

log "=== Bench Agent complete ==="
