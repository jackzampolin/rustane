#!/bin/bash
# project.sh — Run one step of a project's state machine
#
# Usage:
#   ./system/project.sh --project pre-transpose            # run current step
#   ./system/project.sh --project pre-transpose --status    # show status
#   ./system/project.sh --new "name" --desc "description"   # create new project
#   ./system/project.sh --list                              # list all projects
#
# In tmux:
#   tmux new -d -s proj-transpose './system/project.sh --project pre-transpose'
#
# Monitor:  ./system/project.sh --project pre-transpose --status
# Pause:    touch /tmp/rustane-project-PAUSE-pre-transpose
# Resume:   rm /tmp/rustane-project-PAUSE-pre-transpose
# Inject:   echo "focus on X" > /tmp/rustane-project-INJECT-pre-transpose
# Kill:     kill $(cat /tmp/rustane-project-PID-pre-transpose)

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROJECT_DIR="/tmp/rustane-projects"
HW_LOCK="/tmp/rustane-hw-lock"
MODEL="claude-opus-4-6"

mkdir -p "$PROJECT_DIR"

# --- Parse Args ---
PROJECT=""
SHOW_STATUS=false
NEW_PROJECT=false
LIST_PROJECTS=false
PROJECT_DESC=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --project)  PROJECT="$2"; shift 2 ;;
        --status)   SHOW_STATUS=true; shift ;;
        --new)      NEW_PROJECT=true; PROJECT="$2"; shift 2 ;;
        --desc)     PROJECT_DESC="$2"; shift 2 ;;
        --list)     LIST_PROJECTS=true; shift ;;
        --model)    MODEL="$2"; shift 2 ;;
        *)          echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# --- List ---
if $LIST_PROJECTS; then
    echo "=== Active Projects ==="
    for f in "$PROJECT_DIR"/*.md; do
        [ -f "$f" ] || continue
        name=$(basename "$f" .md)
        status=$(grep "^status:" "$f" | head -1 | cut -d' ' -f2-)
        branch=$(grep "^branch:" "$f" | head -1 | cut -d' ' -f2-)
        echo "  $name | $status | $branch"
    done
    exit 0
fi

# --- Validate ---
if [ -z "$PROJECT" ] && ! $LIST_PROJECTS; then
    echo "Usage: ./system/project.sh --project <name> [--status]"
    echo "       ./system/project.sh --new <name> --desc <description>"
    echo "       ./system/project.sh --list"
    exit 1
fi

PROJECT_FILE="$PROJECT_DIR/$PROJECT.md"
STATUS_FILE="/tmp/rustane-project-status-$PROJECT"
PAUSE_FILE="/tmp/rustane-project-PAUSE-$PROJECT"
INJECT_FILE="/tmp/rustane-project-INJECT-$PROJECT"
PID_FILE="/tmp/rustane-project-PID-$PROJECT"
LOG_FILE="/tmp/rustane-project-$PROJECT.log"
GOSSIP_FILE="/tmp/rustane-gossip.md"

log() {
    local msg="[$(date '+%H:%M:%S')] [proj:$PROJECT] $*"
    echo "$msg" | tee -a "$LOG_FILE"
}

# --- New Project ---
if $NEW_PROJECT; then
    if [ -f "$PROJECT_FILE" ]; then
        echo "Project $PROJECT already exists at $PROJECT_FILE"
        exit 1
    fi
    BRANCH="feature/$PROJECT"
    cat > "$PROJECT_FILE" << EOF
# Project: $PROJECT
status: RESEARCH
branch: $BRANCH
created: $(date '+%Y-%m-%d %H:%M')
description: $PROJECT_DESC
model: $MODEL

## Steps
1. [ ] RESEARCH — read codebase, understand the approach
2. [ ] PLAN — write implementation steps
3. [ ] IMPLEMENT — code the change (incremental commits)
4. [ ] TEST — run all tests
5. [ ] BENCHMARK — measure performance
6. [ ] REVIEW — check the diff
7. [ ] DECIDE — merge or abandon

## Notes
(agents write findings here as they work)

## Log
EOF
    # Create branch
    cd "$REPO_ROOT"
    git fetch origin --quiet
    git branch "$BRANCH" origin/auto-max 2>/dev/null || git branch -f "$BRANCH" origin/auto-max
    echo "Created project: $PROJECT_FILE"
    echo "Branch: $BRANCH"
    echo ""
    echo "Next: edit $PROJECT_FILE to add details, then run:"
    echo "  tmux new -d -s proj-$PROJECT './system/project.sh --project $PROJECT'"
    exit 0
fi

# --- Status ---
if $SHOW_STATUS; then
    echo "=== Project: $PROJECT ==="
    if [ ! -f "$PROJECT_FILE" ]; then
        echo "No project file at $PROJECT_FILE"
        exit 1
    fi
    echo ""
    grep -E "^status:|^branch:|^description:" "$PROJECT_FILE" | sed 's/^/  /'
    echo ""
    echo "Agent: $(cat "$STATUS_FILE" 2>/dev/null || echo 'not running')"
    if [ -f "$PID_FILE" ]; then
        PID=$(cat "$PID_FILE")
        if kill -0 "$PID" 2>/dev/null; then
            ELAPSED=$(ps -o etime= -p "$PID" 2>/dev/null | tr -d ' ')
            echo "PID: $PID (elapsed: $ELAPSED)"
        else
            echo "PID: $PID (dead)"
        fi
    fi
    echo ""
    [ -f "$PAUSE_FILE" ] && echo "PAUSED" || echo "Running"
    [ -f "$INJECT_FILE" ] && echo "INJECT queued: $(head -1 "$INJECT_FILE")"
    echo ""
    echo "--- Steps ---"
    grep -E "^\d+\." "$PROJECT_FILE" | head -10
    echo ""
    echo "--- Recent Log ---"
    tail -10 "$LOG_FILE" 2>/dev/null || echo "(no log)"
    exit 0
fi

# --- Run Current Step ---
if [ ! -f "$PROJECT_FILE" ]; then
    echo "No project file at $PROJECT_FILE. Create one with --new"
    exit 1
fi

STATUS=$(grep "^status:" "$PROJECT_FILE" | head -1 | cut -d' ' -f2-)
BRANCH=$(grep "^branch:" "$PROJECT_FILE" | head -1 | cut -d' ' -f2-)
DESC=$(grep "^description:" "$PROJECT_FILE" | head -1 | cut -d' ' -f2-)

log "=== Starting project step: $STATUS ==="
log "Branch: $BRANCH | Model: $MODEL"

# Check pause
if [ -f "$PAUSE_FILE" ]; then
    log "PAUSED — remove $PAUSE_FILE to resume"
    while [ -f "$PAUSE_FILE" ]; do sleep 5; done
    log "RESUMED"
fi

# Set up worktree
WORKTREE="/tmp/rustane-worktree-$PROJECT"
if [ -d "$WORKTREE" ]; then
    cd "$REPO_ROOT"
    git worktree remove --force "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"
fi
cd "$REPO_ROOT"
git fetch origin --quiet
git branch -f "$BRANCH" "origin/auto-max" 2>/dev/null || true
git worktree add "$WORKTREE" "$BRANCH" 2>/dev/null || {
    git worktree remove --force "$WORKTREE" 2>/dev/null || rm -rf "$WORKTREE"
    git worktree add "$WORKTREE" "$BRANCH"
}

# Copy dev/ context
if [ -d "${REPO_ROOT}/dev" ]; then
    mkdir -p "${WORKTREE}/dev"
    for f in CURRENT.md METHODOLOGY.md; do
        [ -f "${REPO_ROOT}/dev/$f" ] && cp "${REPO_ROOT}/dev/$f" "${WORKTREE}/dev/$f"
    done
    [ -d "${REPO_ROOT}/dev/sessions" ] && cp -a "${REPO_ROOT}/dev/sessions" "${WORKTREE}/dev/sessions"
    [ -d "${REPO_ROOT}/dev/plans" ] && cp -a "${REPO_ROOT}/dev/plans" "${WORKTREE}/dev/plans"
    [ -d "${REPO_ROOT}/dev/info" ] && cp -a "${REPO_ROOT}/dev/info" "${WORKTREE}/dev/info"
fi

cd "$WORKTREE"
log "Working in: $WORKTREE"

# Check for inject
INJECT_EXTRA=""
if [ -f "$INJECT_FILE" ]; then
    INJECT_EXTRA="

ADDITIONAL INSTRUCTIONS FROM OPERATOR:
$(cat "$INJECT_FILE")"
    rm -f "$INJECT_FILE"
    log "INJECT: loaded extra instructions"
fi

# --- Build prompt based on state ---
TIMEOUT_SEC=7200  # 2 hours default

case "$STATUS" in
    RESEARCH)
        TIMEOUT_SEC=1800  # 30 min
        PROMPT="You are a research agent for the rustane training engine on Apple M4 Max.

PROJECT: $DESC

Read AGENTS.md first — it has hardware facts and proven dead ends.
Then read the relevant source code to understand what needs to change.
Read dev/plans/ for any existing plans.
Read system/experiments.tsv for what's been tried.

YOUR TASK: Research only. Do NOT write code.
Write your findings to the project file by appending to the Notes section.
When done, update the status line from 'RESEARCH' to 'PLANNING'.

Project file: $PROJECT_FILE
Status file: echo 'RESEARCHING: <what>' > $STATUS_FILE
$INJECT_EXTRA"
        ;;

    PLANNING)
        TIMEOUT_SEC=1800  # 30 min
        PROMPT="You are a planning agent for the rustane training engine.

PROJECT: $DESC

Read the project file at $PROJECT_FILE — it has research notes from the previous step.
Read AGENTS.md for constraints and dead ends.

YOUR TASK: Write a detailed implementation plan.
Update the Steps section in the project file with specific sub-steps for IMPLEMENT.
Each sub-step should be completable in under 30 minutes and touch at most 2 files.
Include which functions to modify, the exact approach, and test strategy.

When done, update status from 'PLANNING' to 'IMPLEMENTING'.

Status file: echo 'PLANNING: <what>' > $STATUS_FILE
$INJECT_EXTRA"
        ;;

    IMPLEMENTING)
        TIMEOUT_SEC=10800  # 3 hours
        PROMPT="You are an implementation agent for the rustane training engine on Apple M4 Max.

PROJECT: $DESC
BRANCH: $BRANCH

Read the project file at $PROJECT_FILE — it has the implementation plan.
Read AGENTS.md for hardware facts and dead ends.

YOUR TASK: Implement the plan step by step.
- Commit after each sub-step (incremental, not one big commit)
- Push to origin $BRANCH after each commit
- Write status updates: echo 'IMPLEMENTING: step N — <what>' > $STATUS_FILE
- If a sub-step fails, note WHY in the project file and move on or adapt

IMPORTANT:
- Do NOT run benchmarks yet (that's a separate step with hardware lock)
- DO run cargo test -p engine --release after each commit to verify correctness
- Commit messages: 'feat($PROJECT): <what this step does>'
- Update the project file's step checkboxes as you complete them

When all implementation steps are done, update status to 'TESTING'.

Status file: echo 'IMPLEMENTING: step N' > $STATUS_FILE
$INJECT_EXTRA"
        ;;

    TESTING)
        TIMEOUT_SEC=1800  # 30 min
        # Hardware lock
        log "Acquiring hardware lock..."
        while [ -f "$HW_LOCK" ]; do
            HOLDER=$(cat "$HW_LOCK" 2>/dev/null || echo "unknown")
            log "HW lock held by $HOLDER — waiting..."
            sleep 10
        done
        echo "$PROJECT" > "$HW_LOCK"
        log "Got hardware lock"
        trap "rm -f '$HW_LOCK'; log 'Released HW lock'" EXIT

        PROMPT="You are a test agent for the rustane training engine.

PROJECT: $DESC
BRANCH: $BRANCH

Run ALL tests:
1. cargo test -p engine --release
2. cargo test -p engine --test phase3_kernels --release
3. cargo test -p engine --test phase4_training --release
4. Any project-specific tests in auto_*.rs

Write results to project file.
If all pass: update status to 'BENCHMARKING'
If any fail: update status to 'IMPLEMENTING' with notes on what broke.

Status file: echo 'TESTING: <which suite>' > $STATUS_FILE
$INJECT_EXTRA"
        ;;

    BENCHMARKING)
        TIMEOUT_SEC=1800  # 30 min
        # Hardware lock (may already have it from TESTING)
        if [ ! -f "$HW_LOCK" ] || [ "$(cat "$HW_LOCK")" != "$PROJECT" ]; then
            log "Acquiring hardware lock..."
            while [ -f "$HW_LOCK" ]; do
                sleep 10
            done
            echo "$PROJECT" > "$HW_LOCK"
            trap "rm -f '$HW_LOCK'; log 'Released HW lock'" EXIT
        fi

        PROMPT="You are a benchmark agent for the rustane training engine.

PROJECT: $DESC
BRANCH: $BRANCH

Run benchmarks:
1. cargo test -p engine --test bench_step_time --release -- --ignored --nocapture
2. cargo test -p engine --test profile_single_layer --release -- --ignored --nocapture
3. Run each 3 times for stability

Compare against baseline (102ms/step on auto-max).
Write results to project file AND system/experiments.tsv.
Write to gossip: echo '[time] [proj:$PROJECT] RESULT: ...' >> $GOSSIP_FILE

Update status to 'REVIEW' with benchmark numbers.

Status file: echo 'BENCHMARKING: run N/3' > $STATUS_FILE
$INJECT_EXTRA"
        ;;

    REVIEW)
        TIMEOUT_SEC=900  # 15 min
        PROMPT="You are a code review agent for the rustane training engine.

PROJECT: $DESC
BRANCH: $BRANCH

Review the changes on this branch vs auto-max:
  git diff origin/auto-max..HEAD

Check:
- Numerical correctness (no math changes unless justified)
- Test coverage (every change has a test)
- No regressions in existing tests
- Code follows patterns in AGENTS.md

Write review findings to project file.
If approved: update status to 'READY_TO_MERGE'
If issues found: update status to 'IMPLEMENTING' with issues to fix.

Status file: echo 'REVIEWING' > $STATUS_FILE
$INJECT_EXTRA"
        ;;

    READY_TO_MERGE|DONE|ABANDONED)
        log "Project is $STATUS — nothing to run"
        exit 0
        ;;

    *)
        log "Unknown status: $STATUS"
        exit 1
        ;;
esac

# --- Run the agent ---
FIFO="/tmp/rustane-fifo-proj-$PROJECT"
rm -f "$FIFO"
mkfifo "$FIFO"

tee -a "$LOG_FILE" < "$FIFO" &
TEE_PID=$!

claude -p \
    --dangerously-skip-permissions \
    --model "$MODEL" \
    "$PROMPT" > "$FIFO" 2>&1 &
AGENT_PID=$!
echo "$AGENT_PID" > "$PID_FILE"
log "Agent PID=$AGENT_PID (timeout: ${TIMEOUT_SEC}s)"

# Watchdog
(
    sleep "$TIMEOUT_SEC"
    if kill -0 "$AGENT_PID" 2>/dev/null; then
        log "TIMEOUT: killing agent after ${TIMEOUT_SEC}s"
        kill "$AGENT_PID" 2>/dev/null
        sleep 3
        kill -9 "$AGENT_PID" 2>/dev/null || true
    fi
) &
WATCHDOG_PID=$!

wait "$AGENT_PID"
EXIT_CODE=$?
wait "$TEE_PID" 2>/dev/null || true
rm -f "$FIFO" "$PID_FILE"

kill "$WATCHDOG_PID" 2>/dev/null || true
wait "$WATCHDOG_PID" 2>/dev/null || true

log "Agent exited with code $EXIT_CODE"

# Release HW lock if we hold it
if [ -f "$HW_LOCK" ] && [ "$(cat "$HW_LOCK" 2>/dev/null)" = "$PROJECT" ]; then
    rm -f "$HW_LOCK"
    log "Released hardware lock"
fi

# Read new status
NEW_STATUS=$(grep "^status:" "$PROJECT_FILE" | head -1 | cut -d' ' -f2-)
log "Status: $STATUS → $NEW_STATUS"

# If status changed, loop to next step
if [ "$NEW_STATUS" != "$STATUS" ] && [ "$NEW_STATUS" != "DONE" ] && [ "$NEW_STATUS" != "ABANDONED" ] && [ "$NEW_STATUS" != "READY_TO_MERGE" ]; then
    log "Advancing to next step..."
    sleep 5
    exec "$0" --project "$PROJECT" --model "$MODEL"
fi

log "=== Project step complete ==="
