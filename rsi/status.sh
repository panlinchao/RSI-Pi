#!/usr/bin/env bash
# Read-only progress view for a running experiment.
#
#   rsi/status.sh my-rsi-run
#   RSI_ROOT=/path/to/active/open_research_RSI rsi/status.sh my-rsi-run
#
# RSI_ROOT lets a separate worktree inspect the active run without changing
# the implementation revision that the experiment has locked.
set -euo pipefail

TAG="${1:-main}"
ROOT="${RSI_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
RUNS="$ROOT/bench/runs"
PYTHON_BIN="$ROOT/.venv/bin/python"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN=python3
fi

LOG="$ROOT/rsi/results/$TAG-driver.log"
if [[ ! -f "$LOG" ]]; then
    LOG="$ROOT/rsi/results/$TAG.log"
fi
echo "=== driver log ($TAG) ==="
if [[ -f "$LOG" ]]; then
    tail -n 25 "$LOG"
else
    echo "(no log yet)"
fi

echo
echo "=== conditions ==="
PYTHONPATH="$(dirname "$ROOT")${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" - "$RUNS" "$TAG" <<'PY'
import json
import sys
from pathlib import Path

from open_research_RSI.rsi import harness as H

runs, tag = Path(sys.argv[1]), sys.argv[2]
jobs = sorted(p for p in runs.glob(f"{tag}-*") if p.is_dir())
if not jobs:
    print("  (nothing started yet)")
for job in jobs:
    config = runs / "_configs" / f"{job.name}.json"
    expected = len(json.loads(config.read_text())["tasks"]) if config.is_file() else "?"
    started = sorted(p for p in job.iterdir() if p.is_dir())
    outcomes = []
    unreadable = 0
    for trial in started:
        result_path = trial / "result.json"
        if not result_path.is_file():
            continue
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            outcomes.append(H._outcome_from_trial(result.get("task_name") or trial.name, trial))
        except (OSError, ValueError, RuntimeError):
            unreadable += 1
    active = len(started) - len(outcomes) - unreadable
    retry_tokens, retry_cost = H._archived_spend(job.name)
    tokens = sum(o.total_tokens for o in outcomes) + retry_tokens
    cost = sum(o.cost_usd for o in outcomes) + retry_cost
    invalid = sum(bool(o.api_error) for o in outcomes)
    flag = f"  invalid trials={invalid}" if invalid else ""
    if unreadable:
        flag += f"  unreadable results={unreadable}"
    print(
        f"  {job.name}: completed {len(outcomes)}/{expected}, active {active}, "
        f"solved {sum(o.solved for o in outcomes)}, "
        f"tokens {tokens:,}, cost ${cost:.4f}{flag}"
    )
PY

echo
echo "=== containers ==="
docker ps --format '  {{.Names}} {{.Status}}' | sed -n '1,6p'
