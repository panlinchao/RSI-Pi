"""Run pi as the evolver.

The evolver is pi itself: it inspects feedback and generates one candidate
modification to the selected module. It is launched with the retained module
read-only and a separate writable candidate workspace. It searches read-only evidence with its own bash/read tools
and rewrites `module/index.ts` with its own edit/write tools. Nothing is pre-digested
into a prompt beyond a short instruction and an index of where things are.

That matters because the published ablation on exactly this interface choice is
lopsided: scores-only 34.6 median, scores-plus-summary 34.9, full raw traces
behind a filesystem 50.0 -- with summaries sometimes doing worse than scores
alone. Packing traces into a prompt is the losing option.

The agent can choose which evidence to inspect before editing, rather than
receiving a fixed pre-built summary.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from . import harness as H

EVOLVER_IMAGE = "rsi-evolver:0.86.1"

# The evolver's own ceiling, so a proposal pass cannot run away. Tokens it
# spends count against the arm's total budget, including proposal generation.
#
# Set above what a careful pass actually needs. A smoke run spent 61 calls
# searching fifteen traces, reading the extension docs, verifying its own
# TypeScript, and then ran out of steps before writing its rationale -- the cap
# was shaping the result rather than guarding against a runaway.
EVOLVER_MAX_TOOL_CALLS = 120
EVOLVER_TIMEOUT_SEC = 3600

INSTRUCTIONS = """# Evolve one module of the coding agent

You are the evolution step of a recursive-self-improvement experiment. The agent
under study is pi driven by deepseek-flash, scored on incremental Go
implementation tasks from CCBench.

Exactly one module is under your control. Everything else about the agent --
the model, the tools, the decoding settings, the task set -- is frozen, and a
change that reaches outside this module is discarded unrun.

## What you have

- `/evidence/report.md` -- a pre-built failure analysis for the last evaluation: failures
  and near-misses with their verifier evidence, comfortable successes as one
  line each. **Start here, but do not trust it blindly.** It is a summary, and
  summaries lose signal. Verify anything you intend to act on.
- `/evidence/tasks/<task>/verifier.log` -- the full tester transcript for that task, ANSI
  stripped. Stage ids look like `[tester::#AB1]`. `\u2713` marks a passing assertion and
  `\U00010102` a failing one, followed by the expected-vs-got text. The root cause of a
  failure is in here.
- `/evidence/tasks/<task>/agent.log` -- the full agent transcript: every tool call with its
  arguments and the result the model saw. This is where waste is visible --
  repeated commands, recompiles after every edit, files re-read instead of
  edited.
- `/evidence/tasks/<task>/stats.json` -- reward, tokens, call count, tool mix, parsed
  verifier outcome.
- `/evidence/versions/` -- every module version evaluated so far.
- `/evidence/history.json`, `/evidence/attribution.md` -- what was tried, and whether its predictions
  held.
- `/current/index.ts` -- the retained module loaded in this session, read-only.
- `/workspace/module/index.ts` -- **the candidate file you edit.**
  `/current/SPEC.md` defines what this module may and may not change.

## The objective

Your change is evaluated on the same development tasks with everything else held
fixed, and kept only if it improves the score. The score is lexicographic:

1. **Mean reward first.** Solving more tasks always wins.
2. **API cost breaks a tie.** If the reward is unchanged, solving the same tasks
   for at least 20% less money counts as an improvement.

Both axes are real, and API cost is the one to watch. Most tokens in the
development dry run were cache reads, while generated tokens accounted for
most of its cost. A change that only trims cached context may save little;
reducing unnecessary turns can matter more. Check the actual usage of each
candidate rather than assuming this pattern will always hold.

## Rules

- Edit `module/index.ts` in place. Leave it a complete, valid pi extension with a
  default export.
- Import only from `@earendil-works/pi-coding-agent` and `typebox` (virtual
  modules, always resolvable) plus Node built-ins. Any other import fails at
  load time and forfeits the task.
- No task-specific hardcoding. Write a general policy, not a patch for one
  task's bug.
- Never crash the agent: the test runner may be absent, tools may be missing,
  and any handler you install must swallow its own errors.
- **Doing nothing is allowed.** If the evidence does not support a change you
  expect to help, leave the file as it is and say so. A speculative edit that
  loses a task is worse than no edit.

## When you are done

Write `/workspace/PROPOSAL.md` next to `module/` containing:

```
## Rationale
What you changed and which failure pattern or waste source it targets, and
whether you expect the gain as more tasks solved or lower API cost.

## Predicted fixes
Task names you expect to flip from failing to solved, or "none".

## Risk tasks
Task names you expect might regress, or "none".
```

Then stop. Do not summarise the evidence back to me.
"""


@dataclass
class EvolverResult:
    ran: bool
    source: str            # the module source after the run
    changed: bool
    rationale: str
    predicted_fixes: list[str]
    risk_tasks: list[str]
    tool_calls: int
    total_tokens: int
    returncode: int
    error: str | None = None


def build_workspace(
    *,
    workspace: Path,
    evidence_dir: Path,
    module: str,
    current_source: str,
) -> Path:
    """Keep the writable candidate apart from read-only evidence and current code."""
    if workspace.exists():
        raise FileExistsError(f"evolver workspace already exists: {workspace}")
    workspace.mkdir(parents=True)
    module_dir = workspace / "module"
    module_dir.mkdir()
    (module_dir / "index.ts").write_text(current_source, encoding="utf-8")
    (evidence_dir / "INSTRUCTIONS.md").write_text(INSTRUCTIONS, encoding="utf-8")
    shutil.copy2(H.MODULES_DIR / "API.md", evidence_dir / "API.md")
    return workspace


def parse_proposal(workspace: Path) -> tuple[str, list[str], list[str]]:
    """Read the rationale and predictions the evolver left behind."""
    path = workspace / "PROPOSAL.md"
    if not path.is_file():
        return "", [], []
    text = path.read_text(encoding="utf-8", errors="replace")

    def section(name: str) -> str:
        marker = f"## {name}"
        if marker not in text:
            return ""
        body = text.split(marker, 1)[1]
        # Stop at the next "## " heading, if any.
        return body.split("\n## ", 1)[0].strip()

    def tasks(name: str) -> list[str]:
        raw = section(name)
        if not raw or raw.strip().lower().startswith("none"):
            return []
        return [
            t.strip().strip("-*`")
            for t in raw.replace("\n", ",").split(",")
            if t.strip() and t.strip().lower() != "none"
        ]

    return section("Rationale"), tasks("Predicted fixes"), tasks("Risk tasks")


def run_evolver(
    *,
    workspace: Path,
    evidence_dir: Path,
    current_dir: Path,
    module: str,
    source_before: str,
    api_key: str,
    progress=print,
) -> EvolverResult:
    """Launch pi over the workspace and collect what it produced."""
    # docker binds require an absolute host path; a relative one is read as a
    # named volume and rejected.
    workspace = workspace.resolve()
    evidence_dir = evidence_dir.resolve()
    current_dir = current_dir.resolve()
    log_path = workspace / "evolver.jsonl"
    stderr_path = workspace / "evolver.stderr"

    # The retained module is loaded from a read-only mount. The agent edits a
    # separate candidate, so edits cannot change the code running this session.
    command = [
        "docker", "run", "--rm",
        "-v", f"{workspace}:/workspace:rw",
        "-v", f"{evidence_dir}:/evidence:ro",
        "-v", f"{current_dir}:/current:ro",
        "-v", f"{H.MODULES_DIR / 'budget'}:/budget:ro",
        "-w", "/workspace",
        "-e", "DEEPSEEK_API_KEY",
        "-e", f"PI_RSI_MAX_TOOL_CALLS={EVOLVER_MAX_TOOL_CALLS}",
        "-e", f"PI_RSI_MAX_TOKENS={H.MAX_TOKENS}",
        "-e", "PI_RSI_BUDGET_REPORT=/workspace/budget.json",
        "-e", "PI_CODING_AGENT_DIR=/tmp/pi-evolver-agent",
        EVOLVER_IMAGE,
        "pi", "--offline", "--mode", "json", "--no-session",
        "--no-extensions", "--no-skills", "--no-prompt-templates",
        "--no-themes", "--no-context-files", "--no-approve",
        # No --tools allowlist: it also filters extension tools, so the evolver
        # would not see any tool the module under test registers. pi's default
        # active set is already read/bash/edit/write.
        "-e", "/budget/index.ts",
        "-e", "/current/index.ts",
        "--model", H.MODEL,
        "--", "Read /evidence/INSTRUCTIONS.md and edit /workspace/module/index.ts as instructed.",
    ]
    progress(f"  evolver: docker run ({module}), cap {EVOLVER_MAX_TOOL_CALLS} calls")

    try:
        with log_path.open("wb") as out, stderr_path.open("wb") as err:
            completed = subprocess.run(
                command, stdout=out, stderr=err, timeout=EVOLVER_TIMEOUT_SEC,
                env={**os.environ, "DEEPSEEK_API_KEY": api_key},
            )
        returncode = completed.returncode
        error = None
    except subprocess.TimeoutExpired:
        returncode = -1
        error = f"evolver timed out after {EVOLVER_TIMEOUT_SEC}s"
    except OSError as exc:
        returncode = -1
        error = f"could not launch the evolver: {exc}"

    # Accounting reuses the same event shape the task runs produce.
    totals = summarize_evolver_log(log_path)
    source_after = (workspace / "module" / "index.ts").read_text(encoding="utf-8")
    rationale, predicted, risks = parse_proposal(workspace)

    # A non-zero exit is a failed proposal even if pi edited the module first.
    # Such a partial edit must never be evaluated as a valid candidate.
    if error is None and returncode != 0:
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace").strip()
        error = f"evolver exited {returncode}: {stderr[:400] or '(no stderr)'}"

    # A provider failure would otherwise look exactly like a deliberate no-op:
    # the module is untouched and the run exits cleanly.
    if error is None:
        from . import signals as S

        error = S.detect_api_error(log_path)

    result = EvolverResult(
        ran=True,
        source=source_after,
        changed=source_after.strip() != source_before.strip(),
        rationale=rationale,
        predicted_fixes=predicted,
        risk_tasks=risks,
        tool_calls=totals["tool_calls"],
        total_tokens=totals["total_tokens"],
        returncode=returncode,
        error=error,
    )
    (workspace / "evolver-result.json").write_text(
        json.dumps(asdict(result), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return result


def summarize_evolver_log(path: Path) -> dict[str, int]:
    """Sum the evolver's own token use and tool calls from its pi transcript."""
    totals = {"tool_calls": 0, "total_tokens": 0, "cost_usd": 0.0}
    if not path.is_file():
        return totals
    cost = 0.0
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "tool_execution_start":
            totals["tool_calls"] += 1
        elif kind == "message_end":
            message = event.get("message") or {}
            if message.get("role") != "assistant":
                continue
            usage = message.get("usage") or {}
            totals["total_tokens"] += (
                int(usage.get("input") or 0)
                + int(usage.get("cacheRead") or 0)
                + int(usage.get("cacheWrite") or 0)
                + int(usage.get("output") or 0)
            )
            entry = usage.get("cost")
            cost += entry if isinstance(entry, (int, float)) else float((entry or {}).get("total") or 0)
    totals["cost_usd"] = cost
    return totals
