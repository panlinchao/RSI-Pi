"""Materialize a round's evidence as a directory the proposer can search.

Why a directory instead of a prompt. The Meta-Harness ablation (Table 3) puts
scores-only at 34.6 median, scores-plus-summary at 34.9, and full raw traces
behind a filesystem at 50.0 -- with the note that "summaries do not recover the
missing signal, and may even hurt". AHE independently refuses to pack traces
into its evolve prompt: it ships a structured query, then tells the agent to go
read the raw tracer files itself.

So the evidence is written to disk and mounted read-only for the proposer's
`bash` and `read` tools. The structured report is still
written -- as `report.md`, an index and starting point, not as the whole input.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict
from pathlib import Path

from . import harness as H
from . import signals as S

EVIDENCE_DIR = H.ROOT / "rsi" / "evidence"

# Per-result cap inside a cleaned agent trace. Large enough that a grep hit is
# self-explanatory, small enough that reading a whole trace is still feasible.
RESULT_EXCERPT = 1800


def _clean_agent_trace(trial_dir: Path, log_name: str = "pi.jsonl") -> str:
    """Render pi's event stream as readable text, pairing calls with results."""
    path = trial_dir / "agent" / log_name
    if not path.is_file():
        return "(no agent transcript)\n"

    out: list[str] = []
    pending: dict[str, dict] = {}
    index = 0

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")

        if kind == "tool_execution_start":
            index += 1
            args = event.get("args") or {}
            record = {"index": index, "name": event.get("toolName") or "?"}
            pending[event.get("toolCallId") or ""] = record
            out.append(f"\n=== call {index}: {record['name']} ===")
            out.append(json.dumps(args, ensure_ascii=False, indent=1)[:RESULT_EXCERPT])
        elif kind == "tool_execution_end":
            record = pending.pop(event.get("toolCallId") or "", None)
            if record is None:
                continue
            result = event.get("result") or {}
            text = "".join(
                block.get("text", "")
                for block in (result.get("content") or [])
                if isinstance(block, dict)
            )
            flag = " ERROR" if result.get("isError") else ""
            out.append(f"--- result {record['index']}{flag} ---")
            out.append(text[:RESULT_EXCERPT] or "(empty)")
        elif kind == "message_end":
            message = event.get("message") or {}
            if message.get("role") == "assistant":
                for block in message.get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                        out.append(f"\n=== assistant turn ===\n{block['text'][:RESULT_EXCERPT]}")

    return "\n".join(out) + "\n"


def _clean_verifier(trial_dir: Path) -> str:
    path = trial_dir / "verifier" / "test-stdout.txt"
    if not path.is_file():
        return "(no verifier transcript)\n"
    return S.strip_ansi(path.read_text(encoding="utf-8", errors="replace"))


def materialize(
    *,
    round_dir: Path,
    outcomes: list,
    module: str,
    tag: str,
    arm,
    report: str,
    history: list,
    attribution: str,
) -> Path:
    """Write the evidence tree for one round and return its root."""
    if round_dir.exists():
        raise FileExistsError(f"round evidence already exists: {round_dir}")
    tasks_dir = round_dir / "tasks"
    tasks_dir.mkdir(parents=True)

    signal_list = [S.collect(o) for o in outcomes]
    by_task = {s.task: s for s in signal_list}

    index_lines = [
        "# Evidence for this round",
        "",
        f"Agent: pi + {H.MODEL}. Module under test: `{module}`.",
        "",
        "## Where things are",
        "",
        "- `report.md` -- a pre-built failure analysis: failures and near-misses in",
        "  full, comfortable successes as one line. Start here, but verify claims",
        "  against the raw files below before acting on them.",
        "- `tasks/<task>/verifier.log` -- the FULL tester transcript for that task,",
        "  with ANSI codes stripped. Stage ids look like `[tester::#AB1]`; `✓` marks",
        "  a passing assertion and `𐄂` a failing one, followed by the expected-vs-got",
        "  text. This is where the actual root cause of a failure lives.",
        "- `tasks/<task>/agent.log` -- the FULL agent transcript: every tool call with",
        "  its arguments and the result the model saw. Use `grep` over this to find",
        "  repeated commands, wasted recompiles, or a verification step the agent",
        "  skipped.",
        "- `tasks/<task>/stats.json` -- reward, token use, call count, tool mix,",
        "  and the parsed verifier outcome for that task.",
        "- `versions/` -- every module version evaluated so far, including the one",
        "  currently retained.",
        "- `history.json` -- each previous round: what was proposed, whether it was",
        "  accepted, and what it predicted.",
        "- `attribution.md` -- whether the previous round's predictions held.",
        "",
        "## This round's task set",
        "",
        "| task | outcome | reward | calls | tokens | verifier |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for signals in signal_list:
        v = signals.verifier
        if signals.solved:
            verdict = "solved"
        elif v.note:
            verdict = v.note
        elif v.failed_stage:
            verdict = f"failed at {v.failed_stage} after clearing {v.stages_cleared_before_failure}/{v.stages_total}"
        else:
            verdict = "failed"
        index_lines.append(
            f"| {signals.task} | {'SOLVED' if signals.solved else 'FAILED'} "
            f"| {signals.reward} | {signals.tool_calls} | {signals.tokens:,} | {verdict} |"
        )

    total_tokens = sum(s.tokens for s in signal_list)
    index_lines += [
        "",
        f"Total across the task set: {sum(1 for s in signal_list if s.solved)}"
        f"/{len(signal_list)} solved, {total_tokens:,} tokens.",
        "",
    ]
    (round_dir / "README.md").write_text("\n".join(index_lines), encoding="utf-8")
    (round_dir / "report.md").write_text(report, encoding="utf-8")
    (round_dir / "attribution.md").write_text(attribution, encoding="utf-8")

    for outcome in outcomes:
        signals = by_task.get(outcome.task)
        if signals is None:
            continue
        task_dir = tasks_dir / outcome.task
        task_dir.mkdir(parents=True, exist_ok=True)
        if outcome.trial_dir is not None:
            (task_dir / "verifier.log").write_text(
                _clean_verifier(outcome.trial_dir), encoding="utf-8"
            )
            (task_dir / "agent.log").write_text(
                _clean_agent_trace(outcome.trial_dir), encoding="utf-8"
            )
        (task_dir / "stats.json").write_text(
            json.dumps(
                {
                    "task": signals.task,
                    "reward": signals.reward,
                    "solved": signals.solved,
                    "tool_calls": signals.tool_calls,
                    "tokens": signals.tokens,
                    "cost_usd": signals.cost_usd,
                    "budget_stopped_by": signals.exhausted_by,
                    "harness_exception": signals.exception,
                    "verifier": asdict(signals.verifier),
                    "behavior": asdict(signals.behavior),
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    # Prior module versions, so the proposer can inspect what was actually tried
    # rather than only reading the rationale for it.
    versions_dir = round_dir / "versions"
    versions_dir.mkdir(parents=True, exist_ok=True)
    for version_dir in sorted((H.ROOT / "rsi" / "versions" / tag / module).glob("v*")):
        source = version_dir / "index.ts"
        if source.is_file():
            (versions_dir / f"{version_dir.name}.ts").write_text(
                source.read_text(encoding="utf-8"), encoding="utf-8"
            )

    (round_dir / "history.json").write_text(
        json.dumps(
            [
                {
                    "round": r.round_index,
                    "decision": r.decision,
                    "reason": r.reason,
                    "rationale": r.rationale,
                    "predicted_fixes": r.predicted_fixes,
                    "risk_tasks": r.risk_tasks,
                    "dev_score_before": r.before_score,
                    "dev_score_after": r.after_score,
                    "dev_tokens_before": r.before_tokens,
                    "dev_tokens_after": r.after_tokens,
                }
                for r in history
            ],
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (round_dir / "ready.json").write_text(
        json.dumps({
            "tag": tag,
            "module": module,
            "tasks": [o.task for o in outcomes],
        }, indent=2) + "\n",
        encoding="utf-8",
    )
    return round_dir
