"""Final comparison: the original agent vs each improved arm, on the test set.

The test tasks are touched exactly once, after all improvement has finished, so
no test feedback can enter the loop. Results are reported as one table alongside
actual token usage and cost.
"""

from __future__ import annotations

import json
from pathlib import Path

from . import harness as H
from .evolve import ArmResult

RESULTS_DIR = H.ROOT / "rsi" / "results"


def _condition_rows(
    arms: dict[str, ArmResult], test_tasks: list[str], tag: str
) -> list[H.RunResult]:
    """Evaluate original, then each arm's final retained version, on test."""
    runs: list[H.RunResult] = []

    runs.append(
        H.run_condition(
            condition="original",
            module_dir=None,
            tasks=test_tasks,
            task_set="test",
            job_name=f"{tag}-original",
            reuse_clean=True,
        )
    )

    for module, arm in arms.items():
        if arm.final_dir is None:
            continue
        runs.append(
            H.run_condition(
                condition=f"{module}-final",
                module_dir=arm.final_dir,
                tasks=test_tasks,
                task_set="test",
                job_name=f"{tag}-{module}-final",
                reuse_clean=True,
            )
        )
    return runs


def render_table(runs: list[H.RunResult]) -> str:
    header = (
        "| Version | Test solved | Test success rate | Mean reward | Tokens | Cost (USD) |\n"
        "| --- | --- | --- | --- | --- | --- |"
    )
    rows = []
    for run in runs:
        total = len(run.outcomes) or 1
        rows.append(
            f"| {run.condition} | {run.solved_count}/{len(run.outcomes)} "
            f"| {run.solved_count / total:.1%} "
            f"| {run.mean_reward:.3f} "
            f"| {run.total_tokens:,} "
            f"| ${run.actual_cost_usd:.4f} |"
        )
    return "\n".join([header, *rows])


def render_rounds(arms: dict[str, ArmResult]) -> str:
    lines = []
    for module, arm in arms.items():
        lines.append(f"\n### Module {module}\n")
        lines.append("| Round | Decision | Dev before | Dev after | Proposal tokens | Reason |")
        lines.append("| --- | --- | --- | --- | --- | --- |")
        for record in arm.rounds:
            after = "—" if record.after_score is None else f"{record.after_score:.3f}"
            lines.append(
                f"| {record.round_index} | {record.decision} "
                f"| {record.before_score:.3f} | {after} "
                f"| {record.proposal_tokens:,} | {record.reason} |"
            )
    return "\n".join(lines)


def write_results(
    *,
    arms: dict[str, ArmResult],
    dev_baseline: dict[str, float],
    runs: list[H.RunResult],
    tag: str,
) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "tag": tag,
        "dev_baseline": dev_baseline,
        "arms": {module: arm.to_json() for module, arm in arms.items()},
        "test_runs": [
            {
                "condition": run.condition,
                "solved": run.solved_count,
                "n": len(run.outcomes),
                "mean_reward": run.mean_reward,
                "total_tokens": run.total_tokens,
                "cost_usd": run.actual_cost_usd,
                "clean_cost_usd": run.total_cost_usd,
                "retry_tokens": run.retry_tokens,
                "retry_cost_usd": run.retry_cost_usd,
                "outcomes": [
                    {
                        "task": o.task,
                        "reward": o.reward,
                        "tool_calls": o.tool_calls,
                        "total_tokens": o.total_tokens,
                        "cost_usd": o.cost_usd,
                        "budget_stop": o.exhausted_by,
                        "exception": o.exception,
                    }
                    for o in run.outcomes
                ],
            }
            for run in runs
        ],
    }
    path = RESULTS_DIR / f"{tag}.json"
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("test_runs") and existing != payload:
            raise FileExistsError(f"completed test results cannot be overwritten: {path}")
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)

    table_path = RESULTS_DIR / f"{tag}-table.md"
    table_path.write_text(
        "# RSI experiment results\n\n## Test set\n\n"
        + render_table(runs)
        + "\n\n## Improvement rounds (development set)\n"
        + render_rounds(arms)
        + "\n",
        encoding="utf-8",
    )
    return table_path
