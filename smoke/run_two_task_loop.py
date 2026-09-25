"""One-round, two-task development dry run for one arm at a time."""

from __future__ import annotations

import argparse

from open_research_RSI.rsi import harness as H
from open_research_RSI.rsi.evolve import improve_module


TASKS = [
    "redis-transactions-go-ape-292",
    "grep-backreferences-go-mouse-358",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("module", choices=("tools", "exec"))
    parser.add_argument("--tag", required=True)
    args = parser.parse_args()

    dev = set(H.load_manifest()["dev"])
    if not set(TASKS) <= dev:
        raise ValueError("dry-run tasks must be in the development split")

    H.MAX_TOOL_CALLS = 80
    H.MAX_TOKENS = 2_000_000
    H.N_CONCURRENT = 2
    print(H.preflight_docker(), flush=True)
    print(H.preflight_model(), flush=True)
    baseline = H.run_condition(
        condition="original", module_dir=None, tasks=TASKS,
        task_set="dev", job_name=f"{args.tag}-dev-original",
        reuse_clean=True,
    )
    print(baseline.summary(), flush=True)
    arm = improve_module(
        args.module, TASKS, baseline=baseline, tag=args.tag, rounds=1,
        budget_tokens=10_000_000,
        checkpoint=H.ROOT / "rsi" / "checkpoints" / args.tag / f"{args.module}.json",
    )
    for record in arm.rounds:
        print(f"{args.module} round {record.round_index}: {record.decision}: {record.reason}")
    print(arm.dev_result.summary())


if __name__ == "__main__":
    main()
