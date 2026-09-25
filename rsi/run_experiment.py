"""Drive the whole RSI experiment end to end.

    python -m open_research_RSI.rsi.run_experiment --help

Runs, in order:
  1. one original-agent development baseline shared by both arms
  2. module A and module B improvement loops on the development set
  3. one final evaluation of original / A / B on the held-out test set
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from . import evaluate, harness as H
from .evolve import ArmResult, RoundRecord, implementation_identity, improve_module, verify_version

MODULES = ("tools", "exec")


def save_experiment_identity(
    *, tag: str, modules: list[str], dev_tasks: list[str], test_tasks: list[str],
    rounds: int, budget_tokens: int,
) -> Path:
    payload = {
        "tag": tag,
        "modules": modules,
        "dev_tasks": dev_tasks,
        "test_tasks": test_tasks,
        "benchmark_commit": H.load_manifest()["commit"],
        "model": H.MODEL,
        "rounds": rounds,
        "budget_tokens": budget_tokens,
        "max_tool_calls": H.MAX_TOOL_CALLS,
        "max_tokens": H.MAX_TOKENS,
        "n_concurrent": H.N_CONCURRENT,
        "implementation": implementation_identity(),
    }
    directory = H.ROOT / "rsi" / "results" / "identities"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{tag}.json"
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != payload:
            raise ValueError(f"experiment identity changed for tag {tag}; use a new tag")
    else:
        with path.open("x", encoding="utf-8") as record:
            json.dump(payload, record, indent=2, ensure_ascii=False)
            record.write("\n")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default="main", help="name for this experiment run")
    parser.add_argument(
        "--modules", nargs="+", default=list(MODULES), choices=list(MODULES)
    )
    parser.add_argument("--rounds", type=int, default=H.ROUNDS)
    parser.add_argument("--budget-tokens", type=int, default=H.TOTAL_TOKEN_BUDGET)
    parser.add_argument("--max-tool-calls", type=int, default=H.MAX_TOOL_CALLS)
    parser.add_argument("--max-tokens", type=int, default=H.MAX_TOKENS)
    parser.add_argument(
        "--dev-limit", type=int, default=0, help="use only the first N dev tasks (debugging)"
    )
    parser.add_argument(
        "--test-limit", type=int, default=0, help="use only the first N test tasks (debugging)"
    )
    parser.add_argument(
        "--skip-test", action="store_true", help="stop after the improvement loops"
    )
    parser.add_argument(
        "--arms-json",
        type=Path,
        default=None,
        help="reuse finished arms from a previous results JSON instead of re-running them",
    )
    parser.add_argument(
        "--final-only",
        action="store_true",
        help="skip all improvement and run only the final test comparison",
    )
    parser.add_argument(
        "--arm-final",
        action="append",
        default=[],
        metavar="MODULE=DIR",
        help=(
            "for --final-only, the retained version of an arm that was improved in an "
            "earlier invocation, e.g. tools=open_research_RSI/rsi/versions/tools/v2"
        ),
    )
    args = parser.parse_args(argv)

    H.MAX_TOOL_CALLS = args.max_tool_calls
    H.MAX_TOKENS = args.max_tokens

    # Refuse to start hours of work when either external dependency is down.
    print(H.preflight_docker())
    print(H.preflight_model())

    manifest = H.load_manifest()
    dev_tasks = manifest["dev"]
    test_tasks = manifest["test"]
    if args.dev_limit:
        dev_tasks = dev_tasks[: args.dev_limit]
    if args.test_limit:
        test_tasks = test_tasks[: args.test_limit]

    identity_path = save_experiment_identity(
        tag=args.tag, modules=args.modules, dev_tasks=dev_tasks,
        test_tasks=test_tasks, rounds=args.rounds, budget_tokens=args.budget_tokens,
    )
    print(f"identity={identity_path}")

    print(f"tag={args.tag}  dev={len(dev_tasks)}  test={len(test_tasks)}")
    print(
        f"model={H.MODEL}  rounds={args.rounds}  "
        f"per-task caps: {H.MAX_TOOL_CALLS} tool calls / {H.MAX_TOKENS:,} tokens"
    )

    arms: dict[str, ArmResult] = {}
    baseline_score = None
    if args.final_only:
        # An arm finished in an earlier, interrupted invocation still counts:
        # its retained version is on disk and the test set has never seen it.
        for spec in args.arm_final:
            name, _, directory = spec.partition("=")
            if name not in MODULES or not directory:
                raise ValueError(f"invalid --arm-final value: {spec}")
            arm = ArmResult(module=name)
            arm.final_dir = (H.WORKSPACE / directory).resolve()
            verify_version(arm.final_dir, name, args.tag)
            arm.final_source = (arm.final_dir / "index.ts").read_text(encoding="utf-8")
            state_path = H.ROOT / "rsi" / "checkpoints" / args.tag / f"{name}.json"
            state = json.loads(state_path.read_text(encoding="utf-8"))
            if state.get("pending") or len(state.get("rounds", [])) != args.rounds:
                raise ValueError(f"arm {name} has not completed {args.rounds} rounds")
            if state.get("final_dir") != str(arm.final_dir):
                raise ValueError(f"arm {name} does not match its checkpoint")
            if state.get("final_sha256") != hashlib.sha256(arm.final_source.encode()).hexdigest():
                raise ValueError(f"arm {name} source changed since its checkpoint")
            arm.rounds = [RoundRecord(**entry) for entry in state["rounds"]]
            if any(r.decision not in ("accepted", "accepted_efficiency", "rejected") for r in arm.rounds):
                raise ValueError(f"arm {name} has an unfinished round")
            arm.proposal_tokens = state["proposal_tokens"]
            arm.evaluation_tokens = state["evaluation_tokens"]
            condition = state["current_condition"]
            arm.dev_result = H.run_condition(
                condition=condition,
                module_dir=None if condition == "original" else arm.final_dir,
                tasks=dev_tasks, task_set="dev", job_name=state["current_job"],
                reuse_clean=True,
            )
            arms[name] = arm
        print(f"final-only: arms = {', '.join(f'{k} -> {v.final_dir.name}' for k, v in arms.items())}")

    if args.arms_json and args.arms_json.is_file():
        arms = load_arms(args.arms_json, tag=args.tag, dev_tasks=dev_tasks, rounds=args.rounds)
        scores = json.loads(args.arms_json.read_text(encoding="utf-8"))["dev_baseline"]
        baseline_score = next(iter(scores.values())) if scores else None
        print(f"reusing arms from {args.arms_json}")

    pending_modules = [
        module for module in (() if args.final_only else args.modules) if module not in arms
    ]
    baseline = None
    if pending_modules:
        print("\n=== shared original development baseline ===")
        baseline = H.run_condition(
            condition="original",
            module_dir=None,
            tasks=dev_tasks,
            task_set="dev",
            job_name=f"{args.tag}-dev-original",
            reuse_clean=True,
        )
        print(f"  {baseline.summary()}")
        baseline_score = baseline.mean_reward
    elif baseline_score is None and args.final_only:
        baseline = H.run_condition(
            condition="original", module_dir=None, tasks=dev_tasks,
            task_set="dev", job_name=f"{args.tag}-dev-original", reuse_clean=True,
        )
        baseline_score = baseline.mean_reward

    for module in pending_modules:
        if module in arms:
            continue
        print(f"\n=== improving module: {module} ===")
        arms[module] = improve_module(
            module,
            dev_tasks,
            baseline=baseline,
            tag=args.tag,
            rounds=args.rounds,
            budget_tokens=args.budget_tokens,
            checkpoint=H.ROOT / "rsi" / "checkpoints" / args.tag / f"{module}.json",
        )
        print(f"[{module}] final: {arms[module].dev_result.summary()}")
        # Save the completed arm before starting the next one. A later provider
        # outage must not discard a finished three-round improvement history.
        evaluate.write_results(
            arms=arms,
            dev_baseline={m: baseline_score for m in arms},
            runs=[],
            tag=args.tag,
        )

    if args.skip_test:
        evaluate.write_results(
            arms=arms,
            dev_baseline={
                m: baseline_score
                for m, a in arms.items()
            },
            runs=[],
            tag=args.tag,
        )
        print("\nskipped the test evaluation (--skip-test)")
        return 0

    if not args.dev_limit and not args.test_limit and set(args.modules) == set(MODULES):
        if set(arms) != set(MODULES) or any(len(arm.rounds) != args.rounds for arm in arms.values()):
            raise RuntimeError("formal test requires both completed improvement arms")

    print("\n=== final test evaluation ===")
    runs = evaluate._condition_rows(arms, test_tasks, tag=args.tag)
    for run in runs:
        print(f"  {run.summary()}")

    table_path = evaluate.write_results(
        arms=arms,
        dev_baseline={
            m: baseline_score for m, a in arms.items()
        },
        runs=runs,
        tag=args.tag,
    )
    print("\n" + evaluate.render_table(runs))
    print(f"\nwrote {table_path}")
    return 0


def load_arms(path: Path, *, tag: str, dev_tasks: list[str], rounds: int) -> dict[str, ArmResult]:
    """Rebuild ArmResult objects from a results JSON, for resuming."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("tag") != tag:
        raise ValueError(f"results tag does not match requested tag: {path}")
    arms: dict[str, ArmResult] = {}
    for module, data in payload.get("arms", {}).items():
        arm = ArmResult(module=module)
        arm.final_dir = Path(data["final_dir"]) if data.get("final_dir") else None
        if arm.final_dir is None:
            raise ValueError(f"missing final version for {module}")
        verify_version(arm.final_dir, module, tag)
        arm.final_source = (arm.final_dir / "index.ts").read_text(encoding="utf-8")
        arm.proposal_tokens = int(data.get("proposal_tokens") or 0)
        arm.evaluation_tokens = int(data.get("evaluation_tokens") or 0)
        arm.rounds = [RoundRecord(**entry) for entry in data.get("rounds", [])]
        if len(arm.rounds) != rounds or any(r.decision not in ("accepted", "accepted_efficiency", "rejected") for r in arm.rounds):
            raise ValueError(f"arm {module} has {len(arm.rounds)} rounds, expected {rounds}")
        checkpoint = H.ROOT / "rsi" / "checkpoints" / tag / f"{module}.json"
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        if (state["final_dir"] != str(arm.final_dir)
                or state["dev_tasks"] != dev_tasks
                or state.get("pending")
                or state.get("round_limit") != rounds):
            raise ValueError(f"arm {module} does not match its checkpoint")
        condition = state["current_condition"]
        arm.dev_result = H.run_condition(
            condition=condition,
            module_dir=None if condition == "original" else arm.final_dir,
            tasks=dev_tasks, task_set="dev", job_name=state["current_job"],
            reuse_clean=True,
        )
        arms[module] = arm
    return arms


if __name__ == "__main__":
    sys.exit(main())
