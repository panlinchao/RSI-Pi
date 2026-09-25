"""Run one development task with a diagnostic extension in a separate tag."""

from __future__ import annotations

import argparse
from pathlib import Path

from open_research_RSI.rsi import harness as H
from open_research_RSI.rsi.evolve import stage_version


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("module", choices=("tools", "exec"))
    parser.add_argument("--tag", required=True)
    parser.add_argument("--task", default="redis-transactions-go-ape-292")
    args = parser.parse_args()

    if args.task not in H.load_manifest()["dev"]:
        raise ValueError("activation check must use a development task")
    source = (H.ROOT / "smoke" / "probes" / f"{args.module}.ts").read_text(encoding="utf-8")
    version = stage_version(args.tag, args.module, 1, source)
    H.MAX_TOOL_CALLS = 4
    H.MAX_TOKENS = 100_000
    run = H.run_condition(
        condition=f"{args.module}-activation", module_dir=version,
        tasks=[args.task], task_set="dev", job_name=f"{args.tag}-{args.module}-activation",
        reuse_clean=True,
    )
    outcome = run.outcomes[0]
    stderr = (Path(outcome.trial_dir) / "agent" / "pi.stderr").read_text(encoding="utf-8")
    prefix = f"RSI_{args.module.upper()}_PROBE_"
    markers = [line for line in stderr.splitlines() if line.startswith(prefix)]
    if not any("LOADED" in marker for marker in markers):
        raise RuntimeError(f"{args.module} extension did not report activation")
    if not any("TOOL_CALL:" in marker for marker in markers):
        raise RuntimeError(f"{args.module} extension did not observe a tool call")
    print(run.summary())
    print("\n".join(markers[:8]))


if __name__ == "__main__":
    main()
