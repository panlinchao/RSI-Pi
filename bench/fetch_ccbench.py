"""Materialize a fixed CCBench task split for the RSI experiment.

Why this exists: the task directories must be byte-identical across every run of
the experiment, and their executable bits must survive materialization (the
CCBench verifier checks `-x /app/.codecrafters/compile.sh` and bails out without
writing a reward file if it is missing). `git archive` restores modes from the
tree, so we materialize with it instead of downloading files one by one.

Materialization also patches `task.toml`:
  - `build_timeout_sec` is raised, because a CCBench image build compiles a
    toolchain (`go install std`, `uv sync`, ...) on a consumer network and the
    stock 600 s is not a meaningful budget for us.

The selected split is written to `manifest.json` and never recomputed with a
different seed: the dev/test boundary has to be frozen before the experiment.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
import subprocess
import sys
from pathlib import Path

BENCH_DIR = Path(__file__).resolve().parent
DEFAULT_REPO = BENCH_DIR / "ccbench" / "repo"
DEFAULT_TASKS_DIR = BENCH_DIR / "ccbench" / "tasks"
DEFAULT_MANIFEST = BENCH_DIR / "manifest.json"

# Pinned so a rerun months later reproduces the same task bytes.
CCBENCH_COMMIT = "da3895ae2f5f072ac6152c32349546344802f722"
CCBENCH_URL = "https://github.com/codecrafters-io/ccbench.git"

# Bumped from the stock 600 s. See module docstring.
BUILD_TIMEOUT_SEC = 2400.0

# `<problem>-<language>-<animal>-<number>`; the problem is everything between
# the start and the language token, and is what we stratify the split over.
TASK_RE = re.compile(r"^(?P<problem>.+?)-(?P<language>[a-z]+)-(?P<animal>[a-z]+)-(?P<number>\d+)$")


def run(cmd: list[str], **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=True, text=True, **kwargs)


def ensure_repo(repo: Path) -> None:
    """Blobless, no-checkout clone pinned to CCBENCH_COMMIT."""
    if (repo / ".git").is_dir():
        head = run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True).stdout.strip()
        if head != CCBENCH_COMMIT:
            print(f"repo at {repo} is on {head}, fetching pinned commit", file=sys.stderr)
            run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", CCBENCH_COMMIT])
            run(["git", "-C", str(repo), "checkout", "--detach", CCBENCH_COMMIT])
        return

    repo.parent.mkdir(parents=True, exist_ok=True)
    print(f"cloning {CCBENCH_URL} -> {repo}", file=sys.stderr)
    # Blobless and unchecked-out: we only ever read trees through `git archive`,
    # which fetches exactly the blobs it needs and no others.
    run(["git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
         CCBENCH_URL, str(repo)])
    head = run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True).stdout.strip()
    if head != CCBENCH_COMMIT:
        run(["git", "-C", str(repo), "fetch", "--depth", "1", "origin", CCBENCH_COMMIT])
        run(["git", "-C", str(repo), "checkout", "--detach", CCBENCH_COMMIT])


def list_tasks(repo: Path) -> list[str]:
    out = run(
        ["git", "-C", str(repo), "ls-tree", "-d", "--name-only", "HEAD:tasks"],
        capture_output=True,
    ).stdout
    return sorted(line.strip() for line in out.splitlines() if line.strip())


def parse_task(name: str) -> dict[str, str] | None:
    m = TASK_RE.match(name)
    return m.groupdict() if m else None


def stratified_split(
    tasks: list[str], language: str, n_dev: int, n_test: int, seed: int
) -> tuple[list[str], list[str], list[str]]:
    """Deal tasks out family by family so dev and test share the same mix.

    Each family is shuffled with `seed`, then families are visited round-robin
    emitting one task at a time. The resulting stream alternates families, so
    taking its head gives a dev set whose family histogram matches the test set
    rather than, say, all eight kafka tasks landing on one side.
    """
    by_family: dict[str, list[str]] = {}
    for name in tasks:
        parsed = parse_task(name)
        if parsed is None or parsed["language"] != language:
            continue
        by_family.setdefault(parsed["problem"], []).append(name)

    rng = random.Random(seed)
    queues: list[list[str]] = []
    for family in sorted(by_family):
        members = sorted(by_family[family])
        rng.shuffle(members)
        queues.append(members)

    stream: list[str] = []
    while any(queues):
        for queue in queues:
            if queue:
                stream.append(queue.pop(0))

    needed = n_dev + n_test
    if len(stream) < needed:
        raise SystemExit(
            f"language {language!r} has only {len(stream)} tasks, need {needed}"
        )
    return stream[:n_dev], stream[n_dev:needed], stream[needed:]


def materialize(repo: Path, name: str, tasks_dir: Path) -> None:
    dest = tasks_dir / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)

    # `git archive` writes the tree with its recorded modes, which is the whole
    # point: the per-file download the first attempt used dropped +x and made
    # every trial fail its compile-script precondition.
    #
    # The `<tree-ish>:<path>` form relocates the subdirectory to the archive
    # root, so entries arrive as `task.toml`, `environment/...` already --
    # no strip-components.
    archive = subprocess.run(
        ["git", "-C", str(repo), "archive", "--format=tar", f"HEAD:tasks/{name}"],
        check=True, capture_output=True,
    )
    subprocess.run(["tar", "-x", "-C", str(dest)], input=archive.stdout, check=True)

    patch_task_toml(dest / "task.toml")


def patch_task_toml(path: Path) -> None:
    if not path.is_file():
        raise SystemExit(f"missing task.toml: {path}")
    text = path.read_text(encoding="utf-8")
    new, count = re.subn(
        r"^build_timeout_sec\s*=\s*[\d.]+",
        f"build_timeout_sec = {BUILD_TIMEOUT_SEC}",
        text,
        flags=re.MULTILINE,
    )
    if count == 0:
        new = text.rstrip("\n") + f"\nbuild_timeout_sec = {BUILD_TIMEOUT_SEC}\n"
    path.write_text(new, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=DEFAULT_REPO)
    parser.add_argument("--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--language", default="go")
    parser.add_argument("--n-dev", type=int, default=10)
    parser.add_argument("--n-test", type=int, default=10)
    parser.add_argument("--seed", type=int, default=7160)
    parser.add_argument("--force", action="store_true", help="re-materialize existing tasks")
    parser.add_argument(
        "--rebalance-test",
        action="store_true",
        help=(
            "keep the development split and rebuild the test split out of the tasks "
            "the development split never used, taking the untouched reserve first and "
            "topping up from the previous test split in manifest order. Purely "
            "mechanical: no task is chosen because of how any agent scored on it."
        ),
    )
    args = parser.parse_args()

    # Reusing an existing manifest is deliberate: the split is frozen once, and
    # re-running with a different seed must not silently move the test boundary.
    if args.manifest.is_file() and not args.force and not args.rebalance_test:
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        if manifest.get("seed") != args.seed or manifest.get("language") != args.language:
            raise SystemExit(
                f"{args.manifest} already freezes a different split "
                f"(language={manifest.get('language')}, seed={manifest.get('seed')}). "
                "Pass --force to replace it."
            )
        print(f"reusing frozen split from {args.manifest}")
    else:
        ensure_repo(args.repo)
        all_tasks = list_tasks(args.repo)
        dev, test, reserve = stratified_split(
            all_tasks, args.language, args.n_dev, args.n_test, args.seed
        )
        if args.rebalance_test:
            # The first split put tasks in test that the stock agent then solved
            # 15/15, leaving no headroom to measure anything. This keeps the
            # development split byte-identical and rebuilds test from the pool
            # development never used: the untouched reserve first, then the
            # previous test split in manifest order. No task is chosen because of
            # how any agent scored on it.
            rebalanced = list(reserve[: args.n_test])
            for candidate in test:
                if len(rebalanced) >= args.n_test:
                    break
                if candidate not in rebalanced:
                    rebalanced.append(candidate)
            reserve = [t for t in (*test, *reserve) if t not in rebalanced]
            test = rebalanced

        manifest = {
            "source": CCBENCH_URL,
            "commit": CCBENCH_COMMIT,
            "language": args.language,
            "seed": args.seed,
            "n_dev": args.n_dev,
            "n_test": args.n_test,
            "build_timeout_sec": BUILD_TIMEOUT_SEC,
            "dev": dev,
            "test": test,
            "reserve": reserve,
        }
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        print(f"froze split -> {args.manifest}")

    # A fresh checkout has the frozen manifest but no downloaded benchmark.
    # Fetch the pinned repository before materializing the first missing task.
    task_names = [name for split_name in ("dev", "test") for name in manifest[split_name]]
    if args.force or any(not (args.tasks_dir / name / "task.toml").is_file() for name in task_names):
        ensure_repo(args.repo)

    for split_name in ("dev", "test"):
        for name in manifest[split_name]:
            dest = args.tasks_dir / name
            # Presence of task.toml is the integrity marker, not mere directory
            # existence: an interrupted materialization leaves a directory
            # behind, and a half-written task would silently load as a
            # different problem than the one the split was frozen against.
            if (dest / "task.toml").is_file() and not args.force:
                continue
            if dest.is_dir():
                print(f"  re-materializing damaged task {name}", file=sys.stderr)
            materialize(args.repo, name, args.tasks_dir)
            print(f"  materialized [{split_name}] {name}")

    print(f"\ndev ({len(manifest['dev'])}):")
    for name in manifest["dev"]:
        print(f"  {name}")
    print(f"\ntest ({len(manifest['test'])}):")
    for name in manifest["test"]:
        print(f"  {name}")
    print(f"\nreserve pool: {len(manifest['reserve'])} tasks")


if __name__ == "__main__":
    main()
