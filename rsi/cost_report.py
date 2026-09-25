"""Token and cost breakdown, recomputed from the raw pi transcripts.

Answers "where did the money go" at the level of pi's own usage record, which
splits every model call into four buckets with different prices:

    input       uncached prompt tokens      (the expensive one)
    cacheRead   prompt tokens served from cache  (~50x cheaper on DeepSeek)
    cacheWrite  prompt tokens written to cache
    output      generated tokens

`input` in pi's Usage is the UNCACHED portion; the adapter's `n_input_tokens`
is the sum of all three prompt buckets. Mixing the two up is what produced the
double-count this script exists to avoid.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

from . import harness as H

BUCKETS = ("input", "cacheRead", "cacheWrite", "output")


@dataclass
class Bucket:
    tokens: int = 0
    cost: float = 0.0

    def add(self, tokens: int, cost: float) -> None:
        self.tokens += tokens
        self.cost += cost


@dataclass
class Breakdown:
    buckets: dict[str, Bucket] = field(
        default_factory=lambda: {name: Bucket() for name in BUCKETS}
    )
    calls: int = 0

    @property
    def tokens(self) -> int:
        return sum(b.tokens for b in self.buckets.values())

    @property
    def cost(self) -> float:
        return sum(b.cost for b in self.buckets.values())

    def merge(self, other: "Breakdown") -> None:
        for name, bucket in other.buckets.items():
            self.buckets[name].add(bucket.tokens, bucket.cost)
        self.calls += other.calls


def scan_transcript(path: Path) -> Breakdown:
    """Sum every assistant message's usage in one pi JSONL transcript."""
    result = Breakdown()
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if '"usage"' not in line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Only `message_end` carries the settled totals for a model call.
        # `message_update` events also carry a usage field, but it is the
        # cumulative stream delta, so counting them doubles every figure.
        if event.get("type") != "message_end":
            continue
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            continue
        usage = message.get("usage") or {}
        if not usage:
            continue
        result.calls += 1
        cost = usage.get("cost") or {}
        if not isinstance(cost, dict):
            cost = {}
        for name in BUCKETS:
            result.buckets[name].add(
                int(usage.get(name) or 0), float(cost.get(name) or 0.0)
            )
    return result


def scan_run(job_dir: Path) -> Breakdown:
    total = Breakdown()
    for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        total.merge(scan_transcript(trial_dir / "agent" / "pi.jsonl"))
    return total


def scan_all(tag: str | None) -> dict[str, Breakdown]:
    runs = H.RUNS_DIR
    groups: dict[str, Breakdown] = {}
    def matches_run(name: str) -> bool:
        return tag is None or name == tag or name.startswith(tag + "-")

    for job_dir in sorted(p for p in runs.iterdir() if p.is_dir()):
        if job_dir.name == "_contaminated":
            for nested in sorted(p for p in job_dir.iterdir() if p.is_dir()):
                if not matches_run(nested.name):
                    continue
                groups[f"_contaminated/{nested.name}"] = scan_run(nested)
            continue
        if not matches_run(job_dir.name):
            continue
        groups[job_dir.name] = scan_run(job_dir)

    failed = runs / "_failed_attempts"
    if failed.is_dir():
        for job_dir in sorted(p for p in failed.iterdir() if p.is_dir()):
            if not matches_run(job_dir.name):
                continue
            for attempt in sorted(p for p in job_dir.iterdir() if p.is_dir()):
                groups[f"retry:{job_dir.name}/{attempt.name}"] = scan_run(attempt)

    # Evolver transcripts live under the evidence tree, not under runs/.
    evidence = H.ROOT / "rsi" / "evidence"
    if evidence.is_dir():
        for path in sorted(evidence.rglob("evolver.jsonl")):
            rel = path.relative_to(evidence)
            if tag is not None and rel.parts[0] != tag:
                continue
            label = "evolver:" + str(rel.parent).replace("-workspace", "")
            groups[label] = scan_transcript(path)
    return groups


def render(groups: dict[str, Breakdown], title: str) -> str:
    lines = [f"# Token and cost breakdown — {title}", ""]
    lines.append(
        "| Run | Calls | Uncached in | Cache read | Cache write | Output | "
        "Total tokens | Cost USD |"
    )
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    grand = Breakdown()
    for name, b in groups.items():
        if b.calls == 0:
            continue
        grand.merge(b)
        lines.append(
            f"| {name} | {b.calls:,} | {b.buckets['input'].tokens:,} "
            f"| {b.buckets['cacheRead'].tokens:,} | {b.buckets['cacheWrite'].tokens:,} "
            f"| {b.buckets['output'].tokens:,} | {b.tokens:,} | ${b.cost:.4f} |"
        )
    lines.append(
        f"| **TOTAL** | {grand.calls:,} | {grand.buckets['input'].tokens:,} "
        f"| {grand.buckets['cacheRead'].tokens:,} | {grand.buckets['cacheWrite'].tokens:,} "
        f"| {grand.buckets['output'].tokens:,} | {grand.tokens:,} | **${grand.cost:.2f}** |"
    )

    lines += ["", "## Where the money went", ""]
    total_cost = grand.cost or 1.0
    lines.append("| Bucket | Tokens | Share of tokens | Cost USD | Share of cost | $/M tokens |")
    lines.append("| --- | --- | --- | --- | --- | --- |")
    for name in BUCKETS:
        bucket = grand.buckets[name]
        per_m = (bucket.cost / bucket.tokens * 1_000_000) if bucket.tokens else 0.0
        lines.append(
            f"| {name} | {bucket.tokens:,} | {bucket.tokens / (grand.tokens or 1):.1%} "
            f"| ${bucket.cost:.4f} | {bucket.cost / total_cost:.1%} | ${per_m:.3f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", default=None, help="only runs whose name starts with this")
    parser.add_argument("--out", default=None, help="markdown output path")
    args = parser.parse_args()

    label = args.tag or "all runs"
    groups = scan_all(args.tag)
    text = render(groups, label)
    print(text)
    if args.out:
        Path(args.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
