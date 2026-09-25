"""Improve one module using the shared original-agent development baseline."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import evidence as E
from . import evolver as EV
from . import harness as H
from . import signals as S

VERSIONS_DIR = H.ROOT / "rsi" / "versions"

# A tie on success rate counts as an improvement only if the candidate cuts
# total development API cost by at least this fraction.
#
# Deliberately conservative. Per-task token use on these tasks spans 0.74M to
# 8.09M, and an observed rerun of the same two tasks moved by 90%, so a 10%
# margin sits inside run-to-run noise: it would accept a candidate on variance
# alone. A false accept contaminates the arm for every later round, whereas a
# false reject only forfeits a round, so the threshold errs high.
EFFICIENCY_GAIN = 0.20

# APIs that belong to the *other* arm. See modules/<name>/SPEC.md.
FORBIDDEN = {
    "tools": ("turn_end", "agent_end", "agent_settled"),
    "exec": ("registerTool(",),
}


# --------------------------------------------------------------------------
# Evolver invocation
# --------------------------------------------------------------------------


def api_key() -> str:
    """The key the evolver's pi process uses, read from the environment."""
    import os

    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API")
    if not key:
        raise SystemExit("Set DEEPSEEK_API_KEY in the environment before running.")
    return key


def boundary_violation(module: str, source: str) -> str | None:
    """Reject a candidate that reaches into the other arm's axis."""
    for token in FORBIDDEN.get(module, ()):
        if token in source:
            return f"reaches outside this module's surface: uses `{token}`"
    if module == "exec" and re.search(r"\bregisterTool\s*\(", source):
        return "reaches outside this module's surface: registers a tool"
    if module == "tools" and re.search(r"\b(?:block|terminate)\s*:", source):
        return "reaches outside this module's surface: blocks or terminates a tool call"
    if "export default" not in source:
        return "no default export; pi could not load it as an extension"
    return None


def syntax_violation(source: str) -> str | None:
    """Parse TypeScript with the same pinned Node runtime as the evolver."""
    with tempfile.TemporaryDirectory(prefix="rsi-candidate-check-") as temp:
        candidate = Path(temp) / "index.ts"
        candidate.write_text(source, encoding="utf-8")
        result = subprocess.run(
            [
                "docker", "run", "--rm", "--network", "none",
                "-v", f"{temp}:/candidate:ro", EV.EVOLVER_IMAGE,
                "node", "--experimental-strip-types", "--check", "/candidate/index.ts",
            ],
            text=True, capture_output=True, timeout=45,
        )
    if result.returncode == 0:
        return None
    detail = (result.stderr or result.stdout).strip()
    if "SyntaxError" in detail or "ERR_INVALID_TYPESCRIPT_SYNTAX" in detail:
        return f"TypeScript syntax invalid: {detail[-300:]}"
    raise RuntimeError(f"candidate syntax check could not run: {detail[-300:]}")


def implementation_identity() -> dict[str, str]:
    """Identify the committed implementation and the actual evolver image."""
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=normal"],
        cwd=H.ROOT, text=True, capture_output=True, timeout=30,
    )
    if status.returncode != 0 or status.stdout.strip():
        raise RuntimeError("commit the implementation before starting or resuming a run")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=H.ROOT, text=True, capture_output=True, timeout=30,
    )
    if revision.returncode != 0 or not revision.stdout.strip():
        raise RuntimeError("implementation has no Git commit")
    image = subprocess.run(
        ["docker", "image", "inspect", EV.EVOLVER_IMAGE, "--format", "{{.Id}}"],
        text=True, capture_output=True, timeout=30,
    )
    if image.returncode != 0 or not image.stdout.strip():
        raise RuntimeError(f"evolver image is unavailable: {EV.EVOLVER_IMAGE}")
    return {"git_commit": revision.stdout.strip(), "evolver_image_id": image.stdout.strip()}


# --------------------------------------------------------------------------
# Stage 1: analysis
# --------------------------------------------------------------------------

def render_attribution(history: list["RoundRecord"], current: H.RunResult) -> str:
    """Did last round's change do what it said it would?

    AHE carries a change-attribution report into every iteration so the proposer
    can KEEP, IMPROVE, or ROLLBACK+PIVOT instead of layering a second guess on
    top of a first one that never worked.
    """
    decided = [r for r in history if r.predicted_fixes]
    if not decided:
        return "(no previous change carried a prediction)"

    by_task = {o.task: o for o in current.outcomes}
    lines = []
    for record in decided:
        verdicts = []
        for task in record.predicted_fixes:
            outcome = by_task.get(task)
            if outcome is None:
                verdicts.append(f"{task}: not in this task set")
            elif outcome.solved:
                verdicts.append(f"{task}: SOLVED (prediction held)")
            else:
                verdicts.append(f"{task}: still FAILING (prediction did not hold)")
        for task in record.risk_tasks:
            outcome = by_task.get(task)
            if outcome is not None and not outcome.solved:
                verdicts.append(f"{task}: REGRESSED (was flagged as a risk)")
        lines.append(
            f"- round {record.round_index} ({record.decision}) predicted: "
            + "; ".join(verdicts)
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Version staging
# --------------------------------------------------------------------------


def stage_version(tag: str, module: str, version: int, source: str) -> Path:
    """Stage a version once; a repeated call must have identical contents."""
    target = VERSIONS_DIR / tag / module / f"v{version}"
    spec = (H.MODULES_DIR / module / "SPEC.md").read_bytes()
    source_bytes = source.encode("utf-8")
    identity = {
        "tag": tag,
        "module": module,
        "version": version,
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "spec_sha256": hashlib.sha256(spec).hexdigest(),
    }
    if target.exists():
        try:
            matches = (
                (target / "index.ts").read_bytes() == source_bytes
                and (target / "SPEC.md").read_bytes() == spec
                and json.loads((target / "artifact.json").read_text(encoding="utf-8")) == identity
            )
        except (OSError, ValueError):
            matches = False
        if not matches:
            raise FileExistsError(f"version already exists with different contents: {target}")
        return target
    target.mkdir(parents=True)
    (target / "index.ts").write_bytes(source_bytes)
    (target / "SPEC.md").write_bytes(spec)
    (target / "artifact.json").write_text(
        json.dumps(identity, indent=2) + "\n", encoding="utf-8"
    )
    return target


def baseline_source(module: str) -> str:
    return (H.MODULES_DIR / module / "index.ts").read_text(encoding="utf-8")


def verify_version(directory: Path, module: str, tag: str | None = None) -> None:
    """Refuse a retained version whose recorded source or identity changed."""
    artifact = json.loads((directory / "artifact.json").read_text(encoding="utf-8"))
    if artifact.get("module") != module or (tag is not None and artifact.get("tag") != tag):
        raise ValueError(f"wrong version identity: {directory}")
    if artifact.get("source_sha256") != hashlib.sha256((directory / "index.ts").read_bytes()).hexdigest():
        raise ValueError(f"version source changed: {directory}")
    if artifact.get("spec_sha256") != hashlib.sha256((directory / "SPEC.md").read_bytes()).hexdigest():
        raise ValueError(f"version spec changed: {directory}")


# --------------------------------------------------------------------------
# The loop
# --------------------------------------------------------------------------


@dataclass
class RoundRecord:
    round_index: int
    module: str
    decision: str  # accepted | accepted_efficiency | rejected | invalid | budget_exhausted | error
    reason: str
    rationale: str
    before_score: float
    after_score: float | None
    proposal_tokens: int
    source: str = ""
    source_before: str = ""
    analysis: str = ""
    predicted_fixes: list[str] = field(default_factory=list)
    risk_tasks: list[str] = field(default_factory=list)
    before_cost: float = 0.0
    after_cost: float = 0.0
    before_tokens: int = 0
    after_tokens: int | None = None


@dataclass
class ArmResult:
    module: str
    rounds: list[RoundRecord] = field(default_factory=list)
    final_source: str = ""
    final_dir: Path | None = None
    dev_result: H.RunResult | None = None
    proposal_tokens: int = 0
    evaluation_tokens: int = 0

    def to_json(self) -> dict:
        return {
            "module": self.module,
            "final_dir": str(self.final_dir) if self.final_dir else None,
            "proposal_tokens": self.proposal_tokens,
            "evaluation_tokens": self.evaluation_tokens,
            "total_arm_tokens": self.proposal_tokens + self.evaluation_tokens,
            "dev_score": self.dev_result.mean_reward if self.dev_result else None,
            "dev_solved": self.dev_result.solved_count if self.dev_result else None,
            "rounds": [asdict(r) for r in self.rounds],
        }


def _baseline_signature(run: H.RunResult) -> list[tuple]:
    return [
        [o.task, o.reward, o.total_tokens, o.cost_usd]
        for o in run.outcomes
    ]


def _save_checkpoint(path: Path | None, payload: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def improve_module(
    module: str,
    dev_tasks: list[str],
    *,
    baseline: H.RunResult,
    tag: str = "main",
    rounds: int = H.ROUNDS,
    budget_tokens: int = H.TOTAL_TOKEN_BUDGET,
    checkpoint: Path | None = None,
    progress=print,
) -> ArmResult:
    """Run the full improvement loop for one module and return its history."""
    arm = ArmResult(module=module)
    arm.final_source = baseline_source(module)
    arm.final_dir = stage_version(tag, module, 0, arm.final_source)
    current_run = baseline
    arm.dev_result = current_run
    arm.evaluation_tokens = current_run.total_tokens
    progress(f"[{module}] using shared original dev baseline: {current_run.summary()}")

    expected = {
        "tag": tag,
        "module": module,
        "dev_tasks": dev_tasks,
        "round_limit": rounds,
        "budget_tokens": budget_tokens,
        "baseline": _baseline_signature(baseline),
        "implementation": implementation_identity(),
        "model": H.MODEL,
        "max_tool_calls": H.MAX_TOOL_CALLS,
        "max_tokens": H.MAX_TOKENS,
    }
    current_job = baseline.job_dir.name if baseline.job_dir else None
    current_condition = "original"
    pending: dict | None = None
    accounted_attempts: list[str] = []
    resumed = checkpoint is not None and checkpoint.exists()
    if resumed:
        saved = json.loads(checkpoint.read_text(encoding="utf-8"))
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError(f"checkpoint parameters changed: {checkpoint}")
        arm.final_dir = Path(saved["final_dir"])
        verify_version(arm.final_dir, module, tag)
        arm.final_source = (arm.final_dir / "index.ts").read_text(encoding="utf-8")
        if hashlib.sha256(arm.final_source.encode()).hexdigest() != saved["final_sha256"]:
            raise ValueError(f"retained version changed: {arm.final_dir}")
        arm.rounds = [RoundRecord(**entry) for entry in saved["rounds"]]
        arm.proposal_tokens = saved["proposal_tokens"]
        arm.evaluation_tokens = saved["evaluation_tokens"]
        current_job = saved["current_job"]
        current_condition = saved["current_condition"]
        pending = saved.get("pending")
        accounted_attempts = saved.get("accounted_attempts", [])
        if arm.rounds and arm.rounds[-1].decision in ("no_op", "invalid", "budget_exhausted"):
            raise RuntimeError(f"[{module}] round {arm.rounds[-1].round_index} needs diagnosis: {arm.rounds[-1].reason}")
        if current_condition != "original":
            current_run = H.run_condition(
                condition=current_condition, module_dir=arm.final_dir,
                tasks=dev_tasks, task_set="dev", job_name=current_job,
                reuse_clean=True,
            )
        arm.dev_result = current_run
        progress(f"[{module}] resumed after round {len(arm.rounds)}")

    def checkpoint_now() -> None:
        _save_checkpoint(checkpoint, {
            **expected,
            "final_dir": str(arm.final_dir),
            "final_sha256": hashlib.sha256(arm.final_source.encode()).hexdigest(),
            "rounds": [asdict(record) for record in arm.rounds],
            "proposal_tokens": arm.proposal_tokens,
            "evaluation_tokens": arm.evaluation_tokens,
            "current_job": current_job,
            "current_condition": current_condition,
            "pending": pending,
            "accounted_attempts": accounted_attempts,
        })

    checkpoint_now()

    first_round = pending["round"] if pending else len(arm.rounds) + 1
    for round_index in range(first_round, rounds + 1):
        spent = arm.proposal_tokens + arm.evaluation_tokens
        if spent >= budget_tokens:
            arm.rounds.append(
                RoundRecord(
                    round_index=round_index,
                    module=module,
                    decision="budget_exhausted",
                    reason=f"{spent:,} tokens spent of a {budget_tokens:,} cap",
                    rationale="",
                    before_score=current_run.mean_reward,
                    after_score=None,
                    proposal_tokens=0,
                    before_tokens=current_run.total_tokens,
                )
            )
            progress(f"[{module}] round {round_index}: budget exhausted, stopping")
            checkpoint_now()
            raise RuntimeError(f"[{module}] budget exhausted before round {round_index}")

        if pending is None:
            # Stage 1 -- write the evidence to a directory the evolver can search.
            attribution = render_attribution(arm.rounds, current_run)
            round_dir = E.EVIDENCE_DIR / tag / module / f"round{round_index}"
            if resumed and round_index == first_round and round_dir.is_dir():
                ready = json.loads((round_dir / "ready.json").read_text(encoding="utf-8"))
                if ready != {"tag": tag, "module": module, "tasks": [o.task for o in current_run.outcomes]}:
                    raise ValueError(f"round evidence does not match the current run: {round_dir}")
            else:
                E.materialize(
                    round_dir=round_dir,
                    outcomes=current_run.outcomes,
                    module=module,
                    tag=tag,
                    arm=current_run,
                    report=S.render_report([S.collect(o) for o in current_run.outcomes]),
                    history=arm.rounds,
                    attribution=attribution,
                )

        # Stage 2 -- the current agent inspects that evidence with its own tools
        # and rewrites the module, so it can decide what to inspect before
        # proposing a candidate.
        if pending is None:
            for attempt in (1, 2):
                suffix = "workspace" if attempt == 1 else "retry-workspace"
                base = E.EVIDENCE_DIR / tag / module / f"round{round_index}-{suffix}"
                for recovery in (0, 1):
                    workspace = base if recovery == 0 else base.with_name(base.name + "-recovery1")
                    attempt_key = f"{round_index}:{attempt}:{recovery}"
                    saved_result = workspace / "evolver-result.json"
                    loaded = saved_result.is_file()
                    if loaded:
                        result = EV.EvolverResult(**json.loads(saved_result.read_text(encoding="utf-8")))
                        if (workspace / "module" / "index.ts").read_text(encoding="utf-8") != result.source:
                            raise ValueError(f"saved evolver result and candidate disagree: {workspace}")
                    elif workspace.exists():
                        # The process stopped before it produced a valid result.
                        # Preserve its trace and charge any settled model calls.
                        if attempt_key not in accounted_attempts:
                            arm.proposal_tokens += int(EV.summarize_evolver_log(workspace / "evolver.jsonl")["total_tokens"])
                            accounted_attempts.append(attempt_key)
                            checkpoint_now()
                        if recovery == 0:
                            progress(f"[{module}] round {round_index}: retrying interrupted evolver pass")
                            continue
                        raise RuntimeError(f"evolver stopped twice without a completed result: {workspace}")
                    else:
                        EV.build_workspace(
                            workspace=workspace,
                            evidence_dir=round_dir,
                            module=module,
                            current_source=arm.final_source,
                        )
                        result = EV.run_evolver(
                            workspace=workspace,
                            evidence_dir=round_dir,
                            current_dir=arm.final_dir,
                            module=module,
                            source_before=arm.final_source,
                            api_key=api_key(),
                            progress=progress,
                        )
                    if attempt_key not in accounted_attempts:
                        arm.proposal_tokens += result.total_tokens
                        accounted_attempts.append(attempt_key)
                        checkpoint_now()
                    if result.error:
                        if recovery == 0 and not loaded and H._is_permanent(result.error):
                            raise RuntimeError(f"[{module}] round {round_index}: {result.error}")
                        if recovery == 0:
                            progress(f"[{module}] round {round_index}: retrying failed evolver pass: {result.error}")
                            continue
                        raise RuntimeError(f"[{module}] round {round_index}: {result.error}")
                    break
                violation = boundary_violation(module, result.source) if result.changed else None
                if result.changed and violation is None:
                    violation = syntax_violation(result.source)
                if result.changed and violation is None:
                    break
                reason = violation or "the evolver left the module unchanged"
                progress(f"[{module}] round {round_index}, attempt {attempt}: {reason}")
                if attempt == 2:
                    arm.rounds.append(RoundRecord(
                        round_index=round_index, module=module,
                        decision="invalid" if violation else "no_op",
                        reason=f"two proposal attempts failed: {reason}",
                        rationale=result.rationale,
                        before_score=current_run.mean_reward, after_score=None,
                        proposal_tokens=result.total_tokens,
                        analysis=attribution,
                        before_cost=current_run.total_cost_usd,
                        before_tokens=current_run.total_tokens,
                    ))
                    checkpoint_now()
                    raise RuntimeError(f"[{module}] round {round_index}: {reason}; stop for diagnosis")
        else:
            candidate_dir = Path(pending["candidate_dir"])
            verify_version(candidate_dir, module, tag)
            source = (candidate_dir / "index.ts").read_text(encoding="utf-8")
            if hashlib.sha256(source.encode()).hexdigest() != pending["candidate_sha256"]:
                raise ValueError(f"pending candidate changed: {candidate_dir}")
            result = EV.EvolverResult(**{**pending["proposal"], "source": source})
            attribution = pending["attribution"]
            progress(f"[{module}] round {round_index}: resuming candidate {candidate_dir.name}")

        proposal = result

        candidate_dir = stage_version(tag, module, round_index, proposal.source)
        if pending is None:
            pending = {
                "round": round_index,
                "candidate_dir": str(candidate_dir),
                "candidate_sha256": hashlib.sha256(proposal.source.encode()).hexdigest(),
                "proposal": {**asdict(proposal), "source": ""},
                "attribution": attribution,
            }
            checkpoint_now()
        progress(f"[{module}] round {round_index}: evaluating candidate on dev")
        before_score = current_run.mean_reward
        before_cost = current_run.total_cost_usd
        before_tokens = current_run.total_tokens
        # Captured before the accept branch reassigns arm.final_source, so a
        # rejected round still records the version it was compared against.
        source_before = arm.final_source
        candidate_run = H.run_condition(
            condition=f"{module}-v{round_index}",
            module_dir=candidate_dir,
            tasks=dev_tasks,
            task_set="dev",
            job_name=f"{tag}-{module}-dev-v{round_index}",
            reuse_clean=True,
        )
        arm.evaluation_tokens += candidate_run.total_tokens
        progress(f"  {candidate_run.summary()}")

        decision, reason = decide(
            before_score=before_score,
            after_score=candidate_run.mean_reward,
            before_cost=before_cost,
            after_cost=candidate_run.total_cost_usd,
        )

        if decision in ("accepted", "accepted_efficiency"):
            arm.final_source = proposal.source
            arm.final_dir = candidate_dir
            arm.dev_result = candidate_run
            current_run = candidate_run
            current_job = f"{tag}-{module}-dev-v{round_index}"
            current_condition = f"{module}-v{round_index}"

        arm.rounds.append(
            RoundRecord(
                round_index=round_index,
                module=module,
                decision=decision,
                reason=reason,
                rationale=proposal.rationale,
                before_score=before_score,
                after_score=candidate_run.mean_reward,
                proposal_tokens=proposal.total_tokens,
                source=proposal.source,
                source_before=source_before,
                analysis=attribution,
                predicted_fixes=proposal.predicted_fixes,
                risk_tasks=proposal.risk_tasks,
                before_cost=before_cost,
                after_cost=candidate_run.total_cost_usd,
                before_tokens=before_tokens,
                after_tokens=candidate_run.total_tokens,
            )
        )
        progress(f"[{module}] round {round_index}: {decision} -- {reason}")
        pending = None
        checkpoint_now()

    return arm


def decide(
    *, before_score: float, after_score: float, before_cost: float, after_cost: float
) -> tuple[str, str]:
    """Lexicographic: success rate first, cost to break a tie.

    Cost, not tokens, is the tie-breaker. In this setup 96% of all tokens are
    cache reads, which bill at one-fiftieth of fresh input and account for only
    17% of the money; output tokens are under 2% of tokens but 65% of the cost.
    A token threshold therefore measures mostly the cheapest resource, and can
    accept a change that does not save anything. API cost is also what the
    experiment uses as the alternative to token accounting, and it is what a
    fixed computational budget actually means.
    """
    if after_score > before_score:
        return "accepted", f"dev mean reward {before_score:.3f} -> {after_score:.3f}"
    if after_score < before_score:
        return "rejected", (
            f"dev mean reward {before_score:.3f} -> {after_score:.3f}, accuracy regressed"
        )
    if before_cost > 0 and after_cost <= before_cost * (1 - EFFICIENCY_GAIN):
        saved = 1 - after_cost / before_cost
        return "accepted_efficiency", (
            f"dev mean reward unchanged at {before_score:.3f}; cost "
            f"${before_cost:.4f} -> ${after_cost:.4f} ({saved:.1%} cheaper)"
        )
    return "rejected", (
        f"dev mean reward unchanged at {before_score:.3f}; cost "
        f"${before_cost:.4f} -> ${after_cost:.4f}, no material saving"
    )
