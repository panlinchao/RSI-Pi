"""Turn a finished trial into structured evidence.

This is the layer that decides what the evolving model gets to see, and it is
the part of the loop most worth getting right. Two sources are parsed:

* **The verifier transcript** (`verifier/test-stdout.txt`). The CodeCrafters
  tester reports per-stage, per-assertion results, so a failure comes with the
  stage it died in, how many stages it cleared first, and the expected-vs-got
  text. That is a root cause, not a symptom.
* **The agent transcript** (`agent/pi.jsonl`). Tool-call histogram, how many
  calls errored and with what, whether the agent ever ran the local tests, and
  how it signed off.

The design follows the analysis pipeline in the vendored
`agentic-harness-engineering` repo: failures are extracted in detail and
clustered by signature, while successful runs collapse to statistics. Dumping
truncated raw traces at the proposer -- the first version of this loop -- throws
away exactly the structured evidence that makes a proposal evidence-driven.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

PASS_MARK = "\u2713"        # ✓
FAIL_MARK = "\U00010102"    # 𐄂
ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STAGE_RE = re.compile(r"tester::(#[A-Z0-9]+)")
LOCAL_TEST_RE = re.compile(r"go test|go build|test_main\.py|your_program\.sh|go vet|compile\.sh")


def strip_ansi(text: str) -> str:
    return ANSI_RE.sub("", text)


# Exception types that mean the trial never gave the agent a fair chance.
# A timeout is deliberately NOT here: an agent that runs out of time is a real
# outcome a module can cause, and it must be scorable.
INFRASTRUCTURE_EXCEPTIONS = (
    "RuntimeError",                 # docker compose build/start failure
    "EnvironmentStartTimeoutError",
    "RewardFileNotFoundError",
)


def detect_run_failure(trial_dir: Path) -> str | None:
    """Classify a trial that carries no information about the agent.

    Three ways a trial can fail to be evidence:

    * the provider never answered (402, 502, ...), which pi reports as an
      ordinary empty assistant message and exit code 0;
    * the container never started, which pier reports as a RuntimeError;
    * the agent produced nothing at all, which shows up as a zero-token trial
      with no exception to explain it.

    The third case is the dangerous one. A trial whose container failed to start
    records zero tokens, and a condition's token total then silently absorbs the
    loss -- which is how a module with no real saving was once credited with a
    32% reduction.
    """
    trial_dir = Path(trial_dir)
    api_error = detect_api_error(trial_dir / "agent" / "pi.jsonl")
    if api_error:
        return api_error

    exit_path = trial_dir / "agent" / "pi.exit_code"
    if not exit_path.is_file():
        return "pi produced no exit code"
    try:
        exit_code = int(exit_path.read_text(encoding="utf-8").strip())
    except ValueError:
        return "pi exit code is unreadable"
    if exit_code != 0:
        stderr_path = trial_dir / "agent" / "pi.stderr"
        stderr = stderr_path.read_text(encoding="utf-8", errors="replace") if stderr_path.is_file() else ""
        if "No API key found" in stderr:
            return "pi exited with code 1: no API key inside the task container"
        if "Failed to load extension" in stderr:
            return "pi exited with code 1: extension load failed"
        return f"pi exited with code {exit_code}"

    result_path = trial_dir / "result.json"
    if not result_path.is_file():
        return "trial produced no result.json"
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return "trial result.json is unreadable"

    exception = (result.get("exception_info") or {}).get("exception_type")
    if exception in INFRASTRUCTURE_EXCEPTIONS:
        message = str((result.get("exception_info") or {}).get("exception_message") or "")
        return f"{exception}: {' '.join(message.split())[:200]}"

    agent = result.get("agent_result") or {}
    tokens = (
        int(agent.get("n_input_tokens") or 0)
        + int(agent.get("n_output_tokens") or 0)
    )
    if tokens == 0:
        return f"trial consumed zero tokens (no agent run){f'; {exception}' if exception else ''}"
    return None


def detect_api_error(log_path: Path) -> str | None:
    """Return the model-side error that ended a run, if one did.

    pi reports a failed provider call as an ordinary assistant message with
    `stopReason: "error"` and exits 0. Without this check the harness records
    such a run as a task the agent simply failed, which is how an exhausted API
    balance turns into a plausible-looking 0% score. Anything that stops the
    agent from ever calling the model is an infrastructure failure and must be
    distinguishable from a wrong answer.
    """
    log_path = Path(log_path)
    if not log_path.is_file():
        return None
    for raw in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if "errorMessage" not in raw:
            continue
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        message = event.get("message") or {}
        if message.get("role") != "assistant":
            continue
        if message.get("stopReason") == "error" or message.get("errorMessage"):
            return str(message.get("errorMessage") or "model call failed")[:300]
    return None


@dataclass
class VerifierReport:
    stages_total: int = 0
    failed_stage: str | None = None
    stages_cleared_before_failure: int = 0
    failure_evidence: list[str] = field(default_factory=list)
    note: str | None = None

    @property
    def is_near_miss(self) -> bool:
        """Failed only in the last stage it reached."""
        return (
            self.failed_stage is not None
            and self.stages_total > 0
            and self.stages_cleared_before_failure >= self.stages_total - 1
        )


@dataclass
class Behavior:
    turns: int = 0
    tool_usage: dict[str, int] = field(default_factory=dict)
    tool_errors: int = 0
    error_samples: list[str] = field(default_factory=list)
    ran_local_tests: bool = False
    final_message: str = ""


@dataclass
class TaskSignals:
    task: str
    reward: float | None
    tool_calls: int
    tokens: int
    cost_usd: float
    exhausted_by: str | None
    exception: str | None
    verifier: VerifierReport
    behavior: Behavior

    @property
    def solved(self) -> bool:
        return self.reward is not None and self.reward >= 1.0


def parse_verifier(trial_dir: Path) -> VerifierReport:
    path = trial_dir / "verifier" / "test-stdout.txt"
    if not path.is_file():
        return VerifierReport(note="no verifier transcript")

    clean = strip_ansi(path.read_text(encoding="utf-8", errors="replace"))

    # test.sh bails out with a bare message and no reward file in these cases,
    # which is worth naming explicitly: it is a harness-visible precondition
    # failure, not an agent failure.
    for marker, note in (
        ("Error: No compile script found", "the task's compile.sh was missing or not executable"),
        ("Error: Compile failed", "the agent's code did not compile"),
        ("Error: No run script found", "the task's run.sh was missing or not executable"),
    ):
        if marker in clean:
            return VerifierReport(note=note)

    lines = clean.splitlines()
    stage_order: list[str] = []
    current_stage: str | None = None
    failed_stage: str | None = None
    stages_before: list[str] = []
    evidence: list[str] = []

    for index, line in enumerate(lines):
        match = STAGE_RE.search(line)
        if match:
            stage = match.group(1)
            if stage not in stage_order:
                stage_order.append(stage)
            current_stage = stage
        if FAIL_MARK in line:
            failed_stage = current_stage
            stages_before = [s for s in stage_order if s != failed_stage]
            evidence = [" ".join(l.split())[:220] for l in lines[index : index + 3] if l.strip()]
            break

    if failed_stage is None and "Tests finished with exit code 0" not in clean:
        evidence = [f"tester exited non-zero: {lines[-1][:200]}"] if lines else []

    return VerifierReport(
        stages_total=len(stage_order),
        failed_stage=failed_stage,
        stages_cleared_before_failure=len(stages_before),
        failure_evidence=evidence,
    )


def parse_behavior(trial_dir: Path, log_name: str = "pi.jsonl") -> Behavior:
    path = trial_dir / "agent" / log_name
    if not path.is_file():
        return Behavior()

    usage: Counter[str] = Counter()
    pending: dict[str, tuple[str, str]] = {}
    errors: list[str] = []
    turns = 0
    ran_tests = False
    final_message = ""

    for raw in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(raw)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "tool_execution_start":
            name = event.get("toolName") or "?"
            args = event.get("args") or {}
            detail = ""
            if isinstance(args, dict):
                detail = str(args.get("command") or args.get("path") or args.get("file_path") or "")
            detail = " ".join(detail.split())
            usage[name] += 1
            pending[event.get("toolCallId") or ""] = (name, detail)
            if name == "bash" and LOCAL_TEST_RE.search(detail):
                ran_tests = True
        elif kind == "tool_execution_end":
            name, detail = pending.pop(event.get("toolCallId") or "", ("?", ""))
            result = event.get("result") or {}
            if result.get("isError"):
                text = " ".join(
                    "".join(
                        block.get("text", "")
                        for block in (result.get("content") or [])
                        if isinstance(block, dict)
                    ).split()
                )
                errors.append(f"{name}: {detail[:140]} -> {text[:220]}")
        elif kind == "turn_end":
            turns += 1
        elif kind == "message_end":
            message = event.get("message") or {}
            if message.get("role") == "assistant":
                for block in message.get("content") or []:
                    if isinstance(block, dict) and block.get("type") == "text" and block.get("text"):
                        final_message = block["text"]

    return Behavior(
        turns=turns,
        tool_usage=dict(usage.most_common()),
        tool_errors=len(errors),
        error_samples=errors[:6],
        ran_local_tests=ran_tests,
        final_message=final_message,
    )


def collect(outcome, pi_log: str = "pi.jsonl") -> TaskSignals:
    """Build TaskSignals from a harness.TaskOutcome."""
    trial_dir = outcome.trial_dir
    if trial_dir is None:
        return TaskSignals(
            task=outcome.task,
            reward=outcome.reward,
            tool_calls=outcome.tool_calls,
            tokens=outcome.total_tokens,
            cost_usd=outcome.cost_usd,
            exhausted_by=outcome.exhausted_by,
            exception=outcome.exception,
            verifier=VerifierReport(note="no trial directory"),
            behavior=Behavior(),
        )
    return TaskSignals(
        task=outcome.task,
        reward=outcome.reward,
        tool_calls=outcome.tool_calls,
        tokens=outcome.total_tokens,
        cost_usd=outcome.cost_usd,
        exhausted_by=outcome.exhausted_by,
        exception=outcome.exception,
        verifier=parse_verifier(trial_dir),
        behavior=parse_behavior(trial_dir, pi_log),
    )


def render(signals: TaskSignals, *, detail: bool) -> str:
    """Render one task's evidence.

    `detail=True` is for tasks that failed or only just scraped through -- the
    ones with something to learn from. `detail=False` is a single line of stats
    for tasks the agent already solves comfortably.
    """
    v = signals.verifier
    b = signals.behavior
    tools = " ".join(f"{name}x{count}" for name, count in list(b.tool_usage.items())[:6])

    if not detail:
        return (
            f"- {signals.task}: SOLVED in {signals.tool_calls} calls / "
            f"{signals.tokens:,} tokens / {b.turns} turns. top tools: {tools}"
        )

    lines = [
        f"### {signals.task}",
        f"outcome: {'SOLVED' if signals.solved else 'FAILED'}  "
        f"reward={signals.reward}  calls={signals.tool_calls}  "
        f"tokens={signals.tokens:,}  cost=${signals.cost_usd:.4f}  turns={b.turns}",
    ]
    if signals.exhausted_by:
        lines.append(f"HARNESS BUDGET STOP: run terminated by the {signals.exhausted_by} ceiling")
    if signals.exception:
        lines.append(f"harness exception: {signals.exception}")

    if v.note:
        lines.append(f"verifier: {v.note}")
    else:
        lines.append(
            f"verifier: reached stage {v.failed_stage or 'end'} after clearing "
            f"{v.stages_cleared_before_failure} of {v.stages_total} stages"
            + ("  <-- failed on the LAST stage it reached" if v.is_near_miss else "")
        )
        for line in v.failure_evidence:
            lines.append(f"    {line}")

    lines.append(f"tool mix: {tools or '(none)'}")
    lines.append(f"tool calls that errored: {b.tool_errors}")
    for sample in b.error_samples:
        lines.append(f"    {sample}")
    lines.append(f"ran the local tests/build at least once: {'yes' if b.ran_local_tests else 'NO'}")
    if b.final_message:
        lines.append(f"sign-off: {' '.join(b.final_message.split())[:400]}")
    return "\n".join(lines)


def render_report(signals_list: list[TaskSignals]) -> str:
    """Failures and near-misses in full, comfortable successes as one line each."""
    failed = [s for s in signals_list if not s.solved]
    # A run that cleared every stage but the last is a near-miss worth the same
    # detail as a failure: that is where a targeted fix pays off.
    marginal = [s for s in signals_list if s.solved and s.verifier.is_near_miss]
    easy = [s for s in signals_list if s.solved and s not in marginal]

    parts = []
    if failed:
        parts.append("## FAILED tasks (full evidence)\n")
        parts.extend(render(s, detail=True) for s in failed)
    if marginal:
        parts.append("\n## SOLVED but only just (full evidence)\n")
        parts.extend(render(s, detail=True) for s in marginal)
    if easy:
        parts.append("\n## SOLVED comfortably (statistics only)\n")
        parts.extend(render(s, detail=False) for s in easy)
    return "\n".join(parts)
