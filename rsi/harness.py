"""Run one agent condition over a fixed task set and read the results back.

Everything that touches pier lives here. The rest of the RSI loop works with
`RunResult` and never shells out itself.
"""

from __future__ import annotations

import json
import hashlib
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent          # open_research_RSI/
WORKSPACE = ROOT.parent                                 # Mini-Project/
BENCH = ROOT / "bench"
TASKS_DIR = BENCH / "ccbench" / "tasks"
RUNS_DIR = BENCH / "runs"
MANIFEST_PATH = BENCH / "manifest.json"
MODULES_DIR = ROOT / "modules"
ADAPTER_IMPORT = "open_research_RSI.adapter.pi_pier_agent:PiAgent"
VENV_PYTHON = ROOT / ".venv" / "bin" / "python"
VENV_PIER = ROOT / ".venv" / "bin" / "pier"

MODEL = "deepseek/deepseek-flash"

# Ceilings shared by every condition, per task. See modules/budget/index.ts.
#
# The call ceiling is a runaway backstop, not a binding constraint: the baseline
# already uses up to 68 calls on a dev task, so a ceiling anywhere near that
# would fail runs for being slightly verbose rather than for being worse. Real
# budget control comes from the task time limit and from recording actual usage.
MAX_TOOL_CALLS = 150
MAX_TOKENS = 4_000_000

# Equal total token cap for each improvement run, covering modification
# generation as well as candidate evaluation. Whichever arm is being improved
# gets the same number.
#
# Sized so it never binds within the three mandated rounds: a 15-task
# evaluation costs roughly 48M tokens at the measured baseline rate, and each
# round adds an evolver pass, so a full arm runs about 200M. A cap that bit
# early would silently drop the experiment to two rounds.
TOTAL_TOKEN_BUDGET = 300_000_000

ROUNDS = 3
N_CONCURRENT = 4

PI_LOG = "pi.jsonl"


# Transient trial failures are retried without rerunning completed valid tasks.
# Deterministic pi failures and provider rejections stop for diagnosis.
TRANSIENT_RETRIES = 2


def _is_permanent(message: str) -> bool:
    """True when retrying cannot possibly help.

    A provider rejection or a deterministic pi/module failure should stop
    immediately. Container starts and transient model errors may recover.
    """
    return (
        bool(re.search(r"\b(401|402|403)\b", message))
        or "Insufficient Balance" in message
        or "pi exited with code" in message
        or "extension hash mismatch" in message
    )


class ModelApiError(RuntimeError):
    """A provider-side failure stopped the agent before it could work."""


def preflight_docker() -> str:
    """Check the Dockerfile frontend used by every CCBench task before a run.

    A failed Hub token request once made all 15 trials fail during image build.
    A tiny build catches that failure before Pier starts a whole condition.
    """
    with tempfile.TemporaryDirectory(prefix="rsi-docker-preflight-") as temp:
        dockerfile = Path(temp) / "Dockerfile"
        dockerfile.write_text("# syntax=docker/dockerfile:1.7-labs\nFROM scratch\n")
        result = subprocess.run(
            ["docker", "build", "--progress=plain", "-f", str(dockerfile), temp],
            text=True,
            capture_output=True,
            timeout=90,
        )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "unknown Docker error")[-800:]
        raise RuntimeError(f"Docker frontend preflight failed: {detail}")
    return "Docker frontend preflight ok (docker/dockerfile:1.7-labs)"


def preflight_model() -> str:
    """Confirm the model endpoint answers before spending hours on a run.

    A completed experiment once turned out to be half worthless because the
    provider balance ran out mid-run: pi reports that as an ordinary empty
    assistant message and exits 0, so it read as a plausible score. One
    five-second call up front is much cheaper than discovering it at the end.
    """
    import urllib.error
    import urllib.request

    key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("DEEPSEEK_API")
    if not key:
        raise ModelApiError("Set DEEPSEEK_API_KEY in the environment before running.")

    payload = json.dumps(
        {
            "model": MODEL.split("/", 1)[-1],
            "messages": [{"role": "user", "content": "reply with the single word ok"}],
            "max_tokens": 128,
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://api.deepseek.com/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:300]
        raise ModelApiError(f"model preflight failed with HTTP {exc.code}: {detail}") from exc
    except OSError as exc:
        raise ModelApiError(f"model preflight could not reach the API: {exc}") from exc

    reply = (body.get("choices") or [{}])[0].get("message", {}).get("content", "")
    if not isinstance(reply, str) or not reply.strip():
        raise ModelApiError("model preflight returned no visible answer")
    usage = body.get("usage") or {}
    return f"model preflight ok ({MODEL} replied {reply.strip()[:20]!r}, {usage.get('total_tokens', '?')} tokens)"


def load_manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def task_path(name: str) -> Path:
    return TASKS_DIR / name


@dataclass(frozen=True)
class TaskOutcome:
    """What one task cost and scored under one condition."""

    task: str
    reward: float | None
    total_tokens: int
    tool_calls: int
    cost_usd: float
    exhausted_by: str | None
    exception: str | None
    trial_dir: Path | None = None
    # Non-None when the model call itself failed. Such a trial carries no
    # information about the agent and must never be scored as a wrong answer.
    api_error: str | None = None

    @property
    def solved(self) -> bool:
        return self.reward is not None and self.reward >= 1.0


@dataclass
class RunResult:
    condition: str
    task_set: str
    outcomes: list[TaskOutcome] = field(default_factory=list)
    job_dir: Path | None = None
    tokens_spent_on_proposals: int = 0
    retry_tokens: int = 0
    retry_cost_usd: float = 0.0

    @property
    def scored(self) -> list[TaskOutcome]:
        return [o for o in self.outcomes if o.reward is not None]

    @property
    def mean_reward(self) -> float:
        """Mean reward over the full task set, only for validated outcomes."""
        if not self.outcomes:
            return 0.0
        if any(o.reward is None or o.api_error for o in self.outcomes):
            raise ValueError("cannot score a condition with invalid trial outcomes")
        return sum(float(o.reward) for o in self.outcomes) / len(self.outcomes)

    @property
    def solved_count(self) -> int:
        return sum(1 for o in self.outcomes if o.solved)

    @property
    def total_tokens(self) -> int:
        return sum(o.total_tokens for o in self.outcomes) + self.tokens_spent_on_proposals + self.retry_tokens

    @property
    def total_cost_usd(self) -> float:
        return sum(o.cost_usd for o in self.outcomes)

    @property
    def actual_cost_usd(self) -> float:
        return self.total_cost_usd + self.retry_cost_usd

    def summary(self) -> str:
        summary = (
            f"{self.condition} on {self.task_set}: "
            f"{self.solved_count}/{len(self.outcomes)} solved, "
            f"mean reward {self.mean_reward:.3f}, "
            f"{self.total_tokens:,} tokens, ${self.total_cost_usd:.4f}"
        )
        if self.retry_tokens or self.retry_cost_usd:
            summary += f" (+{self.retry_tokens:,} retry tokens, +${self.retry_cost_usd:.4f})"
        return summary


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _outcome_from_trial(task_name: str, trial_dir: Path) -> TaskOutcome:
    result = _read_json(trial_dir / "result.json")

    reward: float | None = None
    verifier_result = result.get("verifier_result") or {}
    rewards = verifier_result.get("rewards") or {}
    if "reward" in rewards:
        reward = float(rewards["reward"])
    else:
        reward_file = trial_dir / "verifier" / "reward.txt"
        if reward_file.is_file():
            try:
                reward = float(reward_file.read_text(encoding="utf-8").strip())
            except ValueError:
                reward = None

    agent_result = result.get("agent_result") or {}
    metadata = agent_result.get("metadata") or {}
    budget = metadata.get("budget") or {}

    input_tokens = agent_result.get("n_input_tokens") or 0
    cache_tokens = agent_result.get("n_cache_tokens") or 0
    output_tokens = agent_result.get("n_output_tokens") or 0

    # `n_input_tokens` ALREADY includes cache reads and writes: the adapter sums
    # uncached + cacheRead + cacheWrite into it and reports cacheRead separately
    # as `n_cache_tokens`. Adding the cache figure again double-counts it, which
    # inflated totals by a measured factor of 1.87-1.97x across a full task set.
    # Correct total is input + output. `cache_tokens` is kept for reporting the
    # cache share, not for the sum.
    total_tokens = int(input_tokens) + int(output_tokens)

    exception_info = result.get("exception_info") or {}
    exception = exception_info.get("exception_type") or exception_info.get("type")

    from . import signals as _signals

    run_failure = _signals.detect_run_failure(trial_dir)
    if run_failure is None and reward is None:
        run_failure = "trial has no verifier reward"
    return TaskOutcome(
        task=task_name,
        reward=reward,
        api_error=run_failure,
        total_tokens=total_tokens,
        tool_calls=int(metadata.get("tool_calls") or budget.get("tool_calls") or 0),
        cost_usd=float(agent_result.get("cost_usd") or 0.0),
        exhausted_by=budget.get("exhausted_by"),
        exception=exception,
        trial_dir=trial_dir,
    )


def _collect_outcomes(job_dir: Path) -> list[TaskOutcome]:
    if not job_dir.is_dir():
        raise RuntimeError(f"Pier produced no job directory: {job_dir}")
    outcomes = []
    for trial_dir in sorted(p for p in job_dir.iterdir() if p.is_dir()):
        if not (trial_dir / "result.json").is_file():
            continue
        trial = _read_json(trial_dir / "result.json")
        outcomes.append(
            _outcome_from_trial(trial.get("task_name") or trial_dir.name, trial_dir)
        )
    return outcomes


def _require_complete_condition(
    job_name: str, tasks: list[str], outcomes: list[TaskOutcome]
) -> None:
    """Never score a condition unless it has exactly one trial per task."""
    expected = Counter(tasks)
    observed = Counter(outcome.task for outcome in outcomes)
    missing = sorted((expected - observed).elements())
    extra = sorted((observed - expected).elements())
    if missing or extra or not tasks:
        raise RuntimeError(
            f"incomplete Pier job {job_name}: expected {len(tasks)} tasks, "
            f"got {len(outcomes)} trials; missing={missing}; unexpected_or_duplicate={extra}"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_identity(
    *, module_dir: Path | None, tasks: list[str], task_set: str, condition: str
) -> dict:
    """Record the task selection, limits and exact module used by this job."""
    budget = MODULES_DIR / "budget" / "index.ts"
    manifest = load_manifest()
    return {
        "condition": condition,
        "task_set": task_set,
        "tasks": tasks,
        "benchmark_commit": manifest["commit"],
        "model": MODEL,
        "max_tool_calls": MAX_TOOL_CALLS,
        "max_tokens": MAX_TOKENS,
        "n_concurrent": N_CONCURRENT,
        "module_dir": str(module_dir) if module_dir else None,
        "module_sha256": _sha256(module_dir / "index.ts") if module_dir else None,
        "budget_sha256": _sha256(budget),
    }


def _archive_failed_job(job_dir: Path) -> None:
    if not job_dir.exists():
        return
    archive = RUNS_DIR / "_failed_attempts" / job_dir.name
    archive.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while (archive / f"attempt{attempt}").exists():
        attempt += 1
    job_dir.rename(archive / f"attempt{attempt}")


def _archive_invalid_trials(job_name: str, job_dir: Path, outcomes: list[TaskOutcome]) -> None:
    """Keep valid trials in Pier's job so its resume runs only missing tasks."""
    invalid = {o.trial_dir for o in outcomes if o.api_error and o.trial_dir is not None}
    for trial_dir in job_dir.iterdir():
        if (trial_dir.is_dir() and not (trial_dir / "result.json").is_file()
                and ((trial_dir / "config.json").exists()
                     or (trial_dir / "trial.log").exists()
                     or (trial_dir / "agent").exists())):
            invalid.add(trial_dir)
    if not invalid:
        return
    archive = RUNS_DIR / "_failed_attempts" / job_name
    attempt = 1
    while (archive / f"attempt{attempt}").exists():
        attempt += 1
    destination = archive / f"attempt{attempt}"
    destination.mkdir(parents=True)
    for trial_dir in sorted(invalid):
        trial_dir.rename(destination / trial_dir.name)


def _archived_spend(job_name: str) -> tuple[int, float]:
    """Count all archived trials, including ones Pier never summarized."""
    archive = RUNS_DIR / "_failed_attempts" / job_name
    tokens = 0
    cost = 0.0
    if not archive.is_dir():
        return tokens, cost
    for attempt in archive.iterdir():
        if not attempt.is_dir():
            continue
        for trial in attempt.iterdir():
            if not trial.is_dir():
                continue
            agent = (_read_json(trial / "result.json").get("agent_result") or {})
            if agent:
                tokens += int(agent.get("n_input_tokens") or 0) + int(agent.get("n_output_tokens") or 0)
                cost += float(agent.get("cost_usd") or 0.0)
            else:
                log = trial / "agent" / PI_LOG
                if log.is_file():
                    for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if event.get("type") != "message_end":
                            continue
                        message = event.get("message") or {}
                        if message.get("role") != "assistant":
                            continue
                        usage = message.get("usage") or {}
                        tokens += sum(int(usage.get(key) or 0) for key in ("input", "cacheRead", "cacheWrite", "output"))
                        cost += float((usage.get("cost") or {}).get("total") or 0.0)
    return tokens, cost


def run_condition(
    *,
    condition: str,
    module_dir: Path | None,
    tasks: list[str],
    task_set: str,
    job_name: str,
    log: bool = True,
    reuse_clean: bool = False,
) -> RunResult:
    """Run every task in `tasks` under one agent condition, once each."""
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    job_dir = RUNS_DIR / job_name
    config_dir = RUNS_DIR / "_configs"
    config_dir.mkdir(parents=True, exist_ok=True)
    identity_path = config_dir / f"{job_name}.identity.json"
    identity = _run_identity(
        module_dir=module_dir, tasks=tasks, task_set=task_set, condition=condition
    )
    if identity_path.exists():
        if _read_json(identity_path) != identity:
            raise FileExistsError(f"{job_name} has a different recorded run identity")
    elif job_dir.exists():
        raise FileExistsError(
            f"{job_dir} has no versioned identity; refusing to reuse a legacy job"
        )
    else:
        with identity_path.open("x", encoding="utf-8") as record:
            json.dump(identity, record, indent=2)
            record.write("\n")

    if job_dir.exists():
        # A job name is a cache key; reuse only after its full identity matches.
        if not reuse_clean:
            raise FileExistsError(
                f"{job_dir} already exists; pass reuse_clean=True to accept the "
                "cached result, or remove the directory to re-run it"
            )
        outcomes = _collect_outcomes(job_dir)
        try:
            _require_complete_condition(job_name, tasks, outcomes)
            complete = True
        except RuntimeError:
            complete = False
        contaminated = [o for o in outcomes if o.api_error]
        if not contaminated and complete:
            retry_tokens, retry_cost = _archived_spend(job_name)
            print(f"  reusing cached {job_name} ({len(outcomes)} trials)", file=sys.stderr)
            return RunResult(
                condition=condition, task_set=task_set, outcomes=outcomes, job_dir=job_dir,
                retry_tokens=retry_tokens, retry_cost_usd=retry_cost,
            )
        if any(_is_permanent(o.api_error or "") for o in contaminated):
            raise ModelApiError(f"cached job {job_name} has a deterministic pi failure")
        observed = Counter(o.task for o in outcomes)
        expected = Counter(tasks)
        if any(count > expected[name] for name, count in observed.items()):
            raise RuntimeError(f"cached job {job_name} has duplicate or unexpected tasks")
        if (job_dir / "config.json").is_file():
            _archive_invalid_trials(job_name, job_dir, outcomes)
            print(f"  resuming incomplete {job_name}; keeping valid trials", file=sys.stderr)
        else:
            # Pier could have died before writing its own config. No trial from
            # that directory can be safely resumed under a verified config.
            if outcomes:
                raise RuntimeError(f"cached job {job_name} has trials but no Pier config")
            _archive_failed_job(job_dir)

    kwargs: dict[str, object] = {
        "budget_dir": str(MODULES_DIR / "budget"),
        "budget_sha256": identity["budget_sha256"],
        "max_tool_calls": MAX_TOOL_CALLS,
        "max_tokens": MAX_TOKENS,
    }
    if module_dir is not None:
        kwargs["module_dir"] = str(module_dir)
        kwargs["module_sha256"] = identity["module_sha256"]

    # `pier run -p` is a single-value option, not a repeatable one: passing it
    # once per task silently keeps only the last one. The task list has to go
    # through a JobConfig, which is what this writes.
    config = {
        "job_name": job_name,
        "jobs_dir": str(RUNS_DIR),
        "n_attempts": 1,
        "n_concurrent_trials": N_CONCURRENT,
        "quiet": True,
        "retry": {"max_retries": 0, "wait_multiplier": 1.0, "min_wait_sec": 1.0, "max_wait_sec": 60.0},
        # `delete` defaults to True in pier, and the teardown it triggers is
        # `docker compose down --rmi all`, which removes the task image. Every
        # trial then rebuilds from scratch -- and a CCBench Go image build runs
        # `go install std` plus a Node download and a global npm install, so that
        # is minutes per trial, times every trial in the experiment. Turning it
        # off keeps `down` (containers are still cleaned up) while leaving the
        # image in place for the next trial and the next condition.
        "environment": {"type": "docker", "delete": False},
        "agents": [
            {
                "import_path": ADAPTER_IMPORT,
                "model_name": MODEL,
                "kwargs": kwargs,
                # Pier trial workers do not reliably inherit the driver's
                # process environment. Resolve this template at trial setup;
                # the secret itself must not enter the persisted job config.
                "env": {"DEEPSEEK_API_KEY": "${DEEPSEEK_API_KEY}"},
            }
        ],
        "tasks": [{"path": str(task_path(name))} for name in tasks],
    }
    config_path = config_dir / f"{job_name}.json"
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    run_cmd = [str(VENV_PIER), "run", "-c", str(config_path), "-y"]
    resume_cmd = [str(VENV_PIER), "job", "resume", "-p", str(job_dir)]
    cmd = resume_cmd if (job_dir / "config.json").is_file() else run_cmd

    env = dict(os.environ)
    key = env.get("DEEPSEEK_API_KEY") or env.get("DEEPSEEK_API")
    if not key:
        raise ModelApiError("DEEPSEEK_API_KEY is not set for Pier trials")
    env["DEEPSEEK_API_KEY"] = key
    env["PYTHONPATH"] = str(WORKSPACE)
    if log:
        action = "job resume" if cmd == resume_cmd else "run"
        print(f"  $ pier {action} {job_name}  ({len(tasks)} tasks)", file=sys.stderr)

    # Pier resume schedules only trial configs absent from the job directory.
    # Move invalid trial directories aside first; valid trials stay in place.
    attempts = TRANSIENT_RETRIES + 1
    for attempt in range(1, attempts + 1):
        completed = subprocess.run(
            cmd, cwd=str(WORKSPACE), env=env, text=True, capture_output=True
        )
        if completed.returncode != 0:
            # pier exits non-zero when any trial fails, which is normal for a
            # condition that solved nothing. Only surface the tail for diagnosis.
            print(f"  pier exit {completed.returncode}", file=sys.stderr)
            print("  " + (completed.stderr or "")[-1200:], file=sys.stderr)

        try:
            outcomes = _collect_outcomes(job_dir)
            _require_complete_condition(job_name, tasks, outcomes)
            failure = None
        except RuntimeError as exc:
            outcomes = _collect_outcomes(job_dir) if job_dir.is_dir() else []
            failure = str(exc)

        failed_calls = [o for o in outcomes if o.api_error]
        if not failed_calls and failure is None:
            retry_tokens, retry_cost = _archived_spend(job_name)
            return RunResult(
                condition=condition, task_set=task_set, outcomes=outcomes, job_dir=job_dir,
                retry_tokens=retry_tokens, retry_cost_usd=retry_cost,
            )

        permanent = [o for o in failed_calls if _is_permanent(o.api_error or "")]
        detail = failure or "; ".join(f"{o.task}: {o.api_error}" for o in failed_calls[:3])
        if permanent or attempt == attempts:
            raise ModelApiError(
                f"condition {job_name} is invalid ({detail}). "
                + (
                    "This failure needs repair before a new run."
                    if permanent
                    else f"Still failing after {attempts} attempts."
                )
            )

        print(
            f"  invalid condition {job_name}: {detail}; retrying "
            f"(attempt {attempt + 1}/{attempts})",
            file=sys.stderr,
        )
        observed = Counter(o.task for o in outcomes)
        expected = Counter(tasks)
        if any(count > expected[name] for name, count in observed.items()):
            raise RuntimeError(f"job {job_name} has duplicate or unexpected tasks")
        if job_dir.is_dir() and (job_dir / "config.json").is_file():
            _archive_invalid_trials(job_name, job_dir, outcomes)
            cmd = resume_cmd
        else:
            if job_dir.is_dir():
                if outcomes:
                    raise RuntimeError(f"job {job_name} has trials but no Pier config")
                _archive_failed_job(job_dir)
            cmd = run_cmd

    raise AssertionError("unreachable")
