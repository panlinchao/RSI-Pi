"""Run a pi coding-agent configuration inside a Pier task container.

Three things vary per trial, and only three:

* ``module_dir`` -- the host directory of the evolvable pi extension under test
  (module A = tools/skills, module B = execution strategy). It is tarred and
  unpacked into the container, then loaded with ``-e``.
* ``max_tool_calls`` / ``max_tokens`` -- the per-task ceilings, enforced by the
  fixed budget-guard extension that is injected alongside every module. pi has
  no built-in ceiling, and the experiment requires identical limits
  across conditions, so the guard lives here in the harness where no evolving
  module can reach it.

Everything else -- model, decoding settings, tool allowlist, prompt -- is
pinned, so a difference between two runs is attributable to the module.
"""

import json
import hashlib
import os
import shlex
import tempfile
from pathlib import Path

from pier.agents.installed.base import BaseInstalledAgent
from pier.environments.base import BaseEnvironment
from pier.models.agent.context import AgentContext
from pier.models.agent.install import AgentInstallSpec, InstallStep
from pier.models.agent.network import NetworkAllowlist


PI_VERSION = "0.86.1"

# pi needs Node >= 22.19.0. Task images are language toolchains (golang:1.24-bookworm
# for the CCBench Go set) and ship no Node at all, so the agent provisions its own
# runtime rather than assuming one. Pinned for reproducibility.
NODE_VERSION = "22.23.3"

# Model pinned for every trial. pi's catalog names this "DeepSeek V4.1 Flash".
MODEL_ID = "deepseek/deepseek-flash"

# Where the injected extensions land inside the container.
CONTAINER_ROOT = "/tmp/pi-rsi"
BUDGET_EXT_DIR = f"{CONTAINER_ROOT}/budget"
MODULE_EXT_DIR = f"{CONTAINER_ROOT}/module"
BUDGET_REPORT = f"{CONTAINER_ROOT}/budget.json"
API_KEY_FILE = f"{CONTAINER_ROOT}/api-key"

# The built-in tool surface is part of the fixed harness. It is NOT pinned with
# `--tools`, because that flag is an allowlist over built-in, extension AND
# custom tools alike: passing it silently deactivates every tool an evolving
# module registers, which made the tools/skills arm structurally incapable of
# taking effect. pi's own default active set is exactly these four
# (`defaultActiveToolNames` in the SDK), and newly registered extension tools
# are activated on top of it, so leaving the flag off gives the fixed baseline
# and lets the module under test add to it.
EXPECTED_BUILTIN_TOOLS = ("read", "bash", "edit", "write")


def summarize_pi_events(path: Path) -> dict[str, int | float]:
    """Sum final assistant-message usage, ignoring cumulative stream deltas."""
    totals: dict[str, int | float] = {
        "input_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
        "assistant_messages": 0,
        "tool_calls": 0,
        "compactions": 0,
        "peak_input_tokens": 0,
    }
    if not path.exists():
        return totals

    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "tool_execution_start":
            totals["tool_calls"] += 1
        elif kind == "compaction_end":
            totals["compactions"] += 1
        elif kind == "message_end":
            message = event.get("message") or {}
            if message.get("role") != "assistant":
                continue
            totals["assistant_messages"] += 1
            usage = message.get("usage") or {}
            uncached = usage.get("input", 0)
            cache_read = usage.get("cacheRead", 0)
            cache_write = usage.get("cacheWrite", 0)
            input_tokens = uncached + cache_read + cache_write
            totals["input_tokens"] += input_tokens
            totals["cache_read_tokens"] += cache_read
            totals["cache_write_tokens"] += cache_write
            totals["output_tokens"] += usage.get("output", 0)
            totals["cost_usd"] += (usage.get("cost") or {}).get("total", 0.0)
            totals["peak_input_tokens"] = max(totals["peak_input_tokens"], input_tokens)
    return totals


class PiAgent(BaseInstalledAgent):
    """A pinned pi agent with exactly one evolvable module swapped in."""

    _LOG_FILE = "pi.jsonl"

    def __init__(
        self,
        *args,
        smoke_only: bool | str = False,
        module_dir: str | None = None,
        module_sha256: str | None = None,
        budget_dir: str | None = None,
        budget_sha256: str | None = None,
        max_tool_calls: int | str = 0,
        max_tokens: int | str = 0,
        **kwargs,
    ):
        super().__init__(*args, version=PI_VERSION, **kwargs)
        # Pier merges agent.env into every docker compose exec invocation. A
        # provider key there becomes visible in host process arguments. Keep it
        # in memory and upload a short-lived file just for pi's own process.
        self._api_key = self._extra_env.pop("DEEPSEEK_API_KEY", None) or os.environ.get("DEEPSEEK_API_KEY")
        if isinstance(smoke_only, str):
            smoke_only = smoke_only.lower() == "true"
        self.smoke_only = smoke_only
        self.module_dir = Path(module_dir) if module_dir else None
        self.module_sha256 = module_sha256
        self.budget_dir = Path(budget_dir) if budget_dir else None
        self.budget_sha256 = budget_sha256
        self.max_tool_calls = int(max_tool_calls)
        self.max_tokens = int(max_tokens)

    @staticmethod
    def name() -> str:
        return "pi-rsi"

    def get_version_command(self) -> str:
        return "pi --version"

    def install_spec(self) -> AgentInstallSpec:
        return AgentInstallSpec(
            agent_name=self.name(),
            version=PI_VERSION,
            steps=[
                # Node first: the task image has none. `curl` is not guaranteed
                # either, and the tarball is .tar.gz so no xz-utils is needed.
                InstallStep(
                    user="root",
                    run=(
                        "set -euo pipefail; "
                        "apt-get update && "
                        "apt-get install -y --no-install-recommends curl ca-certificates && "
                        "rm -rf /var/lib/apt/lists/*; "
                        f"curl -fsSL https://nodejs.org/dist/v{NODE_VERSION}/"
                        f"node-v{NODE_VERSION}-linux-x64.tar.gz -o /tmp/node.tar.gz; "
                        "tar -xzf /tmp/node.tar.gz -C /usr/local --strip-components=1; "
                        "rm -f /tmp/node.tar.gz; "
                        "node --version; npm --version"
                    ),
                ),
                InstallStep(
                    user="root",
                    run=(
                        "npm install --global --ignore-scripts --no-audit --no-fund "
                        f"@earendil-works/pi-coding-agent@{PI_VERSION}"
                    ),
                ),
            ],
            verification_command=self.get_version_command(),
        )

    def network_allowlist(self) -> NetworkAllowlist:
        return NetworkAllowlist(domains=["api.deepseek.com"])

    async def _inject_extension(
        self, environment: BaseEnvironment, host_dir: Path, target: str,
        expected_sha256: str | None,
    ) -> None:
        """Copy a host extension directory into the container at `target`.

        Host directories are bind-mounted job-wide in pier, and mounts are
        resolved before the agent is constructed, so per-module injection has to
        go through upload_dir rather than a mount.
        """
        host_dir = Path(host_dir)
        if not host_dir.is_dir():
            raise ValueError(f"module directory does not exist: {host_dir}")
        source = host_dir / "index.ts"
        actual_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
        if expected_sha256 and actual_sha256 != expected_sha256:
            raise ValueError(f"host extension hash changed before upload: {host_dir}")
        await self.exec_as_agent(environment, command=f"mkdir -p {shlex.quote(target)}")
        await environment.upload_dir(host_dir, target)
        # upload_dir lands files as root; pi runs as the agent user and has to
        # be able to read the extension for jiti to load it.
        await self.exec_as_agent(
            environment, command=f"chmod -R a+rX {shlex.quote(target)}", timeout_sec=60
        )
        verified = await self.exec_as_agent(
            environment,
            command=f"sha256sum {shlex.quote(target + '/index.ts')}",
            timeout_sec=60,
        )
        uploaded_sha256 = verified.stdout.strip().split()[0] if verified.stdout.strip() else ""
        if uploaded_sha256 != actual_sha256:
            raise RuntimeError(f"container extension hash mismatch: {target}")

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        if self.smoke_only:
            result = await self.exec_as_agent(
                environment,
                command="pi --version && node --version && git rev-parse --is-inside-work-tree",
                cwd="/app",
                timeout_sec=120,
            )
            (self.logs_dir / "pi-smoke.txt").write_text(result.stdout, encoding="utf-8")
            context.metadata = {"smoke_only": True}
            return

        if self.model_name != MODEL_ID:
            raise ValueError(f"This experiment pins the model to {MODEL_ID}")

        # The budget guard is loaded before the module so its handlers are
        # registered first and cannot be shadowed by the module under test.
        extension_flags = ""
        if self.budget_dir is not None:
            await self._inject_extension(environment, self.budget_dir, BUDGET_EXT_DIR, self.budget_sha256)
            extension_flags += f" -e {shlex.quote(BUDGET_EXT_DIR + '/index.ts')}"
        if self.module_dir is not None:
            await self._inject_extension(environment, self.module_dir, MODULE_EXT_DIR, self.module_sha256)
            extension_flags += f" -e {shlex.quote(MODULE_EXT_DIR + '/index.ts')}"

        if not self._api_key:
            raise ValueError("DEEPSEEK_API_KEY was not supplied to the task agent")
        with tempfile.NamedTemporaryFile(mode="w", prefix="pi-rsi-key-", delete=False) as key_file:
            key_file.write(self._api_key)
            key_path = Path(key_file.name)
        try:
            await environment.upload_file(key_path, API_KEY_FILE)
        finally:
            key_path.unlink(missing_ok=True)
        await self.exec_as_agent(
            environment, command=f"chmod a+r {shlex.quote(API_KEY_FILE)}", timeout_sec=60
        )

        container_log = environment.env_paths.agent_dir / self._LOG_FILE
        container_stderr = environment.env_paths.agent_dir / "pi.stderr"
        container_exit = environment.env_paths.agent_dir / "pi.exit_code"
        quoted_instruction = shlex.quote(instruction)
        # Record pi's exit code without letting Pier hide its logs. The harness
        # checks this code before scoring the condition.
        command = (
            f"DEEPSEEK_API_KEY=\"$(cat {shlex.quote(API_KEY_FILE)})\" "
            "pi --offline --mode json --no-session --no-extensions --no-skills "
            "--no-prompt-templates --no-themes --no-context-files --no-approve "
            f"--model {MODEL_ID}"
            f"{extension_flags} "
            f"-- {quoted_instruction} > {shlex.quote(str(container_log))} "
            f"2> {shlex.quote(str(container_stderr))} </dev/null; "
            f"echo $? > {shlex.quote(str(container_exit))}"
        )
        env = self.build_process_env(
            {
                "PI_CODING_AGENT_DIR": "/tmp/pi-rsi-agent",
                "PI_RSI_MAX_TOOL_CALLS": str(self.max_tool_calls),
                "PI_RSI_MAX_TOKENS": str(self.max_tokens),
                "PI_RSI_BUDGET_REPORT": BUDGET_REPORT,
            }
        )
        await self.exec_as_agent(environment, command=command, cwd="/app", env=env)
        await self.exec_as_agent(
            environment, command=f"rm -f {shlex.quote(API_KEY_FILE)}", timeout_sec=60
        )

        # Pull the guard's accounting record out before the container goes away;
        # it is the only place the "did the budget stop this run" verdict lives.
        budget_record = await self.exec_as_agent(
            environment,
            command=f"cat {shlex.quote(BUDGET_REPORT)} 2>/dev/null || echo '{{}}'",
            timeout_sec=60,
        )
        try:
            self._budget_report = json.loads(budget_record.stdout or "{}")
        except json.JSONDecodeError:
            self._budget_report = {}

        # DeepSWE v1.1's verifier extracts the committed patch, not the worktree,
        # so changes are committed when the task image is a git worktree. The
        # CCBench Go images are not, and there the verifier reads the worktree
        # directly -- so skip rather than fail the trial.
        await self.exec_as_agent(
            environment,
            command=(
                "if git rev-parse --is-inside-work-tree >/dev/null 2>&1; then "
                "git add -A && "
                "if ! git diff --cached --quiet; then "
                "git -c user.name='Pi RSI' -c user.email='pi-rsi@example.invalid' "
                "commit -m 'Save pi task changes'; fi; "
                "else echo 'not a git worktree; leaving the worktree in place'; fi"
            ),
            cwd="/app",
        )

    def populate_context_post_run(self, context: AgentContext) -> None:
        if self.smoke_only:
            return
        totals = summarize_pi_events(self.logs_dir / self._LOG_FILE)
        context.n_input_tokens = int(totals["input_tokens"])
        context.n_cache_tokens = int(totals["cache_read_tokens"])
        context.n_output_tokens = int(totals["output_tokens"])
        context.cost_usd = float(totals["cost_usd"])
        context.peak_context_tokens = int(totals["peak_input_tokens"])
        context.summarization_count = int(totals["compactions"])
        context.n_agent_steps = int(totals["assistant_messages"])
        context.metadata = {
            "tool_calls": int(totals["tool_calls"]),
            "cache_write_tokens": int(totals["cache_write_tokens"]),
            "budget": getattr(self, "_budget_report", {}),
            "module_dir": str(self.module_dir) if self.module_dir else None,
            "module_sha256": self.module_sha256,
            "budget_sha256": self.budget_sha256,
            "max_tool_calls": self.max_tool_calls,
            "max_tokens": self.max_tokens,
        }
