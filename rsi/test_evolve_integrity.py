"""Failures in an evolver pass must not become candidate versions."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from open_research_RSI.rsi import evolve as EVO
from open_research_RSI.rsi import evolver as EV
from open_research_RSI.rsi import harness as H


def run_with_reward(reward: float) -> H.RunResult:
    return H.RunResult(
        condition="tools-v0",
        task_set="dev",
        outcomes=[
            H.TaskOutcome(
                task="task-a",
                reward=reward,
                total_tokens=10,
                tool_calls=1,
                cost_usd=0.01,
                exhausted_by=None,
                exception=None,
            )
        ],
    )


class EvolverIntegrityTest(unittest.TestCase):
    def setUp(self) -> None:
        syntax = patch.object(EVO, "syntax_violation", return_value=None)
        syntax.start()
        self.addCleanup(syntax.stop)
        identity = patch.object(EVO, "implementation_identity", return_value={"test": "same"})
        identity.start()
        self.addCleanup(identity.stop)

    def test_api_cost_can_win_reward_tie(self) -> None:
        self.assertEqual(
            EVO.decide(before_score=0.5, after_score=0.5, before_cost=1.0, after_cost=0.75)[0],
            "accepted_efficiency",
        )
        self.assertEqual(
            EVO.decide(before_score=0.5, after_score=0.5, before_cost=1.0, after_cost=0.81)[0],
            "rejected",
        )
        self.assertEqual(
            EVO.decide(before_score=0.5, after_score=0.0, before_cost=1.0, after_cost=0.1)[0],
            "rejected",
        )

    def test_staged_version_cannot_be_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            modules = root / "modules" / "tools"
            modules.mkdir(parents=True)
            (modules / "SPEC.md").write_text("tools only", encoding="utf-8")
            with patch.object(EVO, "VERSIONS_DIR", root / "versions"), patch.object(EVO.H, "MODULES_DIR", root / "modules"):
                version = EVO.stage_version("tag", "tools", 1, "export default old;")
                self.assertEqual(EVO.stage_version("tag", "tools", 1, "export default old;"), version)
                EVO.verify_version(version, "tools", "tag")
                with self.assertRaises(FileExistsError):
                    EVO.stage_version("tag", "tools", 1, "export default changed;")
                (version / "index.ts").write_text("export default changed;", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "source changed"):
                    EVO.verify_version(version, "tools", "tag")

    def test_interrupted_evaluation_resumes_same_candidate(self) -> None:
        baseline = run_with_reward(0.0)
        proposal = EV.EvolverResult(
            ran=True, source="export default candidate;", changed=True,
            rationale="fix a failure", predicted_fixes=["task-a"], risk_tasks=[],
            tool_calls=1, total_tokens=2, returncode=0,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint.json"

            def stage(_tag, _module, version, source):
                directory = root / f"v{version}"
                directory.mkdir(exist_ok=True)
                (directory / "index.ts").write_text(source, encoding="utf-8")
                return directory

            with (
                patch.object(EVO, "baseline_source", return_value="export default old;"),
                patch.object(EVO, "stage_version", side_effect=stage),
                patch.object(EVO, "verify_version"),
                patch.object(EVO, "api_key", return_value="test-key"),
                patch.object(EVO.E, "materialize"),
                patch.object(EVO.EV, "build_workspace"),
                patch.object(EVO.EV, "run_evolver", return_value=proposal) as evolver,
                patch.object(EVO.S, "collect", return_value=None),
                patch.object(EVO.S, "render_report", return_value=""),
                patch.object(EVO.H, "run_condition", side_effect=[RuntimeError("interrupted"), run_with_reward(1.0)]) as run,
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    EVO.improve_module(
                        "tools", ["task-a"], baseline=baseline, tag="test", rounds=1,
                        checkpoint=checkpoint, progress=lambda _: None,
                    )
                self.assertEqual(json.loads(checkpoint.read_text())["pending"]["candidate_sha256"], EVO.hashlib.sha256(proposal.source.encode()).hexdigest())
                arm = EVO.improve_module(
                    "tools", ["task-a"], baseline=baseline, tag="test", rounds=1,
                    checkpoint=checkpoint, progress=lambda _: None,
                )
                self.assertEqual(evolver.call_count, 1)
                self.assertEqual(run.call_count, 2)
                self.assertEqual(arm.rounds[0].decision, "accepted")
                self.assertIsNone(json.loads(checkpoint.read_text())["pending"])

    def test_rejected_candidate_keeps_retained_source(self) -> None:
        baseline = run_with_reward(1.0)
        proposal = EV.EvolverResult(
            ran=True, source="export default candidate;", changed=True,
            rationale="try something", predicted_fixes=[], risk_tasks=[],
            tool_calls=1, total_tokens=2, returncode=0,
        )
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.object(EVO, "baseline_source", return_value="export default old;"),
                patch.object(EVO, "stage_version", side_effect=lambda _tag, _module, version, _source: Path(temp) / f"v{version}"),
                patch.object(EVO, "api_key", return_value="test-key"),
                patch.object(EVO.E, "materialize"),
                patch.object(EVO.EV, "build_workspace"),
                patch.object(EVO.EV, "run_evolver", return_value=proposal),
                patch.object(EVO.S, "collect", return_value=None),
                patch.object(EVO.S, "render_report", return_value=""),
                patch.object(EVO.H, "run_condition", return_value=run_with_reward(0.0)),
            ):
                arm = EVO.improve_module(
                    "tools", ["task-a"], baseline=baseline, tag="test", rounds=1,
                    progress=lambda _: None,
                )
            self.assertEqual(arm.rounds[0].decision, "rejected")
            self.assertEqual(arm.final_source, "export default old;")
            self.assertEqual(arm.final_dir.name, "v0")

    def test_interrupted_evolver_pass_retries_and_counts_spend(self) -> None:
        baseline = run_with_reward(0.0)
        proposal = EV.EvolverResult(
            ran=True, source="export default candidate;", changed=True,
            rationale="fix", predicted_fixes=[], risk_tasks=[],
            tool_calls=1, total_tokens=2, returncode=0,
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            checkpoint = root / "checkpoint.json"

            def stage(_tag, _module, version, source):
                directory = root / "versions" / f"v{version}"
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "index.ts").write_text(source, encoding="utf-8")
                return directory

            def build(workspace, **_kwargs):
                (workspace / "module").mkdir(parents=True)
                (workspace / "module" / "index.ts").write_text("export default old;", encoding="utf-8")

            calls = 0

            def evolve(workspace, **_kwargs):
                nonlocal calls
                calls += 1
                if calls == 1:
                    (workspace / "evolver.jsonl").write_text(json.dumps({
                        "type": "message_end",
                        "message": {"role": "assistant", "usage": {"input": 10, "output": 2}},
                    }) + "\n", encoding="utf-8")
                    raise RuntimeError("interrupted")
                (workspace / "module" / "index.ts").write_text(proposal.source, encoding="utf-8")
                return proposal

            with (
                patch.object(EVO.E, "EVIDENCE_DIR", root / "evidence"),
                patch.object(EVO, "baseline_source", return_value="export default old;"),
                patch.object(EVO, "stage_version", side_effect=stage),
                patch.object(EVO, "verify_version"),
                patch.object(EVO, "api_key", return_value="test-key"),
                patch.object(EVO.E, "materialize"),
                patch.object(EVO.EV, "build_workspace", side_effect=build),
                patch.object(EVO.EV, "run_evolver", side_effect=evolve),
                patch.object(EVO.S, "collect", return_value=None),
                patch.object(EVO.S, "render_report", return_value=""),
                patch.object(EVO.H, "run_condition", return_value=run_with_reward(1.0)),
            ):
                with self.assertRaisesRegex(RuntimeError, "interrupted"):
                    EVO.improve_module(
                        "tools", ["task-a"], baseline=baseline, tag="recovery", rounds=1,
                        checkpoint=checkpoint, progress=lambda _: None,
                    )
                arm = EVO.improve_module(
                    "tools", ["task-a"], baseline=baseline, tag="recovery", rounds=1,
                    checkpoint=checkpoint, progress=lambda _: None,
                )
            self.assertEqual(calls, 2)
            self.assertEqual(arm.proposal_tokens, 14)
            self.assertEqual(arm.rounds[0].decision, "accepted")

    def test_nonzero_exit_after_tool_use_is_an_error(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            workspace = Path(temp)
            module = workspace / "module"
            module.mkdir()
            (module / "index.ts").write_text("export default old;")

            def failed_container(_command, stdout, stderr, timeout, env):
                stdout.write(json.dumps({"type": "tool_execution_start"}).encode() + b"\n")
                stderr.write(b"pi exited unexpectedly")
                (module / "index.ts").write_text("export default partial;")
                return SimpleNamespace(returncode=1)

            with patch.object(EV.subprocess, "run", side_effect=failed_container):
                result = EV.run_evolver(
                    workspace=workspace,
                    evidence_dir=workspace,
                    current_dir=workspace,
                    module="tools",
                    source_before="export default old;",
                    api_key="test-key",
                    progress=lambda _message: None,
                )

        self.assertTrue(result.changed)
        self.assertIsNotNone(result.error)

    def test_partial_edit_with_error_stops_before_candidate_evaluation(self) -> None:
        baseline = run_with_reward(0.0)
        partial = EV.EvolverResult(
            ran=True,
            source="export default partial;",
            changed=True,
            rationale="",
            predicted_fixes=[],
            risk_tasks=[],
            tool_calls=1,
            total_tokens=10,
            returncode=1,
            error="evolver exited 1",
        )
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.object(EVO, "baseline_source", return_value="export default old;"),
                patch.object(EVO, "stage_version", return_value=Path(temp)),
                patch.object(EVO, "api_key", return_value="test-key"),
                patch.object(EVO.H, "run_condition", side_effect=AssertionError("candidate ran")) as run,
                patch.object(EVO.E, "materialize"),
                patch.object(EVO.EV, "build_workspace"),
                patch.object(EVO.EV, "run_evolver", return_value=partial),
                patch.object(EVO.S, "collect", return_value=None),
                patch.object(EVO.S, "render_report", return_value=""),
            ):
                with self.assertRaisesRegex(RuntimeError, "evolver exited 1"):
                    EVO.improve_module("tools", ["task-a"], baseline=baseline, tag="test", rounds=1, progress=lambda _: None)
                self.assertEqual(run.call_count, 0)

    def test_exec_candidate_cannot_register_a_tool(self) -> None:
        proposal = EV.EvolverResult(
            ran=True,
            source="export default function extension(pi) { pi.registerTool({}); }",
            changed=True,
            rationale="add a tool",
            predicted_fixes=[],
            risk_tasks=[],
            tool_calls=1,
            total_tokens=1,
            returncode=0,
        )
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.object(EVO, "baseline_source", return_value="export default old;"),
                patch.object(EVO, "stage_version", return_value=Path(temp)),
                patch.object(EVO, "api_key", return_value="test-key"),
                patch.object(EVO.H, "run_condition", side_effect=AssertionError("candidate ran")) as run,
                patch.object(EVO.E, "materialize"),
                patch.object(EVO.EV, "build_workspace"),
                patch.object(EVO.EV, "run_evolver", return_value=proposal) as evolver,
                patch.object(EVO.S, "collect", return_value=None),
                patch.object(EVO.S, "render_report", return_value=""),
            ):
                with self.assertRaisesRegex(RuntimeError, "stop for diagnosis"):
                    EVO.improve_module("exec", ["task-a"], baseline=run_with_reward(1.0), tag="test", rounds=1, progress=lambda _: None)
                self.assertEqual(run.call_count, 0)
                self.assertEqual(evolver.call_count, 2)

    def test_rejected_candidate_still_counts_against_arm_budget(self) -> None:
        proposal = EV.EvolverResult(
            ran=True,
            source="export default candidate;",
            changed=True,
            rationale="change a policy",
            predicted_fixes=[],
            risk_tasks=[],
            tool_calls=1,
            total_tokens=1,
            returncode=0,
        )
        with tempfile.TemporaryDirectory() as temp:
            with (
                patch.object(EVO, "baseline_source", return_value="export default old;"),
                patch.object(EVO, "stage_version", return_value=Path(temp)),
                patch.object(EVO, "api_key", return_value="test-key"),
                patch.object(EVO.H, "run_condition", return_value=run_with_reward(0.0)) as run,
                patch.object(EVO.E, "materialize"),
                patch.object(EVO.EV, "build_workspace"),
                patch.object(EVO.EV, "run_evolver", side_effect=[proposal, AssertionError("second proposal ran")]) as evolver,
                patch.object(EVO.S, "collect", return_value=None),
                patch.object(EVO.S, "render_report", return_value=""),
            ):
                with self.assertRaisesRegex(RuntimeError, "budget exhausted"):
                    EVO.improve_module(
                        "tools", ["task-a"], baseline=run_with_reward(1.0), tag="test", rounds=2,
                        budget_tokens=15, progress=lambda _: None,
                    )
                self.assertEqual(run.call_count, 1)
                self.assertEqual(evolver.call_count, 1)


if __name__ == "__main__":
    unittest.main()
