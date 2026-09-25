"""Guard against treating an incomplete Pier job as an experiment result."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from open_research_RSI.rsi import harness as H


TASKS = ["task-a", "task-b"]


def write_trial(job_dir: Path, name: str) -> None:
    trial_dir = job_dir / f"{name}__run"
    trial_dir.mkdir(parents=True)
    agent_dir = trial_dir / "agent"
    agent_dir.mkdir()
    (agent_dir / "pi.exit_code").write_text("0\n", encoding="utf-8")
    (trial_dir / "result.json").write_text(
        json.dumps(
            {
                "task_name": name,
                "verifier_result": {"rewards": {"reward": 1.0}},
                "agent_result": {"n_input_tokens": 10, "n_output_tokens": 1},
            }
        ),
        encoding="utf-8",
    )


@patch.dict("os.environ", {"DEEPSEEK_API_KEY": "unit-test"})
class RunIntegrityTest(unittest.TestCase):
    def test_pier_job_resume_command_exists(self) -> None:
        if not H.VENV_PIER.is_file():
            self.skipTest("Pier is not installed in this checkout")
        completed = subprocess.run(
            [str(H.VENV_PIER), "job", "resume", "--help"],
            capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--job-path", completed.stdout)

    def test_missing_verifier_reward_is_not_scored_as_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp) / "trial"
            trial.mkdir()
            (trial / "agent").mkdir()
            (trial / "agent" / "pi.exit_code").write_text("0\n", encoding="utf-8")
            (trial / "result.json").write_text(json.dumps({
                "agent_result": {"n_input_tokens": 10, "n_output_tokens": 1},
                "verifier_result": {"rewards": {}},
            }), encoding="utf-8")
            outcome = H._outcome_from_trial("task-a", trial)
            self.assertEqual(outcome.api_error, "trial has no verifier reward")
            with self.assertRaisesRegex(ValueError, "invalid trial"):
                _ = H.RunResult(condition="invalid", task_set="dev", outcomes=[outcome]).mean_reward

    def test_cached_job_missing_a_task_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.object(H, "RUNS_DIR", Path(temp)):
            job_dir = Path(temp) / "cached"
            job_dir.mkdir()
            identity_dir = Path(temp) / "_configs"
            identity_dir.mkdir()
            (identity_dir / "cached.identity.json").write_text(
                json.dumps(H._run_identity(
                    module_dir=None, tasks=TASKS, task_set="dev", condition="cached"
                )), encoding="utf-8",
            )
            (job_dir / "config.json").write_text(
                json.dumps({"agents": [{"kwargs": {"module_dir": None}}]}),
                encoding="utf-8",
            )
            write_trial(job_dir, "task-a")

            def still_incomplete(*_args, **_kwargs):
                fresh = Path(temp) / "cached"
                self.assertTrue((fresh / "task-a__run" / "result.json").is_file())
                return SimpleNamespace(returncode=0, stderr="")

            with patch.object(H.subprocess, "run", side_effect=still_incomplete), patch.object(H, "TRANSIENT_RETRIES", 0):
                with self.assertRaisesRegex(RuntimeError, "missing.*task-b"):
                    H.run_condition(
                        condition="cached", module_dir=None, tasks=TASKS,
                        task_set="dev", job_name="cached", reuse_clean=True,
                    )

    def test_same_job_name_with_different_tasks_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.object(H, "RUNS_DIR", Path(temp)):
            identity_dir = Path(temp) / "_configs"
            identity_dir.mkdir()
            (identity_dir / "cached.identity.json").write_text(
                json.dumps(H._run_identity(
                    module_dir=None, tasks=TASKS, task_set="dev", condition="cached"
                )), encoding="utf-8",
            )
            with self.assertRaisesRegex(FileExistsError, "different recorded run identity"):
                H.run_condition(
                    condition="cached", module_dir=None, tasks=["task-a"],
                    task_set="dev", job_name="cached", reuse_clean=True,
                )

    def test_same_module_path_with_changed_source_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.object(H, "RUNS_DIR", Path(temp)):
            module = Path(temp) / "module"
            module.mkdir()
            (module / "index.ts").write_text("export default old;", encoding="utf-8")
            identity_dir = Path(temp) / "_configs"
            identity_dir.mkdir()
            (identity_dir / "cached.identity.json").write_text(
                json.dumps(H._run_identity(
                    module_dir=module, tasks=TASKS, task_set="dev", condition="cached"
                )), encoding="utf-8",
            )
            (module / "index.ts").write_text("export default changed;", encoding="utf-8")
            with self.assertRaisesRegex(FileExistsError, "different recorded run identity"):
                H.run_condition(
                    condition="cached", module_dir=module, tasks=TASKS,
                    task_set="dev", job_name="cached", reuse_clean=True,
                )

    def test_new_job_missing_a_task_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.object(H, "RUNS_DIR", Path(temp)):
            def incomplete_pier(*_args, **_kwargs):
                job_dir = Path(temp) / "fresh"
                job_dir.mkdir()
                write_trial(job_dir, "task-a")
                return SimpleNamespace(returncode=0, stderr="")

            with patch.object(H.subprocess, "run", side_effect=incomplete_pier), patch.object(H, "TRANSIENT_RETRIES", 0):
                with self.assertRaisesRegex(RuntimeError, "missing.*task-b"):
                    H.run_condition(
                        condition="fresh",
                        module_dir=None,
                        tasks=TASKS,
                        task_set="dev",
                        job_name="fresh",
                    )

    def test_transient_retry_keeps_and_counts_failed_attempt(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.object(H, "RUNS_DIR", Path(temp)):
            attempts = 0

            def pier_attempt(cmd, **_kwargs):
                nonlocal attempts
                attempts += 1
                job = Path(temp) / "retry"
                job.mkdir(exist_ok=True)
                if attempts == 1:
                    (job / "config.json").write_text("{}", encoding="utf-8")
                else:
                    self.assertEqual(cmd[1:3], ["job", "resume"])
                write_trial(job, "task-a")
                if attempts == 1:
                    (job / "task-a__run" / "agent" / "pi.exit_code").unlink()
                return SimpleNamespace(returncode=0, stderr="")

            with patch.object(H.subprocess, "run", side_effect=pier_attempt):
                result = H.run_condition(
                    condition="retry", module_dir=None, tasks=["task-a"],
                    task_set="dev", job_name="retry",
                )
            self.assertEqual(attempts, 2)
            self.assertEqual(result.retry_tokens, 11)
            self.assertEqual(result.total_tokens, 22)
            self.assertTrue((Path(temp) / "_failed_attempts" / "retry" / "attempt1").is_dir())
            config_text = (Path(temp) / "_configs" / "retry.json").read_text(encoding="utf-8")
            self.assertIn("${DEEPSEEK_API_KEY}", config_text)
            self.assertNotIn("unit-test", config_text)

    def test_one_failed_trial_does_not_rerun_valid_trials(self) -> None:
        with tempfile.TemporaryDirectory() as temp, patch.object(H, "RUNS_DIR", Path(temp)):
            calls = []
            job = Path(temp) / "partial"

            def pier_attempt(cmd, **_kwargs):
                calls.append(cmd)
                if len(calls) == 1:
                    job.mkdir()
                    (job / "config.json").write_text("{}", encoding="utf-8")
                    write_trial(job, "task-a")
                    write_trial(job, "task-b")
                    (job / "task-b__run" / "agent" / "pi.exit_code").unlink()
                else:
                    self.assertEqual(cmd[1:3], ["job", "resume"])
                    self.assertTrue((job / "task-a__run" / "result.json").is_file())
                    self.assertFalse((job / "task-b__run").exists())
                    write_trial(job, "task-b")
                return SimpleNamespace(returncode=0, stderr="")

            with patch.object(H.subprocess, "run", side_effect=pier_attempt):
                result = H.run_condition(
                    condition="partial", module_dir=None, tasks=TASKS,
                    task_set="dev", job_name="partial",
                )
            self.assertEqual(len(calls), 2)
            self.assertEqual(result.solved_count, 2)
            self.assertEqual(result.total_tokens, 33)
            archived = Path(temp) / "_failed_attempts" / "partial" / "attempt1"
            self.assertTrue((archived / "task-b__run" / "result.json").is_file())
            self.assertFalse((archived / "task-a__run").exists())


if __name__ == "__main__":
    unittest.main()
