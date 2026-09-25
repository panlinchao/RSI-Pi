"""Both arms must compare against one original development run."""

import unittest
import tempfile
import hashlib
import json
from pathlib import Path
from unittest.mock import patch

from open_research_RSI.rsi import run_experiment as R
from open_research_RSI.rsi import harness as H


class SharedBaselineTest(unittest.TestCase):
    def test_experiment_tag_cannot_change_its_task_set(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            manifest.write_text('{"commit":"fixed"}', encoding="utf-8")
            with (
                patch.object(R.H, "ROOT", root),
                patch.object(R.H, "MANIFEST_PATH", manifest),
                patch.object(R, "implementation_identity", return_value={"git_commit": "same", "evolver_image_id": "same"}),
            ):
                common = dict(tag="fixed", modules=["tools", "exec"],
                              test_tasks=["test-a"], rounds=3, budget_tokens=10)
                path = R.save_experiment_identity(dev_tasks=["dev-a"], **common)
                self.assertEqual(R.save_experiment_identity(dev_tasks=["dev-a"], **common), path)
                with self.assertRaisesRegex(ValueError, "use a new tag"):
                    R.save_experiment_identity(dev_tasks=["dev-b"], **common)

    def test_one_original_run_is_passed_to_both_arms(self) -> None:
        baseline = H.RunResult(condition="original", task_set="dev", outcomes=[])

        def improve(module, tasks, **kwargs):
            self.assertIs(kwargs["baseline"], baseline)
            self.assertEqual(tasks, ["dev-a"])
            arm = R.ArmResult(module=module)
            arm.dev_result = baseline
            return arm

        with (
            patch.object(R.H, "preflight_docker", return_value="docker ok"),
            patch.object(R.H, "preflight_model", return_value="model ok"),
            patch.object(R.H, "load_manifest", return_value={"dev": ["dev-a"], "test": ["test-a"]}),
            patch.object(R, "save_experiment_identity", return_value="identity.json"),
            patch.object(R.H, "run_condition", return_value=baseline) as run,
            patch.object(R, "improve_module", side_effect=improve) as evolve,
            patch.object(R.evaluate, "write_results"),
        ):
            self.assertEqual(R.main(["--tag", "unit-shared", "--skip-test"]), 0)
            self.assertEqual(run.call_count, 1)
            self.assertEqual(evolve.call_count, 2)
            self.assertEqual(run.call_args.kwargs["module_dir"], None)

    def test_final_only_recovers_the_retained_source(self) -> None:
        baseline = H.RunResult(condition="original", task_set="dev", outcomes=[])
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            options = []
            for module in ("tools", "exec"):
                version = root / f"{module}-v0"
                version.mkdir()
                source = f"export default function {module}() {{}}"
                (version / "index.ts").write_text(source, encoding="utf-8")
                state_dir = root / "rsi" / "checkpoints" / "fixed"
                state_dir.mkdir(parents=True, exist_ok=True)
                (state_dir / f"{module}.json").write_text(json.dumps({
                    "final_dir": str(version),
                    "final_sha256": hashlib.sha256(source.encode()).hexdigest(),
                    "pending": None,
                    "rounds": [{
                        "round_index": 1, "module": module, "decision": "rejected",
                        "reason": "same score", "rationale": "trial", "before_score": 0.0,
                        "after_score": 0.0, "proposal_tokens": 0,
                    }],
                    "current_condition": "original",
                    "current_job": "fixed-dev-original",
                    "proposal_tokens": 0,
                    "evaluation_tokens": 0,
                }), encoding="utf-8")
                options.extend(["--arm-final", f"{module}={version.name}"])
            with (
                patch.object(R.H, "ROOT", root),
                patch.object(R.H, "WORKSPACE", root),
                patch.object(R.H, "preflight_docker", return_value="docker ok"),
                patch.object(R.H, "preflight_model", return_value="model ok"),
                patch.object(R.H, "load_manifest", return_value={"dev": ["dev-a"], "test": ["test-a"]}),
                patch.object(R, "save_experiment_identity", return_value="identity.json"),
                patch.object(R, "verify_version"),
                patch.object(R.H, "run_condition", return_value=baseline),
                patch.object(R.evaluate, "write_results") as write,
            ):
                self.assertEqual(R.main(["--tag", "fixed", "--rounds", "1", "--final-only", "--skip-test", *options]), 0)
                self.assertEqual(set(write.call_args.kwargs["arms"]), {"tools", "exec"})


if __name__ == "__main__":
    unittest.main()
