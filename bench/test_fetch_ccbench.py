"""Fresh-checkout checks for the frozen CCBench materializer."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from open_research_RSI.bench import fetch_ccbench as F


class FetchCCBenchTests(unittest.TestCase):
    def test_fresh_clone_fetches_pinned_revision_when_head_has_moved(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            repo = Path(temp) / "repo"
            commands = []

            def fake_run(cmd, **_kwargs):
                commands.append(cmd)
                if cmd[-2:] == ["rev-parse", "HEAD"]:
                    return SimpleNamespace(stdout="newer-default-branch\n")
                return SimpleNamespace(stdout="")

            with patch.object(F, "run", side_effect=fake_run):
                F.ensure_repo(repo)

            self.assertIn(
                ["git", "-C", str(repo), "fetch", "--depth", "1", "origin", F.CCBENCH_COMMIT],
                commands,
            )
            self.assertIn(
                ["git", "-C", str(repo), "checkout", "--detach", F.CCBENCH_COMMIT],
                commands,
            )

    def test_saved_manifest_fetches_repo_before_missing_tasks(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "seed": 7160, "language": "go", "dev": ["dev-task"],
                "test": ["test-task"], "reserve": [],
            }))
            tasks_dir = root / "tasks"
            events = []

            def materialize(_repo, name, directory):
                events.append(f"task:{name}")
                target = directory / name
                target.mkdir(parents=True)
                (target / "task.toml").write_text("name = 'test'\n")

            argv = [
                "fetch_ccbench.py", "--repo", str(root / "repo"),
                "--tasks-dir", str(tasks_dir), "--manifest", str(manifest),
            ]
            with patch("sys.argv", argv), \
                 patch.object(F, "ensure_repo", side_effect=lambda _repo: events.append("repo")), \
                 patch.object(F, "materialize", side_effect=materialize), \
                 contextlib.redirect_stdout(io.StringIO()):
                F.main()

            self.assertEqual(events, ["repo", "task:dev-task", "task:test-task"])


if __name__ == "__main__":
    unittest.main()
