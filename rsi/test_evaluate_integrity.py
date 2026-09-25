"""Final test output is immutable once written for a run tag."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from open_research_RSI.rsi import evaluate as E
from open_research_RSI.rsi import harness as H


class ResultsIntegrityTest(unittest.TestCase):
    def test_partial_save_cannot_erase_final_test_results(self) -> None:
        run = H.RunResult(
            condition="original", task_set="test",
            outcomes=[H.TaskOutcome(
                task="held-out", reward=1.0, total_tokens=10, tool_calls=1,
                cost_usd=0.01, exhausted_by=None, exception=None,
            )],
        )
        with tempfile.TemporaryDirectory() as temp, patch.object(E, "RESULTS_DIR", Path(temp)):
            E.write_results(arms={}, dev_baseline={}, runs=[run], tag="formal")
            with self.assertRaisesRegex(FileExistsError, "cannot be overwritten"):
                E.write_results(arms={}, dev_baseline={}, runs=[], tag="formal")
            self.assertIn("held-out", (Path(temp) / "formal.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
