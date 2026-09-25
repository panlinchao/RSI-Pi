"""Cost reports must only include the requested experiment tag."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from open_research_RSI.rsi import cost_report as C


class CostReportTagTest(unittest.TestCase):
    def test_tag_filters_runs_and_evolver_transcripts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runs = root / "bench" / "runs"
            runs.mkdir(parents=True)
            (runs / "main5-tools-dev-v0").mkdir()
            (runs / "main50-tools-dev-v0").mkdir()
            (runs / "_failed_attempts" / "main5-tools-dev-v0" / "attempt1").mkdir(parents=True)
            (runs / "_failed_attempts" / "main50-tools-dev-v0" / "attempt1").mkdir(parents=True)
            for tag in ("main5", "main50"):
                evolver = root / "rsi" / "evidence" / tag / "tools" / "round1-workspace"
                evolver.mkdir(parents=True)
                (evolver / "evolver.jsonl").write_text("")
            with patch.object(C.H, "ROOT", root), patch.object(C.H, "RUNS_DIR", runs):
                groups = C.scan_all("main5")
        self.assertEqual(
            set(groups),
            {"main5-tools-dev-v0", "retry:main5-tools-dev-v0/attempt1", "evolver:main5/tools/round1"},
        )


if __name__ == "__main__":
    unittest.main()
