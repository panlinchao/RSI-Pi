"""Exercise evidence written for rounds after the first proposal."""

import json
import tempfile
import unittest
from pathlib import Path

from open_research_RSI.rsi import evidence as E
from open_research_RSI.rsi.evolve import RoundRecord


class EvidenceHistoryTest(unittest.TestCase):
    def test_second_round_can_include_first_round_accounting(self) -> None:
        first_round = RoundRecord(
            round_index=1,
            module="tools",
            decision="accepted",
            reason="one more task solved",
            rationale="test rationale",
            before_score=0.5,
            after_score=1.0,
            proposal_tokens=100,
            before_tokens=1000,
            after_tokens=900,
        )
        with tempfile.TemporaryDirectory() as temp:
            round_dir = Path(temp) / "round2"
            E.materialize(
                round_dir=round_dir,
                outcomes=[],
                module="tools",
                tag="test-evidence",
                arm=None,
                report="test report",
                history=[first_round],
                attribution="test attribution",
            )
            history = json.loads((round_dir / "history.json").read_text())
        self.assertEqual(history[0]["dev_tokens_before"], 1000)
        self.assertEqual(history[0]["dev_tokens_after"], 900)


if __name__ == "__main__":
    unittest.main()
