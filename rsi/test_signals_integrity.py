"""A failed pi process cannot masquerade as a cheap unsuccessful task."""

import json
import tempfile
import unittest
from pathlib import Path

from open_research_RSI.rsi import signals as S


class TrialFailureTest(unittest.TestCase):
    def test_nonzero_pi_exit_is_rejected_even_with_tokens_and_reward(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            trial = Path(temp)
            agent = trial / "agent"
            agent.mkdir()
            (agent / "pi.exit_code").write_text("1\n", encoding="utf-8")
            (trial / "result.json").write_text(json.dumps({
                "agent_result": {"n_input_tokens": 100, "n_output_tokens": 10},
                "verifier_result": {"rewards": {"reward": 0}},
            }), encoding="utf-8")
            self.assertEqual(S.detect_run_failure(trial), "pi exited with code 1")

    def test_missing_pi_exit_code_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            self.assertEqual(S.detect_run_failure(Path(temp)), "pi produced no exit code")


if __name__ == "__main__":
    unittest.main()
