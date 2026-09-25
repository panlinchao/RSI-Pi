import json
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from open_research_RSI.adapter.pi_pier_agent import PiAgent, summarize_pi_events


class SummarizePiEventsTests(unittest.TestCase):
    def test_final_messages_are_counted_once(self):
        events = [
            {"type": "message_update", "usage": {"input": 999}},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "usage": {
                        "input": 10,
                        "cacheRead": 20,
                        "cacheWrite": 3,
                        "output": 7,
                        "cost": {"total": 0.02},
                    },
                },
            },
            {"type": "tool_execution_start", "toolName": "bash"},
            {"type": "tool_execution_end", "toolName": "bash"},
            {
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "usage": {
                        "input": 5,
                        "cacheRead": 5,
                        "cacheWrite": 0,
                        "output": 4,
                        "cost": {"total": 0.01},
                    },
                },
            },
            {"type": "compaction_end"},
        ]
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "pi.jsonl"
            path.write_text("\n".join(json.dumps(event) for event in events))
            totals = summarize_pi_events(path)

        self.assertEqual(totals["input_tokens"], 43)
        self.assertEqual(totals["cache_read_tokens"], 25)
        self.assertEqual(totals["cache_write_tokens"], 3)
        self.assertEqual(totals["output_tokens"], 11)
        self.assertEqual(totals["peak_input_tokens"], 33)
        self.assertEqual(totals["assistant_messages"], 2)
        self.assertEqual(totals["tool_calls"], 1)
        self.assertEqual(totals["compactions"], 1)
        self.assertAlmostEqual(totals["cost_usd"], 0.03)


class ExtensionUploadTests(unittest.IsolatedAsyncioTestCase):
    async def test_provider_key_is_removed_from_pier_exec_environment(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            agent = PiAgent(
                logs_dir=Path(temporary_dir), model_name="deepseek/deepseek-flash",
                extra_env={"DEEPSEEK_API_KEY": "unit-test-secret"},
            )
            self.assertEqual(agent._api_key, "unit-test-secret")
            self.assertNotIn("DEEPSEEK_API_KEY", agent._extra_env)

    async def test_container_hash_mismatch_stops_trial(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            module = Path(temporary_dir)
            source = b"export default () => {};\n"
            (module / "index.ts").write_bytes(source)
            agent = object.__new__(PiAgent)
            environment = SimpleNamespace(upload_dir=AsyncMock())
            commands = AsyncMock(side_effect=[
                SimpleNamespace(stdout=""),
                SimpleNamespace(stdout=""),
                SimpleNamespace(stdout="0" * 64 + "  /tmp/pi-rsi/module/index.ts\n"),
            ])
            with patch.object(PiAgent, "exec_as_agent", commands):
                with self.assertRaisesRegex(RuntimeError, "hash mismatch"):
                    await agent._inject_extension(
                        environment, module, "/tmp/pi-rsi/module",
                        hashlib.sha256(source).hexdigest(),
                    )
            environment.upload_dir.assert_awaited_once()

    async def test_changed_host_source_stops_before_upload(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            module = Path(temporary_dir)
            (module / "index.ts").write_text("export default changed;", encoding="utf-8")
            agent = object.__new__(PiAgent)
            environment = SimpleNamespace(upload_dir=AsyncMock())
            with self.assertRaisesRegex(ValueError, "host extension hash changed"):
                await agent._inject_extension(environment, module, "/tmp/pi-rsi/module", "0" * 64)
            environment.upload_dir.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
