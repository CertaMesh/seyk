import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("manifest", Path(__file__).with_name("mcp_payload_manifest.py"))
manifest = importlib.util.module_from_spec(spec)
spec.loader.exec_module(manifest)


class ManifestTest(unittest.TestCase):
    def test_claude_pairs_and_deduplicates_completed_results(self):
        use = {"message": {"content": [{"type": "tool_use", "name": "mcp__seyk__search", "id": "public-call"}]}}
        result = {"message": {"content": [{"type": "tool_result", "tool_use_id": "public-call", "content": [{"text": "{}"}]}]}}
        self.assertEqual(list(manifest.claude_calls([use, result, result])), [("public-call", "{}", False)])

    def test_codex_accepts_only_completed_search_not_started_or_other_tool(self):
        item = {"id": "public-call", "type": "mcp_tool_call", "server": "seyk", "tool": "search", "result": {"content": [{"text": "{}"}]}}
        started = {"type": "item.started", "item": item}
        completed = {"type": "item.completed", "item": item}
        self.assertEqual(list(manifest.codex_calls([started, completed, completed])), [("public-call", "{}", False)])
        failed = {"type": "item.completed", "item": dict(item, status="failed", result={"content": [{"text": "public error"}]})}
        self.assertEqual(list(manifest.codex_calls([failed])), [("public-call", "public error", True)])

    def test_scored_records_labels_sources_and_summary_reconcile(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            row = {"agent": "claude", "arm": "code", "cond": "seyk", "id": "public", "seyk_calls": [{}]}
            (root / "results.jsonl").write_text(json.dumps(row) + "\n")
            (root / "labels.json").write_text(json.dumps({"claude-code-seyk-public": "correct"}))
            events = [
                {"message": {"content": [{"type": "tool_use", "name": "mcp__seyk__search", "id": "call"}]}},
                {"message": {"content": [{"type": "tool_result", "tool_use_id": "call", "content": '{"hits":[]}'}]}},
            ]
            (root / "claude-code-seyk-public.jsonl").write_text("\n".join(json.dumps(e) for e in events))
            frozen = manifest.freeze(root)
            self.assertEqual(len(frozen["sources"]), 3)
            self.assertEqual(frozen["tasks"], 1)
            self.assertEqual(frozen["calls"][0]["raw"], '{"hits":[]}')
            self.assertEqual(frozen["reconciliation"]["claude-code-seyk-public"], {"recorded_summary_calls": 1, "raw_calls": 1})
            (root / "labels.json").write_text("{}")
            with self.assertRaises(ValueError):
                manifest.freeze(root)

    def test_public_fixtures_cover_all_locator_kinds_without_private_sources(self):
        fixture = manifest.synthetic()
        self.assertTrue(fixture["synthetic"])
        self.assertEqual(fixture["sources"], [])
        self.assertEqual(len(fixture["calls"]), 13)
        for call in fixture["calls"]:
            self.assertEqual(call["raw_sha256"], manifest.digest(call["raw"].encode()))
            self.assertEqual(len(json.loads(call["raw"])["hits"]), 10)


if __name__ == "__main__":
    unittest.main()
