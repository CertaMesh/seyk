"""Small synthetic fixture: no dataset text, queries or rankings."""
import copy
import hashlib
import json
import os
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

import chart

SHA = "a" * 40


def fixture():
    results = {}
    for ds, (_, queries) in chart.DATASETS.items():
        env = {"git_sha": SHA, "dirty": False, "precision": "q8", "dimension": 256,
               "depth": 200, "lexical_weight": None, "run_id": "fixture", "profile_id": "fixture",
               "manifest_blake3": "fixture", "threads": 2}
        results[ds] = {"dataset": ds, "smoke": False,
                       "environment": {m: dict(env) for m in chart.MODES},
                       "completion": {m: {"queries": queries} for m in chart.MODES},
                       "metrics": {m: {"nDCG@10": v} for m, v in zip(chart.MODES, (0.75, 0.8, 0.6))}}
    return results


class ChartTests(unittest.TestCase):
    def test_deterministic_svg_from_fixture(self):
        args = (fixture(), [{"dataset": "scifact", "system": "BM25 & baseline", "ndcg_at_10": 0.665,
                            "source_url": "https://example.org/?a=1&b=2", "source_title": "Source <1>",
                            "protocol": "whole documents"}], SHA, "0.1.0", "Fixture CPU")
        svg = chart.render(*args)
        self.assertEqual(svg, chart.render(*copy.deepcopy(args)))
        root = ET.fromstring(svg)
        self.assertEqual(root.attrib["role"], "img")
        self.assertIn("BM25 &amp; baseline", svg)
        self.assertIn("Published · different setup", svg)
        self.assertIn("prefers-color-scheme:dark", svg)
        self.assertIn('width="420.000"', svg)  # 0.75 on a fixed 560px, 0–1 axis.
        self.assertEqual(hashlib.sha256(svg.encode()).hexdigest(), "2ea4fd60592be152358319e699871bed78a6bc87d507e1af0db4a96841c85241")

    def test_latest_uses_only_requested_commit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for ds, result in fixture().items():
                (root / ds).mkdir()
                p = root / ds / "old.json"
                p.write_text(json.dumps(result))
                os.utime(p, (1, 1))
                newer = copy.deepcopy(result)
                newer["metrics"]["hybrid"]["nDCG@10"] = 0.9
                (root / ds / "new.json").write_text(json.dumps(newer))
                unrelated = copy.deepcopy(result)
                for env in unrelated["environment"].values():
                    env["git_sha"] = "b" * 40
                (root / ds / "other.json").write_text(json.dumps(unrelated))
                alternate = copy.deepcopy(result)
                for env in alternate["environment"].values():
                    env["dimension"] = 768
                (root / ds / "768.json").write_text(json.dumps(alternate))
            actual = chart.latest_results(root, SHA)
            self.assertTrue(all(r["metrics"]["hybrid"]["nDCG@10"] == 0.9 for r in actual.values()))
            (root / "scifact" / "new.json").unlink()
            (root / "scifact" / "old.json").unlink()
            with self.assertRaisesRegex(ValueError, "missing full results"):
                chart.latest_results(root, SHA)

    def test_rejects_invalid_measurements(self):
        for group, key, value in [("environment", "dirty", True), ("environment", "dimension", 768),
                                  ("environment", "run_id", "different"), ("completion", "queries", 299),
                                  ("metrics", "nDCG@10", float("nan"))]:
            result = fixture()["scifact"]
            result[group]["semantic"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                chart.validate_result(result, SHA)


if __name__ == "__main__":
    unittest.main()
