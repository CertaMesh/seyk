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
ADAPTER = "c" * 40
# Arguments documented in docs/benchmarks.md "Reproduce the chart".
README_CHART = ("a4a0eea63f747b0acf4e3a5fa844f79fc5bef39c", "0.1.0",
                "Apple M5 Max · 18 cores · 64 GiB · macOS 26.5 · 2 inference threads")


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


def competitor(dataset, tool, mode, score, role="headline", empty=0):
    return {"dataset": dataset, "tool": tool, "mode": mode, "ndcg_at_10": score, "comparison_role": role,
            "queries": chart.DATASETS[dataset][1], "queries_without_results": empty,
            "git_sha": ADAPTER, "version": "1.2.3"}


def head_to_head():
    return [competitor("scifact", "digup", "hybrid", 0.7), competitor("nfcorpus", "leann", "contriever", 0.3),
            competitor("scifact", "qmd", "bm25", 0.05, "diagnostic-footnote", 280)]


class ChartTests(unittest.TestCase):
    def test_deterministic_svg_from_fixture(self):
        args = (fixture(), [{"dataset": "scifact", "system": "BM25 & baseline", "ndcg_at_10": 0.665,
                            "source_url": "https://example.org/?a=1&b=2", "source_title": "Source <1>",
                            "protocol": "whole documents"}], SHA, "0.1.0", "Fixture CPU", head_to_head())
        svg = chart.render(*args)
        self.assertEqual(svg, chart.render(*copy.deepcopy(args)))
        root = ET.fromstring(svg)
        self.assertEqual(root.attrib["role"], "img")
        self.assertEqual(root.attrib["width"], "480")
        self.assertIn("BM25 &amp; baseline", svg)
        self.assertIn("Published · different setup", svg)
        self.assertIn("measured on Seyk&#x27;s harness", svg)
        self.assertIn("DigUp hybrid", svg)
        self.assertIn("† QMD BM25 0.050 (diagnostic): 280/300 queries empty", svg)
        self.assertNotIn('width="10.600"', svg)  # diagnostic rows are footnoted, not drawn as bars.
        self.assertIn("prefers-color-scheme:dark", svg)
        self.assertIn('width="159.000"', svg)  # 0.75 on a fixed 212px, 0–1 axis.
        self.assertEqual(hashlib.sha256(svg.encode()).hexdigest(), "36a679c79d3b49f92ad7e2697af70210a1a47cbc37821ee73c041253ea8f6cf2")

    def test_committed_readme_chart_is_reproducible(self):
        docs = chart.REPO / "docs"
        commit, version, machine = README_CHART
        svg = chart.render(chart.latest_results(docs / "benchmarks/results", commit),
                           chart.load_comparisons(docs / "benchmarks/comparisons.toml"), commit, version, machine,
                           chart.load_head_to_head(docs / "benchmarks/head-to-head.toml"))
        self.assertEqual(svg, (docs / "assets/benchmarks.svg").read_text())

    def test_selection_ignores_mtime_and_rejects_ambiguity(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for ds, result in fixture().items():
                (root / ds).mkdir()
                (root / ds / "a.json").write_text(json.dumps(result))
                duplicate = root / ds / "b.json"
                duplicate.write_text(json.dumps(result))
                os.utime(duplicate, (1, 1))
                unrelated = copy.deepcopy(result)
                for env in unrelated["environment"].values():
                    env["git_sha"] = "b" * 40
                (root / ds / "other.json").write_text(json.dumps(unrelated))
                alternate = copy.deepcopy(result)
                for env in alternate["environment"].values():
                    env["dimension"] = 768
                (root / ds / "768.json").write_text(json.dumps(alternate))
            self.assertEqual(chart.latest_results(root, SHA), fixture())  # identical duplicates are harmless.
            changed = fixture()["scifact"]
            changed["metrics"]["hybrid"]["nDCG@10"] = 0.9
            newest = root / "scifact" / "z.json"
            newest.write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "ambiguous scifact aggregates"):
                chart.latest_results(root, SHA)  # newest mtime must not silently win.
            for name in ("a.json", "b.json", "z.json"):
                (root / "scifact" / name).unlink()
            with self.assertRaisesRegex(ValueError, "missing full results"):
                chart.latest_results(root, SHA)

    def test_older_schema_files_are_skipped(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for ds, result in fixture().items():
                (root / ds).mkdir()
                (root / ds / "current.json").write_text(json.dumps(result))
                old = copy.deepcopy(result)
                for env in old["environment"].values():
                    del env["precision"], env["lexical_weight"]
                (root / ds / "old-schema.json").write_text(json.dumps(old))
                (root / ds / "no-env.json").write_text(json.dumps({"dataset": ds, "environment": []}))
                (root / ds / "list.json").write_text("[1, 2]")
                (root / ds / "broken.json").write_text("{")
            self.assertEqual(chart.latest_results(root, SHA), fixture())

    def test_rejects_invalid_measurements(self):
        for group, key, value in [("environment", "dirty", True), ("environment", "dimension", 768),
                                  ("environment", "run_id", "different"), ("completion", "queries", 299),
                                  ("metrics", "nDCG@10", float("nan"))]:
            result = fixture()["scifact"]
            result[group]["semantic"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                chart.validate_result(result, SHA)
        result = fixture()["scifact"]
        del result["completion"]["lexical"]
        with self.assertRaisesRegex(ValueError, "malformed"):
            chart.validate_result(result, SHA)

    def test_rejects_invalid_head_to_head(self):
        mixed = competitor("scifact", "qmd", "vector", 0.7)
        mixed["git_sha"] = "d" * 40
        cases = {"duplicate": head_to_head() + [competitor("scifact", "digup", "hybrid", 0.6)],
                 "unknown tool": [competitor("scifact", "other", "hybrid", 0.6)],
                 "unknown role": [competitor("scifact", "qmd", "vector", 0.6, "featured")],
                 "incomplete": [dict(competitor("scifact", "qmd", "vector", 0.6), queries=299)],
                 "score": [competitor("scifact", "qmd", "vector", 1.5)],
                 "mixed adapter": head_to_head() + [mixed]}
        for name, runs in cases.items():
            with self.subTest(name), self.assertRaises(ValueError):
                chart.validate_head_to_head(runs)

    def test_repository_head_to_head_has_eighteen_runs(self):
        runs = chart.load_head_to_head(chart.REPO / "docs/benchmarks/head-to-head.toml")
        self.assertEqual(len(runs), 18)
        self.assertEqual(sum(r["comparison_role"] == "diagnostic-footnote" for r in runs), 3)


if __name__ == "__main__":
    unittest.main()
