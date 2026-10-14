"""Boundary tests: a plausible score with wrong document mapping is worse than failure."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import bench
import compete


class Tests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = "a" * 16, "b" * 16
        self.corpus = Path("/corpus")
        self.docmap = {self.a: "doc one", self.b: "doc two"}

    def test_first_passage_wins_without_resorting_scores(self):
        hits = [{"path": f"/corpus/{key}.txt", "score": score}
                for key, score in [(self.b, 1), (self.b, 9), (self.a, 3)]]
        ranked = compete.map_hits(hits, "leann", self.corpus, self.docmap)
        self.assertEqual([(h["key"], h["rank"], h["score"]) for h in ranked],
                         [(self.b, 1, 1), (self.a, 2, 3)])
        self.assertEqual(bench.ranking({"ranked": ranked}, self.docmap), ["doc two", "doc one"])

    def test_qmd_uses_filename_not_content_docid(self):
        ranked = compete.map_hits([{"path": f"qmd://bench/{self.a}.txt", "docid": "badbad"}],
                                  "qmd", self.corpus, self.docmap)
        self.assertEqual(ranked[0]["key"], self.a)

    def test_wrong_paths_fail_instead_of_counting_as_misses(self):
        for tool, path in [("qmd", f"qmd://other/{self.a}.txt"),
                           ("qmd", f"qmd://bench/nested/{self.a}.txt"),
                           ("digup", f"/other/{self.a}.txt"),
                           ("leann", f"/corpus/{'c'*16}.txt"),
                           ("leann", f"{self.a}.txt")]:
            with self.subTest(path=path), self.assertRaises(ValueError):
                compete.map_hits([{"path": path}], tool, self.corpus, self.docmap)

    def test_empty_and_invalid_scores(self):
        self.assertEqual(compete.map_hits([], "qmd", self.corpus, self.docmap), [])
        for value in (float("nan"), float("inf"), "1"):
            with self.assertRaises(ValueError):
                compete.map_hits([{"path": f"/corpus/{self.a}.txt", "score": value}],
                                 "leann", self.corpus, self.docmap)

    def test_resume_requires_same_provenance_and_exact_query_prefix(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "partial.jsonl"
            header = dict.fromkeys(("git_sha", "dirty", "profile_id", "manifest_blake3", "run_id", "depth", "models", "install_sha256"), "same")
            header["type"] = "env"
            queries = [{"qid": "q1"}, {"qid": "q2"}]
            first = {"type": "query", "qid": "q1"}
            path.write_text(json.dumps(header) + "\n" + json.dumps(first) + "\n")
            self.assertEqual(compete.resume_rows(path, header, queries), [first])
            with self.assertRaisesRegex(ValueError, "prefix"):
                compete.resume_rows(path, header, list(reversed(queries)))
            with self.assertRaisesRegex(ValueError, "git_sha"):
                compete.resume_rows(path, {**header, "git_sha": "changed"}, queries)

    def test_competitor_scored_with_same_metrics_without_seyk_bm25_band(self):
        bench.ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=bench.ROOT) as directory:
            run = Path(directory) / "run.jsonl"
            env = {"type": "env", "schema": 1, "dataset": "scifact", "dataset_manifest": {},
                   "git_sha": "test", "profile_id": "qmd", "depth": 200, "lexical_weight": None,
                   "manifest_blake3": "test", "run_id": "test", "threads": 2, "dirty": False,
                   "competitor": "qmd"}
            query = {"type": "query", "qid": "q", "mode": "lexical", "ranked": [], "ranked_pages": None}
            run.write_text("".join(json.dumps(r) + "\n" for r in [env, query, {"type": "complete", "queries": 1}]))
            with mock.patch.object(bench, "gold", return_value=({}, self.docmap, [bench.ir.Qrel("q", "doc one", 1)])), \
                 mock.patch.object(bench, "ROOT", Path(directory)), mock.patch("sys.stdout"):
                bench.score("scifact", [run])
                result = json.loads((Path(directory) / "results/scifact/test.json").read_text())
                self.assertEqual(result["metrics"]["lexical"]["nDCG@10"], 0)
                del env["competitor"]
                run.write_text("".join(json.dumps(r) + "\n" for r in [env, query, {"type": "complete", "queries": 1}]))
                with self.assertRaisesRegex(ValueError, "outside"):
                    bench.score("scifact", [run])
