"""Boundary tests: a plausible score with wrong document mapping is worse than failure."""
import json
import io
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import bench
import compete


class Tests(unittest.TestCase):
    def setUp(self):
        self.a, self.b = "a" * 16, "b" * 16
        self.corpus = Path("/corpus")
        self.docmap = {self.a: "doc one", self.b: "doc two"}

    def check_process_group_cleanup(self, kind, exited):
        child_code = """import signal,time
signal.signal(signal.SIGTERM, signal.SIG_IGN)
print('ready', flush=True)
time.sleep(30)
"""
        leader_code = """import json,os,subprocess,sys,time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', sys.argv[2]], stdout=subprocess.PIPE)
assert child.stdout.readline() == b'ready\\n'
Path(sys.argv[1]).write_text(json.dumps([os.getpid(), child.pid]))
if sys.argv[3] == 'wait': time.sleep(30)
"""
        with tempfile.TemporaryDirectory() as directory, tempfile.TemporaryFile() as log:
            pidfile = Path(directory) / "pids.json"
            command = [sys.executable, "-c", leader_code, str(pidfile), child_code,
                       "exit" if exited else "wait"]
            worker = None
            stop_group = compete.stop_process_group
            started = time.monotonic()
            try:
                with mock.patch.object(compete, "offline", side_effect=lambda command: command), \
                     mock.patch.object(compete, "stop_process_group",
                                       side_effect=lambda process: stop_group(process, timeout=0.2)):
                    if kind == "worker":
                        worker = compete.Worker(command, directory, os.environ.copy(), log)
                        deadline = time.monotonic() + 5
                        while not pidfile.exists() and time.monotonic() < deadline:
                            time.sleep(0.01)
                        self.assertTrue(pidfile.exists(), "synthetic child failed to start")
                        if exited:
                            worker.process.wait(timeout=5)
                        wait = worker.process.wait
                        with mock.patch.object(worker.process, "wait",
                                               side_effect=lambda timeout: wait(timeout=min(timeout, 0.2))):
                            worker.close()
                        self.assertTrue(worker.process.stdin.closed)
                        self.assertTrue(worker.process.stdout.closed)
                        self.assertIsNone(worker.selector.get_map())
                        if not exited:
                            self.assertEqual(worker.process.returncode, -signal.SIGTERM)
                    elif exited:
                        compete.run_index(command, directory, os.environ.copy(), log, timeout=5)
                    else:
                        with self.assertRaises(subprocess.TimeoutExpired):
                            compete.run_index(command, directory, os.environ.copy(), log, timeout=1)
                leader, child = json.loads(pidfile.read_text())
                for pid in (leader, child):
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
                self.assertLess(time.monotonic() - started, 12)
            finally:
                if pidfile.exists():
                    leader, _ = json.loads(pidfile.read_text())
                    try:
                        os.killpg(leader, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                if worker is not None:
                    try:
                        os.killpg(worker.process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    worker.process.wait(timeout=5)
                    worker.process.stdin.close()
                    worker.process.stdout.close()
                    worker.selector.close()

    def test_cleanup_after_leader_already_exited(self):
        for kind in ("worker", "indexer"):
            with self.subTest(kind=kind):
                self.check_process_group_cleanup(kind, exited=True)

    def test_cleanup_child_surviving_leader_termination(self):
        for kind in ("worker", "indexer"):
            with self.subTest(kind=kind):
                self.check_process_group_cleanup(kind, exited=False)

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

    def test_digup_encoder_events_do_not_replace_or_extend_search_reply(self):
        worker = compete.Worker.__new__(compete.Worker)
        worker.process = SimpleNamespace(stdin=io.BytesIO())
        worker.counter, worker.expect_ready, worker.encoder_events = 0, True, True
        reply = {"id": 1, "hits": [], "meaning": True}
        worker.read = mock.Mock(side_effect=[{"event": "ready"},
            {"event": "encoder", "state": "ready"}, reply])
        self.assertEqual(worker.request({"op": "search"}, timeout=10), reply)
        timeouts = [call.args[0] for call in worker.read.call_args_list]
        self.assertTrue(all(0 < t <= 10 for t in timeouts))
        self.assertEqual(timeouts, sorted(timeouts, reverse=True))
        worker.read = mock.Mock(return_value={"event": "encoder", "state": "unavailable", "error": "fixture"})
        with self.assertRaisesRegex(ValueError, "encoder unavailable"):
            worker.request({"op": "search"})
        worker.read = mock.Mock(return_value={"event": "unexpected"})
        with self.assertRaisesRegex(ValueError, "invalid competitor response"):
            worker.request({"op": "search"})

    def test_checkpoint_then_complete_is_scored_without_losing_empty_queries(self):
        bench.ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=bench.ROOT) as directory:
            root = Path(directory)
            corpus = root / "corpus/scifact"
            corpus.mkdir(parents=True)
            (corpus / (self.a + ".txt")).write_text("Synthetic fixture")
            meta = root / "meta/scifact"
            meta.mkdir(parents=True)
            manifest = {"files": {self.a + ".txt": compete.blake3.blake3(b"Synthetic fixture").hexdigest()}}
            (meta / "manifest.json").write_text(json.dumps(manifest))
            (meta / "queries.jsonl").write_text("".join(json.dumps({"qid": f"q{i}", "text": "fixture"}) + "\n" for i in range(3)))
            model = {"models": [], "install_sha256": "fixture"}
            marker = root / "competitor-runs/fixture/qmd/scifact/indexed.json"
            marker.parent.mkdir(parents=True)
            marker.write_text(json.dumps({"tool": "qmd", "version": compete.VERSIONS["qmd"],
                "manifest_blake3": compete.blake3.blake3((meta / "manifest.json").read_bytes()).hexdigest(),
                "vectors": False, **model}))
            args = SimpleNamespace(tool="qmd", dataset="scifact", mode="bm25", label="fixture",
                                   phase="query", timeout=30, max_queries=2)
            with mock.patch.object(bench, "ROOT", root), \
                 mock.patch.object(bench, "gold", return_value=(manifest, self.docmap, [bench.ir.Qrel(f"q{i}", "doc one", 1) for i in range(3)])), \
                 mock.patch.object(compete, "network_probe", return_value="fixture"), \
                 mock.patch.object(compete, "installed_provenance", return_value=model), \
                 mock.patch.object(compete, "Worker") as worker, mock.patch("sys.stdout"):
                worker.return_value.request.return_value = {"hits": []}
                compete.run(args)
                out = root / "runs/scifact/qmd-bm25-fixture.jsonl"
                partial = out.with_suffix(".jsonl.partial")
                self.assertFalse(out.exists())
                with self.assertRaisesRegex(ValueError, "incomplete"):
                    bench.read_run(partial)
                compete.run(args)
                self.assertFalse(partial.exists())
                _, rows, complete = bench.read_run(out)
                self.assertEqual(set(rows), {"q0", "q1", "q2"})
                self.assertEqual(complete["queries"], 3)
                self.assertEqual(worker.return_value.request.call_count, 3)
                bench.score("scifact", [out])

    def test_leann_success_exit_cannot_publish_an_incomplete_index(self):
        bench.ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=bench.ROOT) as directory:
            root = Path(directory)
            corpus = root / "corpus/scifact"
            corpus.mkdir(parents=True)
            files = {}
            for key in self.docmap:
                name = key + ".txt"
                (corpus / name).write_text("Synthetic fixture")
                files[name] = compete.blake3.blake3(b"Synthetic fixture").hexdigest()
            manifest = {"files": files}
            meta = root / "meta/scifact"
            meta.mkdir(parents=True)
            (meta / "manifest.json").write_text(json.dumps(manifest))
            args = SimpleNamespace(tool="leann", dataset="scifact", mode="contriever",
                                   phase="index", label="fixture", timeout=30, max_queries=2)
            with mock.patch.object(bench, "ROOT", root), \
                 mock.patch.object(bench, "gold", return_value=(manifest, self.docmap, [])), \
                 mock.patch.object(compete, "network_probe", return_value="fixture"), \
                 mock.patch.object(compete, "installed_provenance", return_value={}), \
                 mock.patch.object(compete, "run_index"), \
                 mock.patch.object(compete, "Worker") as worker, mock.patch("sys.stdout"):
                marker = root / "competitor-runs/fixture/leann/scifact/indexed.json"
                worker.return_value.request.return_value = {"paths": [str(corpus / (self.a + ".txt"))]}
                with self.assertRaisesRegex(ValueError, "incomplete LEANN"):
                    compete.run(args)
                self.assertFalse(marker.exists())
                worker.return_value.request.return_value = {"paths": [str(corpus / name) for name in files]}
                compete.run(args)
                self.assertTrue(marker.is_file())

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
