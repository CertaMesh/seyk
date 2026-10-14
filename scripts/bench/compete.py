"""Offline competitor adapters. Install/provision explicitly; never download here."""
import json
import math
import os
from pathlib import Path
import platform
import re
import selectors
import shutil
import signal
import subprocess
import sys
import time
from urllib.parse import urlsplit

import blake3
import bench

HERE = Path(__file__).resolve().parent
INSTALL = bench.ROOT / "competitors"
MODES = {"qmd": {"bm25": "lexical", "vector": "semantic", "hybrid": "hybrid", "full": "hybrid"},
         "digup": {"hybrid": "hybrid"}, "leann": {"contriever": "semantic"}}
VERSIONS = {"qmd": "2.8.3", "digup": "f509c329e0032fbee4d945e1edbdbca7d94af50f", "leann": "0.3.8"}


def offline(command):
    bench.require(platform.system() == "Darwin", "offline adapter currently requires macOS sandbox-exec")
    return ["/usr/bin/sandbox-exec", "-f", str(HERE / "offline.sb"), *map(str, command)]


def network_probe():
    # Numeric IP avoids mistaking a DNS failure for enforced network isolation.
    code = """import errno,socket
try:
    socket.create_connection(('1.1.1.1',443),timeout=3)
except OSError as e:
    assert e.errno in (errno.EPERM,errno.EACCES), repr(e)
else:
    raise AssertionError('external network was allowed')
listener=socket.socket()
listener.bind(('127.0.0.1',0))
listener.listen(1)
with socket.create_connection(listener.getsockname(),timeout=3):
    connection,_=listener.accept()
    connection.close()
listener.close()
"""
    subprocess.run(offline([sys.executable, "-c", code]), check=True, timeout=10,
                   stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    return "sandbox-exec: external numeric TCP denied with EPERM/EACCES; loopback IPC succeeds"


def installed_provenance(tool):
    if tool == "qmd":
        package = INSTALL / "qmd-install/node_modules/@tobilu/qmd/package.json"
        actual = json.loads(package.read_text())["version"]
        artifact = INSTALL / "qmd-install/package-lock.json"
    elif tool == "digup":
        actual = subprocess.check_output(["git", "-C", str(INSTALL / "digup"), "rev-parse", "HEAD"], text=True).strip()
        artifact = INSTALL / "digup/.build/release/digup"
    else:
        actual = subprocess.check_output([str(INSTALL / "leann-env/bin/python"), "-c",
            "from importlib.metadata import version; print(version('leann-core'))"], text=True).strip()
        artifact = INSTALL / "leann-requirements.txt"
    bench.require(actual == VERSIONS[tool], "installed competitor version does not match adapter")
    models = [r for r in json.loads((INSTALL / "model-provenance.json").read_text()) if r["tool"] == tool]
    bench.require(len(models) == {"qmd": 3, "digup": 2, "leann": 6}[tool], "incomplete model provisioning")
    for model in models:
        path = bench.confined(INSTALL / (tool + "-models") / model["file"])
        bench.require(path.stat().st_size == model["bytes"] and bench.sha256(path) == model["sha256"], "model changed")
    return {"models": models, "install_sha256": bench.sha256(artifact)}


def map_hits(hits, tool, corpus, docmap):
    bench.require(isinstance(hits, list) and len(hits) <= 200, "invalid hit count")
    ranked, seen = [], set()
    for hit in hits:
        path = hit["path"]
        if tool == "qmd":
            parsed = urlsplit(path)
            bench.require(parsed.scheme == "qmd" and parsed.netloc == "bench"
                          and not parsed.query and not parsed.fragment, "foreign QMD collection")
            name = parsed.path.removeprefix("/")
        else:
            target = Path(path)
            bench.require(target.is_absolute() and target.parent == corpus, "hit outside corpus")
            name = target.name
        bench.require(re.fullmatch(r"[0-9a-f]{16}\.txt", name) is not None, "invalid hashed filename")
        key = name[:-4]
        bench.require(key in docmap, "unmapped competitor hit")
        score = hit.get("score")
        bench.require(score is None or (isinstance(score, (float, int)) and math.isfinite(score)), "invalid score")
        if key in seen:
            continue
        seen.add(key)
        ranked.append({"key": key, "rank": len(ranked) + 1, "score": score,
                       "locator": {"kind": "competitor", "offset": hit.get("offset")}})
    return ranked


def environment(work):
    # Do not inherit API keys, endpoints, proxy settings, or another tool's config.
    env = {key: os.environ[key] for key in ("PATH", "TMPDIR", "LANG", "SYSTEMROOT") if key in os.environ}
    home = work / "home"
    home.mkdir(exist_ok=True)
    env.update(HOME=str(home), XDG_CACHE_HOME=str(work / "cache"),
               XDG_CONFIG_HOME=str(work / "config"), HF_HOME=str(INSTALL / "hf"),
               HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1",
               TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="2", MKL_NUM_THREADS="2",
               LEANN_EMBEDDING_DEVICE="cpu", LEANN_EMBEDDING_SERVER_HOST="127.0.0.1",
               DIGUP_MODELS=str(INSTALL / "digup-models"), QMD_LLAMA_GPU="metal",
               QMD_EMBED_PARALLELISM="2", GGML_METAL_NO_RESIDENCY="1", NO_COLOR="1")
    return env


class Worker:
    def __init__(self, command, work, env, log, ready=False):
        self.process = subprocess.Popen(offline(command), stdin=subprocess.PIPE,
            stdout=subprocess.PIPE, stderr=log, cwd=work, env=env, start_new_session=True)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)
        self.buffer = b""
        self.counter = 0
        self.expect_ready = ready

    def request(self, request, timeout=300):
        if self.expect_ready:
            row = self.read(timeout)
            bench.require(row.get("event") == "ready", "missing DigUp startup event")
            self.expect_ready = False
        self.counter += 1
        request = {**request, "id": self.counter}
        self.process.stdin.write((json.dumps(request) + "\n").encode())
        self.process.stdin.flush()
        row = self.read(timeout)
        bench.require(row.get("id") == self.counter and not row.get("error"), "invalid competitor response; see private log")
        return row

    def read(self, timeout):
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            bench.require(remaining > 0 and self.selector.select(max(0, remaining)), "competitor timed out; see private log")
            data = os.read(self.process.stdout.fileno(), 65536)
            bench.require(data, "competitor exited early; see private log")
            self.buffer += data
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            os.killpg(self.process.pid, signal.SIGTERM)
            self.process.wait(timeout=10)
        finally:
            self.process.stdout.close()
            self.selector.close()


def run_index(command, work, env, log, timeout):
    process = subprocess.Popen(offline(command), cwd=work, env=env, stdout=log,
                               stderr=log, start_new_session=True)
    try:
        code = process.wait(timeout=timeout)
        bench.require(code == 0, "indexer failed; see private log")
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def resume_rows(partial, header, queries):
    if not partial.exists():
        return []
    rows = bench.jsonlines(partial.read_text().splitlines())
    bench.require(rows and rows[0].get("type") == "env", "invalid partial run")
    for key in ("git_sha", "dirty", "profile_id", "manifest_blake3", "run_id", "depth", "models", "install_sha256"):
        bench.require(rows[0][key] == header[key], f"partial run changed {key}")
    previous = rows[1:]
    bench.require(all(r.get("type") == "query" for r in previous), "invalid partial rows")
    bench.require([r["qid"] for r in previous] == [q["qid"] for q in queries[:len(previous)]],
                  "partial run is not an exact query prefix")
    return previous


def commands(tool, work, corpus):
    if tool == "qmd":
        return None, [shutil.which("node"), HERE / "qmd_worker.mjs",
            INSTALL / "qmd-install/node_modules/@tobilu/qmd/dist/index.js",
            work / "index.sqlite", corpus, INSTALL / "qmd-models"]
    if tool == "digup":
        executable = INSTALL / "digup/.build/release/digup"
        return ([executable, "index", corpus, "--include-datasets", "--index", work / "index"],
                [executable, "search-service", corpus, "--index", work / "index"])
    return ([INSTALL / "leann-env/bin/leann", "build", "bench", "--docs", corpus,
             "--embedding-model", INSTALL / "leann-models", "--embedding-mode", "sentence-transformers",
             "--backend-name", "hnsw", "--num-threads", "2"],
            [INSTALL / "leann-env/bin/python", HERE / "leann_worker.py",
             work / ".leann/indexes/bench/documents.leann"])


def run(args):
    tool, ds, mode = args.tool, args.dataset, args.mode
    bench.require(mode in MODES[tool], f"choose mode from {list(MODES[tool])}")
    bench.require(0 < args.timeout <= 3000 and args.max_queries > 0, "phase must be positive and at most 3000 seconds")
    bench.require(re.fullmatch(r"[a-zA-Z0-9_-]+", args.label), "invalid label")
    manifest, docmap, _ = bench.gold(ds)
    corpus = bench.confined(bench.ROOT / "corpus" / ds)
    bench.require(set(p.name for p in corpus.iterdir()) == set(manifest["files"]), "corpus inventory changed")
    for name, digest in manifest["files"].items():
        bench.require(blake3.blake3(bench.confined(corpus / name).read_bytes()).hexdigest() == digest, "corpus changed")
    proof = network_probe()
    provenance = installed_provenance(tool)
    manifest_hash = blake3.blake3((bench.ROOT / "meta" / ds / "manifest.json").read_bytes()).hexdigest()
    work = bench.confined(bench.ROOT / "competitor-runs" / args.label / tool / ds)
    work.mkdir(parents=True, exist_ok=True)
    env = environment(work)
    index_command, query_command = commands(tool, work, corpus)
    marker = work / "indexed.json"
    identity = {"tool": tool, "version": VERSIONS[tool], "manifest_blake3": manifest_hash, **provenance}
    with (work / f"{args.phase}-{mode}.log").open("ab") as log:
        worker = None
        try:
            if args.phase == "index":
                bench.require(not marker.exists(), "index already exists; use a fresh label")
                start, load = time.monotonic(), os.getloadavg()
                if tool == "qmd":
                    worker = Worker(query_command, work, env, log)
                    reply = worker.request({"op": "index", "vectors": mode != "bm25"}, timeout=args.timeout)
                    bench.require(reply["status"]["totalDocuments"] == len(docmap), "incomplete QMD index")
                    bench.require(mode == "bm25" or reply["status"]["needsEmbedding"] == 0, "incomplete QMD embeddings")
                else:
                    run_index(index_command, work, env, log, args.timeout)
                bench.write_json(marker, {**identity, "index_ms": (time.monotonic() - start) * 1000,
                    "vectors": mode != "bm25", "load_before": load, "load_after": os.getloadavg(), "offline": proof})
                print(marker)
                return
            index = json.loads(marker.read_text())
            bench.require(all(index[k] == v for k, v in identity.items()), "index provenance mismatch")
            bench.require(mode == "bm25" or index["vectors"], "index has no vectors")
            run_id = f"{tool}-{mode}-{args.label}"
            out = bench.confined(bench.ROOT / "runs" / ds / f"{run_id}.jsonl")
            partial = out.with_suffix(".jsonl.partial")
            bench.require(not out.exists(), "completed run exists; use a fresh label")
            out.parent.mkdir(parents=True, exist_ok=True)
            git = lambda *a: subprocess.check_output(["git", *a], cwd=HERE, text=True).strip()
            header = {"type": "env", "schema": 1, "dataset": ds, "dataset_manifest": manifest,
                "git_sha": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
                "profile_id": f"{tool}-{VERSIONS[tool]}-{mode}", "depth": 200,
                "lexical_weight": None, "manifest_blake3": manifest_hash, "run_id": run_id,
                "threads": 1 if tool == "leann" else 2, "competitor": tool, "competitor_version": VERSIONS[tool],
                "thread_settings": "LEANN forces OMP/MKL=1; index num_threads=2" if tool == "leann" else "QMD embedding parallelism=2; native runtime defaults",
                "competitor_mode": mode, "hardware_path": "CPU" if tool == "leann" or mode == "bm25" else "Metal",
                "platform": platform.platform(), "cpu": subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"], text=True).strip(),
                **provenance, "index": index,
                "load_before": os.getloadavg(), "offline": proof}
            queries = bench.jsonlines((bench.ROOT / "meta" / ds / "queries.jsonl").read_text().splitlines())
            previous = resume_rows(partial, header, queries)
            for row in previous:
                bench.ranking(row, docmap)
            finished = len(previous)
            query_ms = sum(row["elapsed_ms"] for row in previous)
            start = time.monotonic()
            worker = Worker(query_command, work, env, log, ready=tool == "digup")
            if tool == "digup":
                status = worker.request({"op": "status"})
                bench.require(status["files"] == len(docmap) and status["searchable"] == len(docmap)
                              and status["pending"] == 0 and status["unfinished"] == 0, "incomplete DigUp index")
            with partial.open("a" if partial.exists() else "x") as f:
                if not previous and f.tell() == 0:
                    f.write(json.dumps(header) + "\n")
                for number, query in enumerate(queries[finished:], finished + 1):
                    if number - len(previous) > args.max_queries or time.monotonic() - start >= args.timeout:
                        break
                    before = time.monotonic()
                    reply = worker.request({"op": "search", "q": query["text"], "mode": mode,
                                            "limit": 200, "meaning": True, "prefix_last": False})
                    bench.require(tool != "digup" or reply.get("meaning") is True, "DigUp silently fell back to keywords")
                    ranked = map_hits(reply["hits"], tool, corpus, docmap)
                    elapsed_ms = (time.monotonic() - before) * 1000
                    f.write(json.dumps({"type": "query", "qid": query["qid"], "mode": MODES[tool][mode],
                        "ranked": ranked, "ranked_pages": None, "elapsed_ms": elapsed_ms}) + "\n")
                    f.flush()
                    finished = number
                    query_ms += elapsed_ms
                    if number % 25 == 0: print(f"{tool}/{ds}/{mode}: {number}/{len(queries)}", flush=True)
                if finished == len(queries):
                    f.write(json.dumps({"type": "complete", "queries": len(queries),
                        "query_total_ms": query_ms, "load_after": os.getloadavg()}) + "\n")
            if finished < len(queries):
                print(f"checkpoint {tool}/{ds}/{mode}: {finished}/{len(queries)}; rerun the same query command")
                return
            partial.replace(out)
            print(out)
        finally:
            if worker: worker.close()


def parser(sub):
    sub.add_parser("offline-check", help="verify external network denial and local IPC")
    p = sub.add_parser("compete", help=__doc__)
    p.add_argument("tool", choices=MODES)
    p.add_argument("dataset", choices=("nanoscifact", "scifact", "nfcorpus", "irpapers"))
    p.add_argument("--mode", required=True)
    p.add_argument("--phase", choices=("index", "query"), required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--timeout", type=int, default=3000, help="bounded phase seconds")
    p.add_argument("--max-queries", type=int, default=50, help="queries per bounded phase; same command resumes")
