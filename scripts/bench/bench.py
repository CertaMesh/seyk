# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["blake3==1.0.8", "ir-measures==0.4.3", "pyarrow==19.0.1"]
# ///
"""Public data only. Usage: uv run --locked scripts/bench/bench.py --help.

All downloaded, materialized and per-query artifacts stay in ~/.cache/seyk-bench.
Scoring uses ir_measures/pytrec_eval, with rank-derived scores to retain ties in
Seyk's original order. No metrics are computed from truncated/incomplete queries.
"""
import argparse
import csv
import hashlib
import io
from importlib.metadata import version
import json
import math
import shutil
import sys
import tempfile
import tomllib
import unittest
from unittest import mock
import urllib.request
import zipfile
from pathlib import Path

import blake3
import ir_measures as ir

ROOT = Path.home() / ".cache/seyk-bench"
CONFIG = Path(__file__).with_name("datasets.toml")
MODES = ("hybrid", "semantic", "lexical")
METRICS = [ir.nDCG @ 10, ir.RR @ 10, *(ir.R @ k for k in (1, 5, 10, 20, 100))]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def confined(path):
    path = Path(path).expanduser().absolute()
    require(".." not in path.parts, "parent traversal is not allowed")
    require(path == ROOT or ROOT in path.parents, "path must be under ~/.cache/seyk-bench")
    for parent in (path, *path.parents):
        require(not parent.is_symlink(), f"symlink is not allowed: {parent}")
        if parent == ROOT:
            break
    return path


def config(ds):
    return tomllib.loads(CONFIG.read_text())[ds]


def sha256(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def write_json(path, value):
    path = confined(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = confined(path.with_suffix(path.suffix + ".partial"))
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def fetch(ds):
    for spec in config(ds)["files"]:
        path = confined(ROOT / "raw" / ds / spec["name"])
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            tmp = confined(path.with_suffix(path.suffix + ".partial"))
            print(f"fetch {ds}/{spec['name']}", file=sys.stderr)
            with urllib.request.urlopen(spec["url"], timeout=120) as source, tmp.open("wb") as target:
                shutil.copyfileobj(source, target)
            require(sha256(tmp) == spec["sha256"], "download checksum mismatch")
            tmp.replace(path)
        require(sha256(path) == spec["sha256"], f"checksum mismatch: {path}")


def raw(ds, name):
    return confined(ROOT / "raw" / ds / name)


def jsonlines(lines):
    return [json.loads(line) for line in lines if line.strip()]


def csv_rows(path):
    # Upstream IRPAPERS embeds base64 image columns in the CSV. Stream rows and
    # discard those fields; never decode or materialize the images.
    csv.field_size_limit(128 * 1024 * 1024)
    with path.open(newline="") as f:
        yield from csv.DictReader(f)


def page_id(paper, page):
    require(int(paper) >= 1 and int(page) >= 1, "invalid one-based paper/page")
    return f"{int(paper)}#{int(page)}"


def ir_queries(rows):
    queries, qrels = [], []
    for ordinal, row in enumerate(rows, 1):
        # dataset_id identifies a page, not a unique question (179 IDs/180 rows).
        qid = f"{row['dataset_id']}:{ordinal}"
        queries.append({"qid": qid, "text": row["question"]})
        qrels.append((qid, page_id(row["pdf_id"], row["page_number"]), 1))
    return queries, qrels


def inputs(ds):
    kind = config(ds)["kind"]
    if kind == "beir":
        with zipfile.ZipFile(raw(ds, ds + ".zip")) as z:
            # Read exact members; never extract archive paths to disk.
            docs = jsonlines(z.read(ds + "/corpus.jsonl").splitlines())
            queries = jsonlines(z.read(ds + "/queries.jsonl").splitlines())
            rows = csv.DictReader(io.StringIO(z.read(ds + "/qrels/test.tsv").decode()), delimiter="\t")
            qrels = [(r["query-id"], r["corpus-id"], int(r["score"])) for r in rows]
        query_ids = {q for q, _, _ in qrels}
        return ((str(d["_id"]), (d["title"] + "\n\n" + d["text"]).encode(), "txt") for d in docs), [
            {"qid": str(q["_id"]), "text": q["text"]} for q in queries if str(q["_id"]) in query_ids
        ], qrels
    if kind == "nano":
        import pyarrow.parquet as pq
        docs = pq.read_table(raw(ds, "corpus.parquet")).to_pylist()
        queries = pq.read_table(raw(ds, "queries.parquet")).to_pylist()
        qrels = pq.read_table(raw(ds, "qrels.parquet")).to_pylist()
        return ((d["_id"], d["text"].encode(), "txt") for d in docs), [
            {"qid": q["_id"], "text": q["text"]} for q in queries
        ], [(r["query-id"], r["corpus-id"], 1) for r in qrels]
    queries, qrels = ir_queries(csv_rows(raw(ds, "queries.csv")))
    if kind == "irpapers":
        def pages():
            for row in csv_rows(raw(ds, "docs.csv")):
                require(None not in row and all(value is not None for value in row.values()), "malformed IRPAPERS row")
                # pdf_id_x is the physical PDF ID; pdf_id_y comes from the OCR join.
                paper = row["pdf_id_x"]
                if row["pdf_id_y"]:
                    require(float(row["pdf_id_y"]) == int(paper), "IRPAPERS PDF join mismatch")
                docid = page_id(paper, row["page_number"])
                # Validate the row even when its transcription is empty. None
                # retains its identity for raw-row counts and qrel validation.
                content = (row["pdf_title"] + "\n\n" + row["transcription"]).encode()
                yield docid, content if row["transcription"].strip() else None, "txt"
        return pages(), queries, qrels
    require(kind == "irpapers-pdf", "unknown adapter")
    def pdfs():
        with zipfile.ZipFile(raw(ds, "source.zip")) as z:
            prefix = "IRPAPERS-" + config(ds)["revision"] + "/data/PDFs/"
            for name in sorted(z.namelist()):
                if name.startswith(prefix) and name.endswith(".pdf"):
                    paper = str(int(Path(name).stem))
                    content = z.read(name)
                    require(content.startswith(b"%PDF-"), "not a PDF")
                    yield paper, content, "pdf"
    return pdfs(), queries, qrels


def materialize(ds):
    cfg = config(ds)
    for spec in cfg["files"]:
        require(sha256(raw(ds, spec["name"])) == spec["sha256"], "raw data checksum mismatch; fetch first")
    corpus = confined(ROOT / "corpus" / ds)
    meta = confined(ROOT / "meta" / ds)
    docs, queries, qrels = inputs(ds)
    corpus.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)
    # Remove completion marker first: failed materialization cannot be run.
    confined(meta / "manifest.json").unlink(missing_ok=True)
    mapping, ids, inventory, skipped = {}, set(), {}, set()
    for docid, content, extension in docs:
        require(docid not in ids and not any(c in docid for c in "\t\r\n"), "duplicate or invalid doc ID")
        ids.add(docid)
        key = blake3.blake3(docid.encode()).hexdigest()[:16]
        require(key not in mapping and key not in skipped, "opaque key collision")
        if content is None:
            require(cfg["kind"] == "irpapers", "missing document content")
            skipped.add(key)
            continue
        mapping[key] = docid
        name = key + "." + extension
        path = confined(corpus / name)
        # Preserve mtime and warm-cache reuse on repeat materialization.
        if not path.exists() or path.read_bytes() != content:
            path.write_bytes(content)
        inventory[name] = blake3.blake3(content).hexdigest()
    require(len(ids) == cfg["documents"], "unexpected document count")
    require({p.name for p in corpus.iterdir()} == set(inventory), "stale files in corpus; use a fresh dataset directory")
    qids = {q["qid"] for q in queries}
    require(len(qids) == len(queries) == cfg["queries"], "unexpected/duplicate queries")
    require(qids == {q for q, _, _ in qrels}, "query/qrels mismatch")
    require(len({(q,d) for q,d,_ in qrels}) == len(qrels), "duplicate qrel")
    skipped_qrels = 0
    for q, doc, relevance in qrels:
        target = doc.split("#")[0] if ds == "irpapers-pdf" else doc
        require(target in ids and relevance >= 0, "qrel references missing document or invalid grade")
        skipped_qrels += blake3.blake3(target.encode()).hexdigest()[:16] in skipped
    with confined(meta / "docmap.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["key", "docid"])
        writer.writerows(sorted(mapping.items()))
    with confined(meta / "queries.jsonl").open("w") as f:
        for q in sorted(queries, key=lambda q: q["qid"]):
            f.write(json.dumps(q, sort_keys=True) + "\n")
    with confined(meta / "qrels.tsv").open("w", newline="") as f:
        writer = csv.writer(f, delimiter="\t", lineterminator="\n")
        writer.writerow(["query-id", "corpus-id", "score"])
        writer.writerows(sorted(qrels))
    metadata_hashes = {name: blake3.blake3((meta / name).read_bytes()).hexdigest() for name in ("docmap.tsv", "queries.jsonl", "qrels.tsv")}
    manifest = {"dataset":ds, "source":cfg, "files":inventory, "metadata_blake3":metadata_hashes}
    if cfg["kind"] == "irpapers":
        manifest["empty_transcriptions"] = {"skipped_page_keys":sorted(skipped), "skipped_pages":len(skipped),
                                            "qrels_targeting_skipped_pages":skipped_qrels,
                                            "raw_rows":len(ids), "materialized_pages":len(mapping)}
    write_json(meta / "manifest.json", manifest)
    print(f"materialized {ds}: {len(mapping)} documents, {len(queries)} queries; {len(skipped)} skipped pages, {skipped_qrels} qrels targeting skipped pages")


def gold(ds):
    meta = confined(ROOT / "meta" / ds)
    manifest = json.loads(confined(meta / "manifest.json").read_text())
    for name, digest in manifest["metadata_blake3"].items():
        require(blake3.blake3(confined(meta / name).read_bytes()).hexdigest() == digest, "metadata changed since materialization")
    with confined(meta / "docmap.tsv").open() as f:
        docmap = {r["key"]: r["docid"] for r in csv.DictReader(f, delimiter="\t")}
    with confined(meta / "qrels.tsv").open() as f:
        qrels = [ir.Qrel(r["query-id"], r["corpus-id"], int(r["score"])) for r in csv.DictReader(f, delimiter="\t")]
    return manifest, docmap, qrels


def read_run(path):
    rows = jsonlines(confined(path).read_text().splitlines())
    require(len(rows) >= 3 and rows[0].get("type") == "env" and rows[-1].get("type") == "complete", "incomplete run")
    require(rows[0].get("schema") == 1, "unsupported run schema")
    body = rows[1:-1]
    require(all(r.get("type") == "query" and r["mode"] in MODES for r in body), "invalid query row")
    require(len(body) == rows[-1]["queries"], "incomplete query count")
    require(len({r["qid"] for r in body}) == len(body), "duplicate query")
    require(len({r["mode"] for r in body}) == 1, "mixed mode file")
    return rows[0], {r["qid"]:r for r in body}, rows[-1]


def ranking(row, docmap, pages=False):
    hits = row["ranked_pages"] if pages else row["ranked"]
    require(isinstance(hits, list), "missing page rankings")
    docs = []
    for expected_rank, hit in enumerate(hits, 1):
        require(hit["rank"] == expected_rank, "non-contiguous ranks")
        require(hit["key"] in docmap, "unmapped key")
        doc = docmap[hit["key"]]
        if pages:
            require(hit["locator"]["kind"] == "page_text" and hit["locator"]["page"] >= 1, "non-text page locator")
            doc = page_id(doc, hit["locator"]["page"])
        require(doc not in docs, "duplicate result after collapse")
        docs.append(doc)
    return docs


def metrics(qrels, rankings):
    require(set(rankings) == {q.query_id for q in qrels}, "missing or extra queries")
    # Raw fusion scores can tie. -rank is strictly ordered and preserves the
    # actual engine ranking, avoiding trec_eval's doc-ID tie breaker.
    runs = [ir.ScoredDoc(q, d, -float(rank)) for q, docs in rankings.items() for rank, d in enumerate(docs, 1)]
    return {str(k):v for k,v in ir.calc_aggregate(METRICS, qrels, runs).items()}


def sanity_band(value):
    require(0.60 <= value <= 0.68, f"lexical SciFact nDCG@10={value:.4f} outside [.60,.68]; investigate materialization/mapping before ranking conclusions")


def score(ds, paths):
    manifest, docmap, qrels = gold(ds)
    per_mode, envs, completions, rows_by_mode = {}, {}, {}, {}
    reference = None
    for path in paths:
        env, rows, completion = read_run(path)
        require(env["dataset"] == ds and env["dataset_manifest"] == manifest, "dataset provenance mismatch")
        identity = {k:env[k] for k in ("git_sha", "profile_id", "depth", "lexical_weight", "manifest_blake3", "run_id", "threads", "dirty")}
        if reference is None:
            reference = identity
        require(identity == reference, "cannot combine different runs/profiles/settings")
        mode = next(iter(rows.values()))["mode"]
        require(mode not in per_mode, "duplicate mode")
        rankings = {q:ranking(row, docmap, ds == "irpapers-pdf") for q,row in rows.items()}
        per_mode[mode] = metrics(qrels, rankings)
        if ds == "scifact" and mode == "lexical" and not env.get("competitor"):
            sanity_band(per_mode[mode]["nDCG@10"])
        if ds == "irpapers-pdf":
            # Resource recall and physical-page correctness are different views.
            paper_gold = sorted({(q.query_id, q.doc_id.split("#")[0], q.relevance) for q in qrels})
            paper_metrics = metrics([ir.Qrel(*r) for r in paper_gold], {q:ranking(r,docmap) for q,r in rows.items()})
            per_mode[mode] = {"page":per_mode[mode], "resource":paper_metrics}
        envs[mode], completions[mode], rows_by_mode[mode] = env, completion, rows
    require(per_mode, "no run files")
    positive = {}
    for q in qrels:
        if q.relevance > 0:
            positive.setdefault(q.query_id, set()).add(q.doc_id)
    join, false_counts, top_counts = [], dict.fromkeys(per_mode,0), dict.fromkeys(per_mode,0)
    for qid, relevant in sorted(positive.items()):
        ranks, tops, false = {}, {}, {}
        for mode, rows in rows_by_mode.items():
            docs = ranking(rows[qid], docmap, ds == "irpapers-pdf")
            ranks[mode] = {doc:i for i,doc in enumerate(docs,1)}
            tops[mode] = rows[qid]["ranked_pages" if ds == "irpapers-pdf" else "ranked"][0]["key"] if docs else None
            false[mode] = (docs[0] not in relevant) if docs else None
            top_counts[mode] += bool(docs)
            false_counts[mode] += bool(false[mode])
        join.append({"qid":qid, "gold":[{"docid":d, "rank":{m:ranks.get(m,{}).get(d) for m in MODES}} for d in sorted(relevant)], "top1_key":tops, "top1_nonrelevant":false})
    run_id = reference["run_id"]
    out = confined(ROOT / "results" / ds / (run_id + ".json"))
    diagnostics = {m:{"top_result_false_evidence_rate":false_counts[m]/top_counts[m] if top_counts[m] else None,
                       "queries_with_result":top_counts[m], "queries_without_result":len(positive)-top_counts[m],
                       "queries_under_100_resources":sum(len(r["ranked"])<100 for r in rows_by_mode[m].values())} for m in per_mode}
    result = {"dataset":ds, "smoke":ds=="nanoscifact", "metrics":per_mode, "diagnostics":diagnostics,
              "environment":envs, "completion":completions, "scoring":f"ir_measures=={version('ir-measures')}; pytrec_eval (pytrec-eval-terrier)=={version('pytrec-eval-terrier')}; graded linear nDCG; relevance >= 1"}
    result["empty_transcriptions"] = manifest.get("empty_transcriptions", {"skipped_pages":0, "qrels_targeting_skipped_pages":0})
    write_json(out, result)
    log = confined(out.with_suffix(".queries.jsonl"))
    log.write_text("".join(json.dumps(row, sort_keys=True)+"\n" for row in join))
    print(out)
    print(f"skipped pages: {result['empty_transcriptions']['skipped_pages']}; qrels targeting skipped pages: {result['empty_transcriptions']['qrels_targeting_skipped_pages']}")


def determinism(left, right):
    a, ar, _ = read_run(left)
    b, br, _ = read_run(right)
    for key in ("dataset", "manifest_blake3", "profile_id", "depth", "lexical_weight", "git_sha"):
        require(a[key] == b[key], f"different {key}")
    require(b["index"]["sync"]["segments_embedded"] == 0, "comparison run did not use a warm cache")
    require(set(ar) == set(br), "query sets differ")
    for q in ar:
        require(ar[q]["mode"] == br[q]["mode"], "mode mismatch")
        for field in ("ranked", "ranked_pages"):
            def signature(row):
                return [(h["key"],h["rank"],h["locator"]) for h in (row[field] or [])]
            require(signature(ar[q]) == signature(br[q]), f"nondeterministic {q}/{field}")
    print(f"deterministic: {len(ar)} queries; identical ranked keys, ranks and locators")


def report(paths):
    print("| Dataset | Mode | Precision / dims | nDCG@10 | MRR@10 | R@1 | R@5 | R@10 | R@20 | R@100 |")
    print("|---|---|---|---:|---:|---:|---:|---:|---:|---:|")
    skip_notes = []
    for path in paths:
        r = json.loads(confined(path).read_text())
        if r["dataset"] == "irpapers":
            skipped = r["empty_transcriptions"]
            skip_notes.append(f"IRPAPERS skipped pages: {skipped['skipped_pages']}; qrels targeting skipped pages: {skipped['qrels_targeting_skipped_pages']}")
        for mode, metrics_ in r["metrics"].items():
            env = r["environment"][mode]
            profile = f"{env.get('precision', '?')}/{env.get('dimension', '?')}"
            views = metrics_ if r["dataset"] == "irpapers-pdf" else {"":metrics_}
            for view, values in views.items():
                label = r["dataset"] + (" smoke" if r["smoke"] else "") + (" " + view if view else "")
                print("| " + label + " | " + mode + " | " + profile + " | " + " | ".join(f"{values[str(m)]:.4f}" for m in METRICS) + " |")
    for note in skip_notes:
        print("\n" + note)
    print("\nPublished references (different systems): SciFact BM25/TAS-B nDCG@10 .665/.643; NFCorpus .325/.319; IRPAPERS text hybrid R@1/5/20 .46/.78/.91. NanoSciFact is smoke only.")


class Tests(unittest.TestCase):
    def test_hand_computed_metrics_with_miss(self):
        # q1: grade-1 b at 1, nonrelevant x at 2, grade-2 a at 3.
        # trec_eval nDCG uses linear gains: DCG=1+2/log2(4)=2.
        # q2: grade-1 c first found at rank 2. q3: no returned results.
        qrels = [ir.Qrel("q1","a",2),ir.Qrel("q1","b",1),ir.Qrel("q2","c",1),ir.Qrel("q3","d",1)]
        m = metrics(qrels,{"q1":["b","x","a"],"q2":["x","c"],"q3":[]})
        self.assertAlmostEqual(m["nDCG@10"],(2/(2+1/math.log2(3))+1/math.log2(3))/3)
        self.assertAlmostEqual(m["RR@10"],(1+0.5)/3)
        self.assertAlmostEqual(m["R@1"],0.5/3)
        self.assertAlmostEqual(m["R@5"],2/3)
        with self.assertRaises(ValueError): metrics(qrels,{"q1":[]})

    def test_rank_mapping_and_page_identity(self):
        row={"ranked_pages":[{"key":"k","rank":1,"locator":{"kind":"page_text","page":2}}]}
        self.assertEqual(ranking(row,{"k":"7"},True),["7#2"])
        with self.assertRaises(ValueError): ranking(row,{},True)
        row["ranked_pages"] *= 2
        with self.assertRaises(ValueError): ranking(row,{"k":"7"},True)

    def test_irpapers_repeated_page_questions(self):
        rows = [{"dataset_id":"7_2", "pdf_id":"7", "page_number":"2", "question":"one"},
                {"dataset_id":"7_2", "pdf_id":"7", "page_number":"2", "question":"two"}]
        queries, qrels = ir_queries(rows)
        self.assertEqual(len({q["qid"] for q in queries}), 2)
        self.assertEqual([r[1] for r in qrels], ["7#2", "7#2"])

    def test_irpapers_text_skip_materialization_score_and_validation(self):
        ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            raw_dir = root / "raw/irpapers"
            raw_dir.mkdir(parents=True)
            fields = ["pdf_id_x", "pdf_id_y", "page_number", "pdf_title", "transcription", "base64_str"]
            rows = [dict(zip(fields, ["7", "", "1", "Title", "First page", "not decoded"])),
                    dict(zip(fields, ["7", "7.0", "2", "Title", "Second page", "not decoded"]))]
            def write_docs():
                with (raw_dir / "docs.csv").open("w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fields)
                    writer.writeheader()
                    writer.writerows(rows)
            write_docs()
            with (raw_dir / "queries.csv").open("w", newline="") as f:
                query_fields = ["dataset_id", "pdf_id", "page_number", "question"]
                writer = csv.DictWriter(f, fieldnames=query_fields)
                writer.writeheader()
                writer.writerows([dict(zip(query_fields, ["7_2", "7", "2", "First question"])),
                                  dict(zip(query_fields, ["7_2", "7", "2", "Second question"]))])
            with mock.patch.dict(globals(), ROOT=root):
                docs, queries, qrels = inputs("irpapers")
                self.assertEqual(list(docs), [("7#1", b"Title\n\nFirst page", "txt"),
                                              ("7#2", b"Title\n\nSecond page", "txt")])
                self.assertEqual(len({q["qid"] for q in queries}), 2)
                self.assertTrue(all(r[1] == "7#2" for r in qrels))
                rows[1]["transcription"] = " \n\t"
                rows.append(dict(zip(fields, ["7", "", "3", "Title", "", "not decoded"])))
                write_docs()
                cfg = {"kind":"irpapers", "documents":3, "queries":2,
                       "files":[{"name":name, "sha256":sha256(raw_dir/name)} for name in ("docs.csv", "queries.csv")]}
                with mock.patch(__name__+".config", return_value=cfg), mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    materialize("irpapers")
                    manifest, docmap, gold_qrels = gold("irpapers")
                    skipped = manifest["empty_transcriptions"]
                    self.assertEqual(skipped, {"skipped_page_keys":sorted(blake3.blake3(d.encode()).hexdigest()[:16] for d in ("7#2", "7#3")),
                                               "skipped_pages":2, "qrels_targeting_skipped_pages":2,
                                               "raw_rows":3, "materialized_pages":1})
                    self.assertEqual(set(docmap.values()), {"7#1"})
                    self.assertEqual(len(list((root/"corpus/irpapers").iterdir())), 1)
                    self.assertEqual(len(gold_qrels), 2)
                    env = {"type":"env", "schema":1, "dataset":"irpapers", "dataset_manifest":manifest,
                           "git_sha":"test", "profile_id":"test", "depth":200, "lexical_weight":None,
                           "manifest_blake3":"test", "run_id":"test", "threads":1, "dirty":False}
                    run = root/"run.jsonl"
                    run_rows = [{"type":"query", "qid":q["qid"], "mode":"lexical", "ranked":hits, "ranked_pages":None}
                                for q, hits in zip(queries, ([{"key":next(iter(docmap)), "rank":1}], []))]
                    run.write_text("".join(json.dumps(r)+"\n" for r in [env, *run_rows, {"type":"complete", "queries":2}]))
                    score("irpapers", [run])
                    result_path = root/"results/irpapers/test.json"
                    result = json.loads(result_path.read_text())
                    self.assertTrue(all(value == 0 for value in result["metrics"]["lexical"].values()))
                    self.assertEqual(result["empty_transcriptions"], skipped)
                    self.assertEqual(result["diagnostics"]["lexical"]["queries_without_result"], 1)
                    report([result_path])
                    self.assertIn("skipped pages: 2; qrels targeting skipped pages: 2", stdout.getvalue())
                    cfg["documents"] = 2
                    with self.assertRaisesRegex(ValueError, "unexpected document count"):
                        materialize("irpapers")
                    cfg["documents"] = 3
                    rows[2]["page_number"] = "2"
                    write_docs()
                    cfg["files"][0]["sha256"] = sha256(raw_dir/"docs.csv")
                    with self.assertRaisesRegex(ValueError, "duplicate or invalid doc ID"):
                        materialize("irpapers")
                    rows[2]["page_number"] = "3"
                    rows[1]["page_number"] = "4"
                    write_docs()
                    cfg["files"][0]["sha256"] = sha256(raw_dir/"docs.csv")
                    with self.assertRaisesRegex(ValueError, "qrel references missing document"):
                        materialize("irpapers")
                    rows[1]["page_number"] = "2"
                rows[1]["pdf_id_y"] = "7.5"
                write_docs()
                with self.assertRaises(ValueError): list(inputs("irpapers")[0])
                rows[1]["pdf_id_y"] = ""
                rows[1]["page_number"] = "0"
                write_docs()
                with self.assertRaisesRegex(ValueError, "invalid one-based paper/page"):
                    list(inputs("irpapers")[0])
                rows[1]["page_number"] = "2"
                write_docs()
                with (raw_dir/"docs.csv").open("a") as f:
                    f.write("7,,4,Title\n")
                with self.assertRaisesRegex(ValueError, "malformed IRPAPERS row"):
                    list(inputs("irpapers")[0])

    def test_qrel_on_skipped_page_preserves_metric_denominators(self):
        qrels = [ir.Qrel("answerable", "7#1", 1), ir.Qrel("skipped", "7#2", 1)]
        for skipped_ranking in ([], ["7#1"]):
            values = metrics(qrels, {"answerable":["7#1"], "skipped":skipped_ranking})
            self.assertTrue(all(value == 0.5 for value in values.values()))

    def test_materialize_score_report_and_integrity(self):
        ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            archive = root / "raw/nfcorpus/nfcorpus.zip"
            archive.parent.mkdir(parents=True)
            with zipfile.ZipFile(archive,"w") as z:
                z.writestr("nfcorpus/corpus.jsonl", json.dumps({"_id":"d", "title":"Title", "text":"Body"})+"\n")
                z.writestr("nfcorpus/queries.jsonl", json.dumps({"_id":"q", "text":"Body"})+"\n")
                z.writestr("nfcorpus/qrels/test.tsv", "query-id\tcorpus-id\tscore\nq\td\t2\n")
            cfg={"kind":"beir", "documents":1, "queries":1, "files":[{"name":"nfcorpus.zip", "sha256":sha256(archive)}]}
            with mock.patch.dict(globals(), ROOT=root), mock.patch(__name__+".config",return_value=cfg), mock.patch("sys.stdout",new_callable=io.StringIO) as stdout:
                materialize("nfcorpus")
                manifest, docmap, _ = gold("nfcorpus")
                key=next(iter(docmap))
                path=root/"corpus/nfcorpus"/(key+".txt")
                self.assertEqual(path.read_text(),"Title\n\nBody")
                before=path.stat().st_mtime_ns
                materialize("nfcorpus")
                self.assertEqual(before,path.stat().st_mtime_ns)
                env={"type":"env", "schema":1, "dataset":"nfcorpus", "dataset_manifest":manifest, "git_sha":"test", "profile_id":"test", "depth":200, "lexical_weight":None, "manifest_blake3":"test", "run_id":"test", "threads":1, "dirty":False}
                run=root/"run.jsonl"
                row={"type":"query", "qid":"q", "mode":"lexical", "ranked":[{"key":key,"rank":1}], "ranked_pages":None}
                run.write_text("".join(json.dumps(r)+"\n" for r in [env,row,{"type":"complete","queries":1}]))
                score("nfcorpus",[run])
                result=root/"results/nfcorpus/test.json"
                self.assertEqual(json.loads(result.read_text())["metrics"]["lexical"]["nDCG@10"],1.0)
                report([result])
                self.assertIn("1.0000",stdout.getvalue())
                (root/"meta/nfcorpus/docmap.tsv").write_text("corrupt")
                with self.assertRaises(ValueError): gold("nfcorpus")

    def test_pdf_top1_uses_scored_page_ranking(self):
        ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            env = {"dataset":"irpapers-pdf", "dataset_manifest":{}, "git_sha":"test", "profile_id":"test", "depth":200, "lexical_weight":None, "manifest_blake3":"test", "run_id":"test", "threads":1, "dirty":False}
            row = {"mode":"hybrid", "ranked":[{"key":"nonpage", "rank":1}, {"key":"page", "rank":2}],
                   "ranked_pages":[{"key":"page", "rank":1, "locator":{"kind":"page_text", "page":2}}]}
            with mock.patch.dict(globals(), ROOT=root), mock.patch(__name__+".gold", return_value=({}, {"nonpage":"1", "page":"2"}, [ir.Qrel("q", "2#2", 1)])), mock.patch(__name__+".read_run", return_value=(env, {"q":row}, {})), mock.patch("sys.stdout", new_callable=io.StringIO):
                score("irpapers-pdf", [root/"run.jsonl"])
                result = json.loads((root/"results/irpapers-pdf/test.json").read_text())
                joined = json.loads((root/"results/irpapers-pdf/test.queries.jsonl").read_text())
                self.assertEqual(joined["top1_key"], {"hybrid":"page"})
                self.assertEqual(joined["top1_nonrelevant"], {"hybrid":False})
                self.assertEqual(result["metrics"]["hybrid"]["page"]["R@1"], 1.0)
                self.assertEqual(result["metrics"]["hybrid"]["resource"]["R@1"], 0.0)
                self.assertIn(f"pytrec_eval (pytrec-eval-terrier)=={version('pytrec-eval-terrier')}", result["scoring"])

    def test_sanity_band(self):
        sanity_band(.665)
        for value in (.599,.681,float("nan")):
            with self.assertRaises(ValueError): sanity_band(value)

    def test_confined(self):
        with self.assertRaises(ValueError): confined(Path.home()/"private")
        with self.assertRaises(ValueError): confined(ROOT/".."/"private")
        ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            link=Path(directory)/"escape"
            link.symlink_to(Path.home(),target_is_directory=True)
            with self.assertRaises(ValueError): confined(link/"file")

    def test_reject_incomplete_run(self):
        ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            path=Path(directory)/"run.jsonl"
            path.write_text('{"type":"env"}\n')
            with self.assertRaises(ValueError): read_run(path)


def main():
    import compete
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    datasets = list(tomllib.loads(CONFIG.read_text()))
    for name in ("fetch", "materialize", "score"):
        p = sub.add_parser(name)
        p.add_argument("dataset", choices=datasets)
        if name == "score": p.add_argument("runs", nargs="+", type=Path)
    p = sub.add_parser("report")
    p.add_argument("results", nargs="+", type=Path)
    p = sub.add_parser("determinism")
    p.add_argument("left", type=Path)
    p.add_argument("right", type=Path)
    sub.add_parser("test")
    compete.parser(sub)
    args = parser.parse_args()
    if args.command == "fetch": fetch(args.dataset)
    elif args.command == "materialize": materialize(args.dataset)
    elif args.command == "score": score(args.dataset, args.runs)
    elif args.command == "report": report(args.results)
    elif args.command == "determinism": determinism(args.left,args.right)
    elif args.command == "compete": compete.run(args)
    elif args.command == "offline-check": print(compete.network_probe())
    else:
        import test_compete
        suite = unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromTestCase(Tests),
                                   unittest.defaultTestLoader.loadTestsFromModule(test_compete)])
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        sys.exit(not result.wasSuccessful())


if __name__ == "__main__":
    main()
