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
import urllib.parse
import zipfile
from pathlib import Path

import blake3
import ir_measures as ir

ROOT = Path.home() / ".cache/seyk-bench"
CONFIG = Path(__file__).with_name("datasets.toml")
MODES = ("hybrid", "semantic", "lexical")
METRICS = [ir.nDCG @ 10, ir.RR @ 10, *(ir.R @ k for k in (1, 5, 10, 20, 100))]
PDF_PROOF = "pdf-policy-miss-v2"
PDF_FAILURES = {"pdf_parse_failed", "pdf_parser_panicked"}
PDF_LIMIT = 33_554_432
PDF_NO_TEXT = {"no usable embedded text; PDF rendering is not compiled in (feature `pdf-render`)",
               "no usable embedded text; image embedding is disabled"}
METADATA = {"docmap.tsv", "queries.jsonl", "qrels.tsv"}


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
    if ds == "irpapers-pdf":
        require(manifest["dataset"] == ds and manifest["source"] == config(ds), "PDF source config mismatch")
        require(set(manifest["metadata_blake3"]) == METADATA, "PDF metadata identities mismatch")
    for name, digest in manifest["metadata_blake3"].items():
        require(blake3.blake3(confined(meta / name).read_bytes()).hexdigest() == digest, "metadata changed since materialization")
    with confined(meta / "docmap.tsv").open() as f:
        mapping = list(csv.DictReader(f, delimiter="\t"))
        docmap = {r["key"]: r["docid"] for r in mapping}
    with confined(meta / "qrels.tsv").open() as f:
        qrels = [ir.Qrel(r["query-id"], r["corpus-id"], int(r["score"])) for r in csv.DictReader(f, delimiter="\t")]
    if ds == "irpapers-pdf":
        require(len(mapping) == len(docmap) == config(ds)["documents"]
                and len(set(docmap.values())) == len(docmap), "PDF duplicate/missing document identity")
        require(all(doc.isascii() and doc.isdecimal() and int(doc) > 0 and str(int(doc)) == doc
                    and blake3.blake3(doc.encode()).hexdigest()[:16] == key for key, doc in docmap.items()), "invalid PDF identity")
        require(set(manifest["files"]) == {k+".pdf" for k in docmap}, "PDF manifest/docmap mismatch")
        queries = jsonlines(confined(meta / "queries.jsonl").read_text().splitlines())
        qids = {q["qid"] for q in queries}
        require(len(qids) == len(queries) == config(ds)["queries"], "PDF question count/identity mismatch")
        require(len(qrels) == len({(q.query_id, q.doc_id) for q in qrels}) == len(queries)
                and {q.query_id for q in qrels} == qids, "PDF original qrels mismatch")
        for q in qrels:
            paper, page = q.doc_id.split("#")
            require(paper in docmap.values() and page_id(paper, page) == q.doc_id and q.relevance == 1,
                    "invalid original PDF qrel")
    return manifest, docmap, qrels


def pdf_coverage(env, manifest, docmap, qrels):
    """Check runner-observed evidence; the scorer does not remeasure source sizes."""
    proof = env.get("pdf_coverage")
    require(isinstance(proof, dict) and proof.get("proof") == PDF_PROOF
            and proof.get("limit") == PDF_LIMIT, "missing/unknown PDF policy proof or default limit")
    require(set(proof) == {"proof", "limit", "full_resources", "eligible_keys", "exclusions", "omitted_text_pages"},
            "unknown PDF proof fields/counts")
    session = env.get("pdf_cache_session")
    require(isinstance(session, str) and 1 <= len(session) <= 96
            and all(c.isascii() and (c.isalnum() or c == '-') for c in session), "invalid PDF cache session")
    require(proof.get("full_resources") == len(docmap), "PDF full resource count mismatch")
    exclusions = proof.get("exclusions")
    require(isinstance(exclusions, list), "missing PDF exclusions")
    excluded = set()
    for item in exclusions:
        require(set(item) == {"key", "blake3", "size_bytes", "reason"}, "unknown exclusion fields")
        key = item["key"]
        require(key in docmap and key not in excluded, "unknown/duplicate PDF exclusion")
        require(item.get("reason") == "encoded_size_exceeds_default"
                and type(item.get("size_bytes")) is int and item["size_bytes"] > PDF_LIMIT
                and item.get("blake3") == manifest["files"][key+".pdf"]
                and isinstance(item["blake3"], str) and len(item["blake3"]) == 64
                and all(c in "0123456789abcdef" for c in item["blake3"]), "invalid PDF exclusion evidence")
        excluded.add(key)
    require([e["key"] for e in exclusions] == sorted(excluded), "unsorted PDF exclusions")
    eligible = set(docmap) - excluded
    require(proof.get("eligible_keys") == sorted(eligible), "PDF eligible identity mismatch")
    omitted = proof.get("omitted_text_pages")
    require(isinstance(omitted, list), "missing omitted text pages")
    pages = set()
    for item in omitted:
        require(set(item) == {"key", "page", "reason"}, "unknown omitted-page fields")
        key, page = item["key"], item["page"]
        require(key in eligible and type(page) is int and page > 0
                and item.get("reason") in PDF_FAILURES | {"no_usable_embedded_text"}
                and (key, page) not in pages, "invalid omitted PDF page")
        pages.add((key, page))
    require([(p["key"], p["page"]) for p in omitted] == sorted(pages), "unsorted PDF pages")
    index = env.get("index", {})
    sync = index.get("sync", {})
    require(index.get("resources_removed") == 0 and sync.get("resources_discovered") == len(eligible)
            and sync.get("resources_updated") == len(eligible) and sync.get("resources_unchanged") == 0
            and sync.get("skipped") == [] and sync.get("skipped_counts") == [], "PDF preparation was not fresh/complete")
    diagnostics = []
    for item in sync.get("page_diagnostics", []):
        uri = urllib.parse.urlsplit(item["resource"])
        path = Path(urllib.parse.unquote(uri.path))
        code, reason, page = item.get("code"), item.get("reason"), item.get("page")
        require(uri.scheme == "file" and not uri.netloc and not uri.query and not uri.fragment
                and path.suffix == ".pdf" and path.stem in eligible
                and type(page) is int and page > 0, "invalid PDF diagnostic identity/page")
        if code == "other":
            require(reason in PDF_NO_TEXT, "fatal PDF diagnostic")
            classified = "no_usable_embedded_text"
        else:
            require(code in PDF_FAILURES and reason == code, "fatal/unsafe PDF diagnostic")
            classified = code
        diagnostics.append((path.stem, page, classified))
    outcomes = {(p["key"],p["page"],p["reason"]) for p in omitted}
    require(len(diagnostics) == len(outcomes) and set(diagnostics) == outcomes, "PDF diagnostic evidence mismatch")
    excluded_papers = {docmap[k] for k in excluded}
    affected = [q for q in qrels if q.relevance > 0 and q.doc_id.split("#")[0] in excluded_papers]
    omitted_ids = {page_id(docmap[k], p) for k, p in pages}
    omitted_qrels = [q for q in qrels if q.relevance > 0 and q.doc_id in omitted_ids]
    failed = [p for p in omitted if p["reason"] in PDF_FAILURES]
    failed_ids = {page_id(docmap[p["key"]], p["page"]) for p in failed}
    failed_qrels = [q for q in qrels if q.relevance > 0 and q.doc_id in failed_ids]
    return {"proof":PDF_PROOF, "size_evidence":"runner-observed length and BLAKE3; scorer consistency-checked",
            "original_queries":len({q.query_id for q in qrels}), "original_qrels":len(qrels),
            "full_resources":len(docmap), "eligible_resources":len(eligible), "excluded_resources":len(excluded),
            "exclusions":exclusions, "excluded_physical_pages":None,
            "positive_qrels_targeting_exclusions":len(affected),
            "questions_targeting_exclusions":len({q.query_id for q in affected}),
            "omitted_text_pages":omitted, "parser_failed_pages":failed,
            "parser_failure_counts":{reason:sum(p["reason"] == reason for p in failed) for reason in sorted(PDF_FAILURES)},
            "positive_qrels_targeting_parser_failures":len(failed_qrels),
            "questions_targeting_parser_failures":len({q.query_id for q in failed_qrels}),
            "positive_qrels_targeting_omitted_pages":len(omitted_qrels),
            "questions_targeting_omitted_pages":len({q.query_id for q in omitted_qrels})}, eligible


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
    coverage = None
    for path in paths:
        env, rows, completion = read_run(path)
        require(env["dataset"] == ds and env["dataset_manifest"] == manifest, "dataset provenance mismatch")
        identity = {k:env[k] for k in ("git_sha", "profile_id", "depth", "lexical_weight", "manifest_blake3", "run_id", "threads", "dirty")}
        if ds == "irpapers-pdf":
            checked, eligible = pdf_coverage(env, manifest, docmap, qrels)
            if coverage is None:
                coverage = checked
            require(coverage == checked, "different PDF coverage")
            identity.update({k:env[k] for k in ("pdf_coverage", "pdf_cache_session", "profile", "precision", "dimension")})
            expected_manifest = blake3.blake3(confined(ROOT / "meta" / ds / "manifest.json").read_bytes()).hexdigest()
            require(env["manifest_blake3"] == expected_manifest, "PDF manifest digest mismatch")
            for row in rows.values():
                for view in ("ranked", "ranked_pages"):
                    require(isinstance(row.get(view), list) and all(h["key"] in eligible for h in row[view]),
                            "excluded/unknown PDF hit")
                    omitted_pages = {(p["key"],p["page"]) for p in checked["omitted_text_pages"]}
                    for hit in row[view]:
                        locator = hit.get("locator", {})
                        page = locator.get("page")
                        require(locator.get("kind") in ("page_text", "page") and type(page) is int and page > 0,
                                "invalid PDF hit page")
                        require((hit["key"],page) not in omitted_pages, "omitted/failed PDF page hit")
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
    require(ds != "irpapers-pdf" or set(per_mode) == set(MODES), "PDF aggregate requires all three modes")
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
    if ds == "irpapers-pdf":
        result["pdf_coverage"] = coverage
        result["diagnostic_view"] = "physical-page; top1_nonrelevant is page-based"
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
        elif r["dataset"] == "irpapers-pdf":
            coverage = r["pdf_coverage"]
            skip_notes.append(f"IRPAPERS PDF original queries: {coverage['original_queries']}; original qrels: {coverage['original_qrels']}; "
                              f"resources: full {coverage['full_resources']}, eligible {coverage['eligible_resources']}, excluded {coverage['excluded_resources']}; "
                              f"known omitted text pages: {len(coverage['omitted_text_pages'])}; "
                              f"positive qrels targeting exclusions: {coverage['positive_qrels_targeting_exclusions']}; "
                              f"distinct questions targeting exclusions: {coverage['questions_targeting_exclusions']}; "
                              f"positive qrels targeting omitted pages: {coverage['positive_qrels_targeting_omitted_pages']}; "
                              f"distinct questions targeting omitted pages: {coverage['questions_targeting_omitted_pages']}; "
                              "excluded physical pages: unknown")
            skip_notes.append(f"IRPAPERS PDF parser failures by reason: {json.dumps(coverage['parser_failure_counts'], sort_keys=True)}; "
                              f"positive qrels targeting parser failures: {coverage['positive_qrels_targeting_parser_failures']}; "
                              f"distinct questions targeting parser failures: {coverage['questions_targeting_parser_failures']}; "
                              f"opaque key/physical page/reason: {json.dumps(coverage['parser_failed_pages'], sort_keys=True)}")
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
    def pdf_fixture(self):
        docmap = {blake3.blake3(d.encode()).hexdigest()[:16]:d for d in ("1", "2", "3")}
        a, b, c = [next(k for k, v in docmap.items() if v == d) for d in ("1", "2", "3")]
        manifest = {"files":{k+".pdf":"a"*64 for k in docmap}}
        qrels = [ir.Qrel("q1", "1#3", 1), ir.Qrel("q2", "2#1", 1), ir.Qrel("q3", "2#1", 1)]
        proof = {"proof":PDF_PROOF, "limit":PDF_LIMIT, "full_resources":3, "eligible_keys":sorted([a,c]),
                 "exclusions":[{"key":b, "blake3":"a"*64, "size_bytes":PDF_LIMIT+1, "reason":"encoded_size_exceeds_default"}],
                 "omitted_text_pages":[{"key":a, "page":2, "reason":"no_usable_embedded_text"}]}
        env = {"type":"env", "schema":1, "dataset":"irpapers-pdf", "dataset_manifest":manifest,
               "git_sha":"test", "profile_id":"test", "profile":{}, "precision":"q8", "dimension":256,
               "depth":200, "lexical_weight":None, "manifest_blake3":"test", "run_id":"test", "threads":1,
               "dirty":False, "pdf_cache_session":"synthetic", "pdf_coverage":proof,
               "index":{"resources_removed":0, "sync":{"resources_discovered":2, "resources_updated":2,
                   "resources_unchanged":0, "skipped":[], "skipped_counts":[], "segments_embedded":0,
                   "page_diagnostics":[{"resource":f"file:///public/{a}.pdf", "page":2, "code":"other",
                                        "reason":sorted(PDF_NO_TEXT)[0]}]}}}
        return env, manifest, docmap, qrels, (a,b,c)

    def test_pdf_parser_failures_are_misses_with_original_denominators(self):
        import copy
        env, manifest, docmap, qrels, (a,b,c) = self.pdf_fixture()
        qrels[:] = [ir.Qrel("q1", "1#2", 1), ir.Qrel("q2", "1#4", 1), ir.Qrel("q3", "2#1", 1)]
        env["pdf_coverage"]["omitted_text_pages"] = [
            {"key":a, "page":2, "reason":"pdf_parse_failed"},
            {"key":a, "page":4, "reason":"pdf_parser_panicked"}]
        env["index"]["sync"]["page_diagnostics"] = [
            {"resource":f"file:///public/{a}.pdf", "page":page, "code":reason, "reason":reason}
            for page,reason in [(2,"pdf_parse_failed"),(4,"pdf_parser_panicked")]]
        checked, _ = pdf_coverage(env, manifest, docmap, qrels)
        self.assertEqual(checked["parser_failure_counts"], {"pdf_parse_failed":1,"pdf_parser_panicked":1})
        self.assertEqual(checked["positive_qrels_targeting_parser_failures"], 2)
        self.assertEqual(checked["questions_targeting_parser_failures"], 2)
        self.assertEqual(checked["original_queries"], 3)
        ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            meta = root/"meta/irpapers-pdf"
            meta.mkdir(parents=True)
            data = json.dumps(manifest).encode()
            (meta/"manifest.json").write_bytes(data)
            env["manifest_blake3"] = blake3.blake3(data).hexdigest()
            def runs(hit_page=None, view=None):
                for mode in MODES:
                    rows = {q.query_id:{"mode":mode, "ranked":[], "ranked_pages":[]} for q in qrels}
                    if hit_page is not None:
                        hit = {"key":a, "rank":1, "locator":{"kind":"page_text", "page":hit_page}}
                        rows["q1"][view or "ranked"] = [hit]
                        if view is None: rows["q1"]["ranked_pages"] = [hit]
                    yield copy.deepcopy(env), rows, {}
            with mock.patch.dict(globals(), ROOT=root), mock.patch(__name__+".gold", return_value=(manifest,docmap,qrels)), mock.patch("sys.stdout", new_callable=io.StringIO):
                for page in (None, 3):
                    with mock.patch(__name__+".read_run", side_effect=runs(page)):
                        score("irpapers-pdf", list(MODES))
                    result = json.loads((root/"results/irpapers-pdf/test.json").read_text())
                    for mode in MODES:
                        self.assertTrue(all(v == 0 for v in result["metrics"][mode]["page"].values()))
                        self.assertTrue(all(abs(v - (0 if page is None else 1/3)) < 1e-9 for v in result["metrics"][mode]["resource"].values()))
                with mock.patch("sys.stdout", new_callable=io.StringIO) as printed:
                    report([root/"results/irpapers-pdf/test.json"])
                self.assertIn('"pdf_parse_failed": 1', printed.getvalue())
                self.assertIn('"pdf_parser_panicked": 1', printed.getvalue())
                self.assertIn("positive qrels targeting parser failures: 2; distinct questions targeting parser failures: 2", printed.getvalue())
                self.assertIn(f'"key": "{a}", "page": 2, "reason": "pdf_parse_failed"', printed.getvalue())
                for view in ("ranked", "ranked_pages"):
                    with mock.patch(__name__+".read_run", side_effect=runs(2,view)), self.assertRaisesRegex(ValueError,"omitted/failed"):
                        score("irpapers-pdf", list(MODES))
        for mutate in [
            lambda e:e["pdf_coverage"].update(proof="pdf-policy-miss-v1"),
            lambda e:e["pdf_coverage"]["omitted_text_pages"].append({"key":a,"page":2,"reason":"no_usable_embedded_text"}),
            lambda e:e["index"]["sync"]["page_diagnostics"][0].update(code="other"),
            lambda e:e["index"]["sync"]["page_diagnostics"][0].update(reason="dependency payload"),
            lambda e:e["index"]["sync"]["page_diagnostics"][0].update(page=True),
        ]:
            broken = copy.deepcopy(env)
            mutate(broken)
            with self.assertRaises(ValueError): pdf_coverage(broken,manifest,docmap,qrels)

    def test_pdf_evidence_refusals_and_affected_original_qrels(self):
        import copy
        env, manifest, docmap, qrels, (a,b,c) = self.pdf_fixture()
        report_, eligible = pdf_coverage(env, manifest, docmap, qrels)
        self.assertEqual(eligible, {a,c})
        self.assertEqual(report_["positive_qrels_targeting_exclusions"], 2)
        self.assertEqual(report_["questions_targeting_exclusions"], 2)
        self.assertEqual(report_["original_queries"], 3)
        self.assertEqual(report_["original_qrels"], 3)
        self.assertEqual(report_["questions_targeting_omitted_pages"], 0)
        self.assertIsNone(report_["excluded_physical_pages"])
        mutations = [
            lambda e:e["pdf_coverage"].update(limit=PDF_LIMIT+1),
            lambda e:e["pdf_coverage"].update(proof="unknown"),
            lambda e:e["pdf_coverage"].update(full_resources=2),
            lambda e:e["pdf_coverage"].update(questions_targeting_exclusions=0),
            lambda e:e["pdf_coverage"].update(eligible_keys=sorted([a,b])),
            lambda e:e["pdf_coverage"]["exclusions"][0].update(blake3="b"*64),
            lambda e:e["pdf_coverage"]["exclusions"][0].update(size_bytes=PDF_LIMIT),
            lambda e:e["pdf_coverage"]["exclusions"][0].update(reason="other"),
            lambda e:e["pdf_coverage"]["exclusions"].append(e["pdf_coverage"]["exclusions"][0]),
            lambda e:e["pdf_coverage"]["exclusions"][0].update(key="unknown"),
            lambda e:e["pdf_coverage"]["omitted_text_pages"][0].update(page=0),
            lambda e:e["pdf_coverage"]["omitted_text_pages"][0].update(key=b),
            lambda e:e["index"]["sync"].update(resources_updated=0, resources_unchanged=2),
            lambda e:e["index"]["sync"].update(skipped=[{}]),
            lambda e:e["index"]["sync"]["page_diagnostics"][0].update(code="pdf_parse_failed"),
            lambda e:e["index"]["sync"]["page_diagnostics"][0].update(reason="embedded text is not supported by this embedder"),
            lambda e:e["index"]["sync"].update(page_diagnostics=[]),
        ]
        for mutate in mutations:
            broken = copy.deepcopy(env)
            mutate(broken)
            with self.subTest(broken=broken), self.assertRaises(ValueError):
                pdf_coverage(broken, manifest, docmap, qrels)

    def test_pdf_all_modes_excluded_targets_and_empty_denominators(self):
        import copy
        ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            env, manifest, docmap, qrels, (a,b,c) = self.pdf_fixture()
            meta = root/"meta/irpapers-pdf"
            meta.mkdir(parents=True)
            data = json.dumps(manifest).encode()
            (meta/"manifest.json").write_bytes(data)
            env["manifest_blake3"] = blake3.blake3(data).hexdigest()
            paths = [root/(mode+".jsonl") for mode in MODES]
            def write_runs(empty=False, change=None):
                for path, mode in zip(paths, MODES):
                    rows = [{"type":"query", "mode":mode, "qid":q.query_id,
                             "ranked":[], "ranked_pages":[]} for q in qrels]
                    if not empty:
                        hit = {"key":a, "rank":1, "locator":{"kind":"page_text", "page":3}}
                        rows[0].update(ranked=[hit], ranked_pages=[hit])
                    local = copy.deepcopy(env)
                    if change: change(local, rows, mode)
                    path.write_text("".join(json.dumps(r)+"\n" for r in [local, *rows, {"type":"complete", "queries":3}]))
            with mock.patch.dict(globals(), ROOT=root), mock.patch(__name__+".gold", return_value=(manifest,docmap,qrels)), mock.patch("sys.stdout", new_callable=io.StringIO):
                for empty in (False, True):
                    write_runs(empty)
                    score("irpapers-pdf", paths)
                    result = json.loads((root/"results/irpapers-pdf/test.json").read_text())
                    for mode in MODES:
                        for view in ("page", "resource"):
                            self.assertTrue(all(abs(value - (0 if empty else 1/3)) < 1e-9 for value in result["metrics"][mode][view].values()))
                    self.assertEqual(result["pdf_coverage"]["questions_targeting_exclusions"], 2)
                    self.assertEqual(result["pdf_coverage"]["original_queries"], 3)
                    self.assertEqual(result["pdf_coverage"]["original_qrels"], 3)
                    self.assertEqual(result["pdf_coverage"]["questions_targeting_omitted_pages"], 0)
                with self.assertRaisesRegex(ValueError, "all three modes"):
                    score("irpapers-pdf", paths[:-1])
                changes = [
                    lambda e,r,m:r[0].update(qid="forged"),
                    lambda e,r,m:r[0].update(qid=r[1]["qid"]),
                    lambda e,r,m:e["pdf_coverage"].update(full_resources=4) if m == "semantic" else None,
                    lambda e,r,m:r[0].update(ranked=[{"key":b, "rank":1}]),
                    lambda e,r,m:r[0].update(ranked_pages=[{"key":b, "rank":1}]),
                ]
                for change in changes:
                    write_runs(change=change)
                    with self.assertRaises(ValueError): score("irpapers-pdf", paths)
                write_runs()
                paths[0].write_text("\n".join(paths[0].read_text().splitlines()[:-1])+"\n")
                with self.assertRaisesRegex(ValueError, "incomplete run"): score("irpapers-pdf", paths)

    def test_pdf_report_shows_exclusions_and_textless_gold_targets(self):
        ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            env, manifest, docmap, qrels, _ = self.pdf_fixture()
            qrels[0] = ir.Qrel("q1", "1#2", 1)  # Known textless page; keep it in gold.
            meta = root/"meta/irpapers-pdf"
            meta.mkdir(parents=True)
            data = json.dumps(manifest).encode()
            (meta/"manifest.json").write_bytes(data)
            env["manifest_blake3"] = blake3.blake3(data).hexdigest()
            runs = [(env, {q.query_id:{"mode":mode, "ranked":[], "ranked_pages":[]} for q in qrels}, {}) for mode in MODES]
            with mock.patch.dict(globals(), ROOT=root), mock.patch(__name__+".gold", return_value=(manifest,docmap,qrels)), mock.patch(__name__+".read_run", side_effect=runs):
                with mock.patch("sys.stdout", new_callable=io.StringIO):
                    score("irpapers-pdf", [root/(m+".jsonl") for m in MODES])
                path = root/"results/irpapers-pdf/test.json"
                result = json.loads(path.read_text())
                with mock.patch("sys.stdout", new_callable=io.StringIO) as stdout:
                    report([path])
                self.assertIn("IRPAPERS PDF original queries: 3; original qrels: 3; resources: full 3, eligible 2, excluded 1; "
                              "known omitted text pages: 1; positive qrels targeting exclusions: 2; distinct questions targeting exclusions: 2; "
                              "positive qrels targeting omitted pages: 1; distinct questions targeting omitted pages: 1; "
                              "excluded physical pages: unknown", stdout.getvalue())
                self.assertEqual(result["pdf_coverage"]["questions_targeting_omitted_pages"], 1)
                self.assertEqual(result["pdf_coverage"]["original_queries"], 3)
                self.assertEqual(result["pdf_coverage"]["original_qrels"], 3)
                for mode in MODES:
                    for view in ("page", "resource"):
                        self.assertTrue(all(value == 0 for value in result["metrics"][mode][view].values()))

    def test_pdf_asymmetric_paper_and_physical_page(self):
        gold = [ir.Qrel("q", "7#3", 1)]
        papers = [ir.Qrel("q", "7", 1)]
        for returned, expected in [("7#2", (1,0)), ("8#3", (0,0)), ("7#3", (1,1))]:
            self.assertEqual(metrics(papers,{"q":[returned.split('#')[0]]})["R@1"], expected[0])
            self.assertEqual(metrics(gold,{"q":[returned]})["R@1"], expected[1])
        # Textless physical page 2 remains gold; returning page 3 cannot satisfy it.
        self.assertEqual(metrics([ir.Qrel("q","7#2",1)], {"q":["7#3"]})["R@1"], 0)

    def test_pdf_gold_original_metadata_and_identity_contract(self):
        import copy
        ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=ROOT) as directory:
            root = Path(directory)
            _, manifest, docmap, qrels, _ = self.pdf_fixture()
            cfg = dict(config("irpapers-pdf"), documents=3, queries=3)
            manifest.update(dataset="irpapers-pdf", source=cfg)
            meta = root/"meta/irpapers-pdf"
            meta.mkdir(parents=True)
            original = {"docmap.tsv":"key\tdocid\n"+"".join(f"{k}\t{d}\n" for k,d in docmap.items()),
                        "queries.jsonl":"".join(json.dumps({"qid":q.query_id,"text":"public synthetic question"})+"\n" for q in qrels),
                        "qrels.tsv":"query-id\tcorpus-id\tscore\n"+"".join(f"{q.query_id}\t{q.doc_id}\t1\n" for q in qrels)}
            def write_metadata(changes=None, missing=None):
                contents = dict(original)
                if changes: contents.update(changes)
                local = copy.deepcopy(manifest)
                local["metadata_blake3"] = {n:blake3.blake3(b.encode()).hexdigest() for n,b in contents.items()}
                if missing: del local["metadata_blake3"][missing]
                for name, data in contents.items(): (meta/name).write_text(data)
                (meta/"manifest.json").write_text(json.dumps(local))
            with mock.patch.dict(globals(), ROOT=root), mock.patch(__name__+".config", return_value=cfg):
                write_metadata()
                self.assertEqual(len(gold("irpapers-pdf")[2]), 3)
                write_metadata(missing="qrels.tsv")
                with self.assertRaisesRegex(ValueError, "metadata identities"): gold("irpapers-pdf")
                duplicate = original["docmap.tsv"]+original["docmap.tsv"].splitlines()[1]+"\n"
                write_metadata({"docmap.tsv":duplicate})
                with self.assertRaisesRegex(ValueError, "document identity"): gold("irpapers-pdf")
                write_metadata({"docmap.tsv":original["docmap.tsv"].replace(next(iter(docmap)),"f"*16)})
                with self.assertRaisesRegex(ValueError, "invalid PDF identity"): gold("irpapers-pdf")
                write_metadata()
                (meta/"queries.jsonl").write_text("tampered")
                with self.assertRaisesRegex(ValueError, "metadata changed"): gold("irpapers-pdf")
                write_metadata({"qrels.tsv":"query-id\tcorpus-id\tscore\nq1\t1#3\t1\n"})
                with self.assertRaisesRegex(ValueError, "original qrels"): gold("irpapers-pdf")
                write_metadata({"queries.jsonl":""})
                with self.assertRaisesRegex(ValueError, "question count"): gold("irpapers-pdf")

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
            env, manifest, docmap, _, (a,b,c) = self.pdf_fixture()
            meta = root/"meta/irpapers-pdf"
            meta.mkdir(parents=True)
            data = json.dumps(manifest).encode()
            (meta/"manifest.json").write_bytes(data)
            env["manifest_blake3"] = blake3.blake3(data).hexdigest()
            row = {"ranked":[{"key":c, "rank":1, "locator":{"kind":"page_text", "page":1}},
                              {"key":a, "rank":2, "locator":{"kind":"page_text", "page":3}}],
                   "ranked_pages":[{"key":a, "rank":1, "locator":{"kind":"page_text", "page":3}}]}
            runs = [(env, {"q":dict(row, mode=mode)}, {}) for mode in MODES]
            with mock.patch.dict(globals(), ROOT=root), mock.patch(__name__+".gold", return_value=(manifest, docmap, [ir.Qrel("q", "1#3", 1)])), mock.patch(__name__+".read_run", side_effect=runs), mock.patch("sys.stdout", new_callable=io.StringIO):
                score("irpapers-pdf", [root/(m+".jsonl") for m in MODES])
                result = json.loads((root/"results/irpapers-pdf/test.json").read_text())
                joined = json.loads((root/"results/irpapers-pdf/test.queries.jsonl").read_text())
                self.assertEqual(joined["top1_key"], dict.fromkeys(MODES, a))
                self.assertEqual(joined["top1_nonrelevant"], dict.fromkeys(MODES, False))
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
