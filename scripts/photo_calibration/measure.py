"""Fit on A only; compare frozen candidates on B plus the untouched eval fixture."""
import hashlib
import json
from pathlib import Path

import blake3
import numpy as np

from build_set import ROOT, SEED, write_json

ASSETS = ROOT / "crates/seyk-embed/assets/photo-anchor"
PERCENTILES = (0.1, 0.25, 0.5, 1.0, 2.0)


def asset(dim):
    anchor = np.fromfile(ASSETS / f"photo_anchor_{dim}.f32.bin", dtype="<f4")
    table = np.fromfile(ASSETS / f"quant4_{dim}.f32.bin", dtype="<f4").reshape(2, dim)
    return anchor, table


def check_table(table):
    table = np.asarray(table, dtype=np.float32)
    if table.ndim != 2 or table.shape[0] != 2 or table.shape[1] % 2 or not np.isfinite(table).all() or (table[1] <= 0).any():
        raise ValueError("invalid quantization table")
    return table


def encode(images, anchor, table):
    offset, scale = check_table(table)
    centered = np.clip(images - anchor, offset, offset + np.float32(15) * scale)
    codes = np.clip(np.rint((centered - offset) / scale), 0, 15).astype(np.uint8)
    return (codes[:, ::2] << 4) | codes[:, 1::2]


def unpack(codes):
    values = np.empty((len(codes), codes.shape[1] * 2), dtype=np.float32)
    values[:, ::2] = codes >> 4
    values[:, 1::2] = codes & 15
    return values


def sequential_sum(values):
    return np.cumsum(values, axis=-1, dtype=np.float32)[..., -1]


def lut_scores(query, codes, anchor, table):
    offset, scale = check_table(table)
    bias = sequential_sum((anchor + offset) * query)
    weighted = scale * query
    pairs = (codes >> 4).astype(np.float32) * weighted[::2]
    pairs += (codes & 15).astype(np.float32) * weighted[1::2]
    return np.float32(bias + sequential_sum(pairs))


def decode_scores(query, codes, anchor, table):
    decoded = (anchor + table[0]) + unpack(codes) * table[1]
    return sequential_sum(decoded * query)


def dot_scores(query, images):
    """Match seyk_embed::dot's eight accumulators and final sequential reduction."""
    products = (images * query).reshape(len(images), -1, 8)
    lanes = np.cumsum(products, axis=1, dtype=np.float32)[:, -1, :]
    return sequential_sum(lanes)


def order(scores):
    return np.lexsort((np.arange(len(scores)), -scores))


def rerank(approx, exact, depth):
    shortlist = order(approx)[:depth]
    return shortlist[np.lexsort((shortlist, -exact[shortlist]))]


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    starts = np.cumsum(counts) - counts
    return (starts + (counts - 1) / 2)[inverse]


def spearman(a, b):
    a, b = ranks(a), ranks(b)
    if np.std(a) == 0 or np.std(b) == 0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float(np.corrcoef(a, b)[0, 1])


def ndcg(indices, labels):
    gains = np.exp2(labels) - 1
    discount = np.log2(np.arange(2, 12))
    ideal = np.sort(gains)[::-1][:10] / discount[:min(10, len(labels))]
    denominator = ideal.sum()
    if denominator <= 0:
        raise ValueError("query has no relevance labels in evaluation pool")
    return float((gains[indices[:10]] / discount[:min(10, len(indices))]).sum() / denominator)


def diagnostics(images, anchor, table):
    centered = images - anchor
    clipped = (centered < table[0]) | (centered > table[0] + np.float32(15) * table[1])
    rates = clipped.mean(axis=0)
    codes = encode(images, anchor, table)
    decoded = (anchor + table[0]) + unpack(codes) * table[1]
    return {"clip_rate_per_coordinate": rates.tolist(), "clip_rate_mean": float(rates.mean()),
            "clip_rate_max": float(rates.max()), "coordinates_over_3_percent": int((rates > .03).sum()),
            "reconstruction_mse": float(np.mean((decoded.astype(float) - images) ** 2))}


def fit_tables(images, queries, anchor, v1):
    """A has its own fit/validation partition; B and fixture vectors never enter."""
    cut = len(images) * 4 // 5
    train, validation = images[:cut] - anchor, images[cut:] - anchor
    if min(len(train), len(validation), len(queries)) == 0:
        raise ValueError("insufficient fitting data")
    weight = np.mean(queries.astype(float) ** 2, axis=0)
    lows = np.percentile(train, PERCENTILES, axis=0).astype(np.float32)
    highs = np.percentile(train, [100 - p for p in PERCENTILES], axis=0).astype(np.float32)

    def best(asymmetric):
        result = v1.copy()
        loss = np.full(len(anchor), np.inf)
        choices = np.zeros((2, len(anchor)))
        for lo_index, lo in enumerate(lows):
            for hi_index, hi in enumerate(highs):
                if not asymmetric and lo_index != hi_index:
                    continue
                scale = np.maximum((hi - lo) / np.float32(15), np.finfo(np.float32).tiny)
                quant = np.clip(np.rint((np.clip(validation, lo, hi) - lo) / scale), 0, 15)
                error = np.mean(((lo + quant * scale).astype(float) - validation) ** 2, axis=0)
                if asymmetric:
                    error *= weight
                improved = error < loss
                loss[improved] = error[improved]
                result[0, improved], result[1, improved] = lo[improved], scale[improved]
                choices[0, improved], choices[1, improved] = PERCENTILES[lo_index], PERCENTILES[hi_index]
        return result, choices.tolist()

    m1, p1 = best(False)
    m2, p2 = best(True)
    # Minimal widen uses all of A, without inspecting B or fixture clipping.
    centered = images - anchor
    outliers = ((centered < v1[0]) | (centered > v1[0] + np.float32(15) * v1[1])).mean(axis=0) > .03
    lo, hi = np.percentile(centered, [.5, 99.5], axis=0).astype(np.float32)
    lo = np.minimum(v1[0], lo)
    hi = np.maximum(v1[0] + np.float32(15) * v1[1], hi)
    m3 = v1.copy()
    m3[0, outliers] = lo[outliers]
    m3[1, outliers] = ((hi - lo) / np.float32(15))[outliers]
    return {"v1": (v1, None), "M1": (m1, p1), "M2": (m2, p2),
            "M3": (m3, {"widened_coordinates": np.flatnonzero(outliers).tolist()})}


def evaluate(images, queries, labels, eval_mask, anchor, table, expected=None):
    codes = encode(images, anchor, table)
    if expected is not None and not np.array_equal(codes, expected["codes"]):
        raise ValueError("v1 codes differ from production Rust encoder")
    recalls = {f"k{k}_R{r}": [] for k in (1, 5, 10) for r in sorted({k, 2 * k, 50})}
    mae, correlations, per_query = [], [], []
    decode_error = 0.0
    exact_order = True
    max_ndcg_drop = 0.0
    dim = len(anchor)
    for i, (query, relevance) in enumerate(zip(queries, labels)):
        approx, exact = lut_scores(query, codes, anchor, table), dot_scores(query, images)
        if expected is not None:
            for name, actual in (("q4", approx), ("f32", exact)):
                reference = np.asarray(expected[name][i], dtype=np.float32)
                if not np.array_equal(actual.view(np.uint32), reference.view(np.uint32)):
                    raise ValueError(f"v1 {name} score bits differ from production Rust at query {i}")
        error = float(np.max(np.abs(approx - decode_scores(query, codes, anchor, table))))
        decode_error = max(error, decode_error)
        if error > (5e-5 if dim == 256 else 1.5e-4):
            raise ValueError("accuracy B sequential-decode bound failed")
        baseline, approximate = order(exact), order(approx)
        for key, values in recalls.items():
            k, r = (int(value[1:]) for value in key.split("_"))
            values.append(len(set(baseline[:k]) & set(approximate[:r])) / k)
        qmae = float(np.mean(np.abs(approx[eval_mask].astype(float) - exact[eval_mask])))
        mae.append(qmae)
        correlations.append(spearman(approx[eval_mask], exact[eval_mask]))
        identical = {}
        for k in (1, 5, 10, 20, 50, 0, 281):
            depth = len(images) if k == 0 else min(len(images), max(50, 4 * k))
            rank = rerank(approx, exact, depth)
            limit = len(images) if k == 0 else min(k, len(images))
            identical[str(k)] = bool(np.array_equal(rank[:limit], baseline[:limit]))
        exact_order &= all(identical.values())
        drop = ndcg(baseline, relevance) - ndcg(rerank(approx, exact, min(len(images), 50)), relevance)
        max_ndcg_drop = max(max_ndcg_drop, drop)
        per_query.append({"mae": qmae, "identical_rerank": identical, "ndcg_drop_rerank50": drop})
    return {"recall": {key: float(np.mean(value)) for key, value in recalls.items()},
            "mae": float(np.mean(mae)), "spearman": float(np.mean(correlations)),
            "decode_reference_max_abs_error": decode_error,
            "identical_rerank": exact_order, "maximum_ndcg_drop_rerank50": max_ndcg_drop,
            "per_query": per_query}


def select(results):
    primary = lambda method: np.mean([results[str(d)][method]["quality"]["recall"][f"k{k}_R{2*k}"]
                                      for d in (256, 768) for k in (1, 5, 10)])
    mae = lambda method: np.mean([results[str(d)][method]["quality"]["mae"] for d in (256, 768)])
    best = min(("M1", "M2", "M3"), key=lambda method: (-primary(method), mae(method)))
    if primary(best) - primary("M3") <= .005:
        best = "M3"
    no_recall_loss = all(results[str(d)][best]["quality"]["recall"][key] >= value
                         for d in (256, 768) for key, value in results[str(d)]["v1"]["quality"]["recall"].items())
    gain = primary(best) - primary("v1")
    mae_gain = 1 - mae(best) / mae("v1") if mae("v1") > 0 else 0
    contracts = all(results[str(d)][best]["quality"]["identical_rerank"] and
                    results[str(d)][best]["quality"]["maximum_ndcg_drop_rerank50"] <= .005 for d in (256, 768))
    return {"candidate": best, "shortlist_recall_gain": float(gain), "mae_reduction": float(mae_gain),
            "no_recall_loss": no_recall_loss, "quality_contracts": contracts,
            "keep_v1": not (no_recall_loss and contracts and (gain >= .01 or mae_gain >= .10)),
            "merge_eligible": False,
            "remaining_gates": ["independent Rust candidate arithmetic tests against both references",
                                "codec migration and semantic/hybrid engine contracts", "quiet timing slot"]}


def measure(export_path, job_path, output, model_hashes):
    export_bytes, job_bytes = export_path.read_bytes(), job_path.read_bytes()
    export = json.loads(export_bytes)
    if export["job_blake3"] != blake3.blake3(job_bytes).hexdigest():
        raise ValueError("export belongs to another job")
    job = json.loads(job_bytes)
    if [(r["id"], r["split"]) for r in export["images"]] != [(r["id"], r["split"]) for r in job["images"]]:
        raise ValueError("export image order/splits changed")
    if [(r["text"], r["split"]) for r in export["queries"]] != [(r["text"], r["split"]) for r in job["queries"]]:
        raise ValueError("export query order/splits changed")
    results, provenance = {}, {}
    fit = [r for r in export["images"] if r["split"] == "fit"]
    pool = [r for r in export["images"] if r["split"] != "fit"]
    training_queries = [r for r in export["queries"] if r["split"] == "fit"]
    queries = [r for r in export["queries"] if r["split"] == "eval"]
    eval_mask = np.array([r["split"] == "eval" for r in pool])
    label_map = {r["text"]: r["relevant"] for r in job["queries"] if r["split"] == "eval"}
    labels = np.array([[label_map[q["text"]].get(r["id"], 0) for r in pool] for q in queries])
    for dim in (256, 768):
        key = str(dim)
        vectors = lambda rows: np.asarray([r["vectors"][key] for r in rows], dtype=np.float32)
        a, images, text, fit_text = map(vectors, (fit, pool, queries, training_queries))
        for array in (a, images, text, fit_text):
            if array.ndim != 2 or array.shape[1] != dim or not np.isfinite(array).all() or not np.allclose(np.linalg.norm(array, axis=1), 1, atol=1e-5):
                raise ValueError("invalid exported unit vectors")
        anchor, v1 = asset(dim)
        results[key], provenance[key] = {}, {}
        for method, (table, choices) in fit_tables(a, fit_text, anchor, v1).items():
            check_table(table)
            path = output / f"quant4_{dim}.{method}.f32.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(np.asarray(table, dtype="<f4").tobytes())
            expected = None
            if method == "v1":
                expected = {"codes": np.asarray([r["codes"][key] for r in pool], dtype=np.uint8),
                            **{name: [r["scores"][key][name] for r in queries] for name in ("q4", "f32")}}
            quality = evaluate(images, text, labels, eval_mask, anchor, table, expected)
            results[key][method] = {"quality": quality,
                                    "heldout": diagnostics(images[~eval_mask], anchor, table),
                                    "fixture": diagnostics(images[eval_mask], anchor, table)}
            provenance[key][method] = {"table_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                       "choices": choices, "anchor_sha256": hashlib.sha256((ASSETS / f"photo_anchor_{dim}.f32.bin").read_bytes()).hexdigest()}
    evidence = {"schema": 1, "seed": SEED, "identity": export["identity"],
                "model_files_sha256": model_hashes, "job_blake3": export["job_blake3"],
                "export_sha256": hashlib.sha256(export_bytes).hexdigest(),
                "calibration_manifest_sha256": job["calibration_manifest_sha256"],
                "fit_photos": len(fit), "heldout_photos": int((~eval_mask).sum()), "eval_photos": int(eval_mask.sum()),
                "generic_fit_queries": len(training_queries), "eval_queries": len(queries),
                "methods": provenance, "results": results, "decision": select(results),
                "scope": "offline table experiment; no product codec or scan-loop change, no timing claim"}
    write_json(output / "evidence.json", evidence)
    return evidence
