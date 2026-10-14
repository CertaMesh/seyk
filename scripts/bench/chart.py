#!/usr/bin/env python3
"""Render aggregate retrieval scores as SVG (Python 3.11+, no dependencies)."""
import argparse
import html
import json
import math
import re
import tomllib
from pathlib import Path

DATASETS = {"scifact": ("SciFact", 300), "nfcorpus": ("NFCorpus", 323), "irpapers": ("IRPAPERS text", 180)}
MODES = ("hybrid", "semantic", "lexical")
REPO = Path(__file__).resolve().parents[2]


def validate_result(result, commit):
    """Reject smoke, partial, dirty or differently configured measurements."""
    ds = result["dataset"]
    if ds not in DATASETS or result.get("smoke") is not False:
        raise ValueError("expected a full retrieval dataset")
    for mode in MODES:
        env = result["environment"][mode]
        if (env["git_sha"] != commit or env["dirty"] is not False
                or env["precision"] != "q8" or env["dimension"] != 256
                or env["depth"] != 200 or env["lexical_weight"] is not None):
            raise ValueError("expected clean commit, q8, 256 dims and default ranking/depth")
        if result["completion"][mode]["queries"] != DATASETS[ds][1]:
            raise ValueError("incomplete query set")
        score = result["metrics"][mode]["nDCG@10"]
        if isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("nDCG@10 must be finite and between 0 and 1")
    first = result["environment"]["hybrid"]
    for mode in MODES[1:]:
        env = result["environment"][mode]
        for key in ("run_id", "profile_id", "manifest_blake3", "threads"):
            if env[key] != first[key]:
                raise ValueError(f"mismatched mode provenance: {key}")


def latest_results(root, commit):
    """Newest aggregate per dataset by file mtime; filename breaks ties."""
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("--commit must be a full 40-character Git SHA")
    selected = {}
    for path in sorted(root.glob("*/*.json"), key=lambda p: (p.stat().st_mtime_ns, str(p))):
        result = json.loads(path.read_text())
        if result.get("dataset") not in DATASETS:
            continue
        if result.get("environment", {}).get("hybrid", {}).get("git_sha") != commit:
            continue
        env = result["environment"]["hybrid"]
        if (env["precision"], env["dimension"], env["depth"], env["lexical_weight"]) != ("q8", 256, 200, None):
            continue
        validate_result(result, commit)
        selected[result["dataset"]] = result
    if set(selected) != set(DATASETS):
        raise ValueError("missing full results for: " + ", ".join(sorted(set(DATASETS) - set(selected))))
    return selected


def load_comparisons(path):
    data = tomllib.loads(path.read_text())
    rows = data["comparison"]
    seen = set()
    for row in rows:
        key = (row["dataset"], row["system"])
        score = row["ndcg_at_10"]
        if key in seen or key[0] not in DATASETS:
            raise ValueError("duplicate comparison or unknown dataset")
        seen.add(key)
        if isinstance(score, bool) or not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("invalid published nDCG@10")
        for field in ("source_url", "source_title", "protocol", "retrieved"):
            if not row.get(field):
                raise ValueError(f"missing comparison provenance: {field}")
        if not row["source_url"].startswith("https://"):
            raise ValueError("comparison source must be an HTTPS URL")
    return rows


def render(results, comparisons, commit, version, machine):
    for result in results.values():
        validate_result(result, commit)
    height = 222 + sum(130 + 34 * (3 + sum(r["dataset"] == ds for r in comparisons)) for ds in DATASETS)
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" viewBox="0 0 960 {height}" role="img" aria-labelledby="title desc">',
           '<title id="title">Seyk retrieval benchmarks: nDCG@10</title>',
           '<desc id="desc">Full dataset results. Seyk uses segmented documents. Muted bars are published results from different setups, not head-to-head measurements.</desc>',
           '<style>text{font-family:system-ui,-apple-system,Segoe UI,sans-serif;fill:#17212b;font-size:15px}.bg{fill:#fff}.muted{fill:#526170}.grid{stroke:#dce2e8}.hybrid{fill:#007f78}.semantic{fill:#326bc1}.lexical{fill:#91a9a7}.published{fill:#a0a7b0}.small{font-size:13px}.heading{font-size:20px;font-weight:650}@media(prefers-color-scheme:dark){text{fill:#eef2f6}.bg{fill:#111820}.muted{fill:#bac4ce}.grid{stroke:#35414e}.hybrid{fill:#42c7b9}.semantic{fill:#79a8ef}.lexical{fill:#849e9b}.published{fill:#79838f}}</style>',
           f'<rect class="bg" width="960" height="{height}" rx="12"/>']

    def text(x, y, value, cls="", extra=""):
        out.append(f'<text x="{x}" y="{y}" class="{cls}" {extra}>{html.escape(str(value))}</text>')

    text(28, 38, "Retrieval quality", "heading")
    text(28, 64, "nDCG@10 · higher is better · 0–1 scale", "muted")
    text(28, 91, f"Seyk {version} · {commit[:12]} · q8 · 256 dims · CPU", "small")
    text(28, 113, machine, "small muted")
    y = 155
    for ds, (name, _) in DATASETS.items():
        published = sorted((r for r in comparisons if r["dataset"] == ds), key=lambda r: (-r["ndcg_at_10"], r["system"]))
        text(28, y, name, "heading")
        text(28, y + 23, "Seyk: segmented documents → resource ranking", "small muted")
        y += 50
        rows = [(f"Seyk {m}" + (" (default)" if m == "hybrid" else ""), results[ds]["metrics"][m]["nDCG@10"], m, None) for m in MODES]
        rows += [(r["system"], r["ndcg_at_10"], "published", r) for r in published]
        for label, score, cls, source in rows:
            if source:
                out.append(f'<a href="{html.escape(source["source_url"], quote=True)}"><title>{html.escape(source["source_title"] + "; " + source["protocol"])}</title>')
            text(28, y + 16, label, "muted" if source else "")
            out.append(f'<rect x="302" y="{y}" width="{score * 560:.3f}" height="23" rx="3" class="{cls}"/>')
            text(916, y + 17, f"{score:.3f}", "", 'text-anchor="end"')
            if source:
                out.append('</a>')
            y += 34
        for tick in range(6):
            x = 302 + tick * 112
            text(x, y + 7, f"{tick / 5:.1f}", "small muted", 'text-anchor="middle"')
        text(28, y + 31, "Published · different setup: whole documents, no chunking" if published else "No sourced published nDCG@10 comparison available", "small muted")
        y += 80
    if not any("embeddinggemma" in row["system"].casefold() for row in comparisons):
        text(28, height - 50, "EmbeddingGemma: no published SciFact/NFCorpus score found.", "small muted")
    text(28, height - 26, "Sources and settings: docs/benchmarks.md · Published bars are context, not a controlled comparison.", "small muted")
    out.append('</svg>')
    return "\n".join(out) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path.home() / ".cache/seyk-bench/results")
    parser.add_argument("--comparisons", type=Path, default=REPO / "docs/benchmarks/comparisons.toml")
    parser.add_argument("--output", type=Path, default=REPO / "docs/assets/benchmarks.svg")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--machine", required=True, help="Recorded benchmark machine, not the rendering host")
    args = parser.parse_args()
    svg = render(latest_results(args.results, args.commit), load_comparisons(args.comparisons), args.commit, args.version, args.machine)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)


if __name__ == "__main__":
    main()
