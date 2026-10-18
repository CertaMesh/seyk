#!/usr/bin/env python3
"""Render aggregate retrieval scores as SVG (Python 3.11+, no dependencies)."""
import argparse
import html
import json
import math
import re
import textwrap
import tomllib
from pathlib import Path

DATASETS = {"scifact": ("SciFact", 300), "nfcorpus": ("NFCorpus", 323), "irpapers": ("IRPAPERS text", 180)}
MODES = ("hybrid", "semantic", "lexical")
REPO = Path(__file__).resolve().parents[2]
SETTINGS = ("q8", 256, 200, None)  # precision, dimension, depth, lexical_weight
# Display names for competitor configurations in docs/benchmarks/head-to-head.toml.
TOOL_NAMES = {"qmd": "QMD", "digup": "DigUp", "leann": "LEANN"}
MODE_NAMES = {("qmd", "bm25"): "QMD BM25", ("qmd", "vector"): "QMD vector",
              ("qmd", "hybrid"): "QMD hybrid", ("qmd", "full"): "QMD full (reranker)",
              ("digup", "hybrid"): "DigUp hybrid", ("leann", "contriever"): "LEANN Contriever"}
ROLES = ("headline", "diagnostic-footnote")

# Narrow canvas so the README image stays legible on phones: at ~370 CSS px wide
# (a 400 px viewport minus page padding) the smallest 15px text renders at ~11.5px.
WIDTH = 480
LEFT = 20
BAR_X = 196
BAR_W = 212
SCORE_X = WIDTH - LEFT
ROW = 26
WRAP = 52  # characters per footnote line at 15px

STYLE = ("text{font-family:system-ui,-apple-system,Segoe UI,sans-serif;fill:#17212b;font-size:16px}"
         ".bg{fill:#fff}.muted{fill:#526170}.grid{stroke:#dce2e8}"
         ".hybrid{fill:#007f78}.semantic{fill:#326bc1}.lexical{fill:#b45f06}.measured{fill:#8e3b8a}"
         ".published{fill:#e3e7eb;stroke:#6b7682;stroke-width:1.5;stroke-dasharray:4 3}"
         ".small{font-size:15px}.group{font-size:15px;font-weight:650}.heading{font-size:20px;font-weight:650}"
         "@media(prefers-color-scheme:dark){text{fill:#eef2f6}.bg{fill:#111820}.muted{fill:#bac4ce}.grid{stroke:#35414e}"
         ".hybrid{fill:#42c7b9}.semantic{fill:#79a8ef}.lexical{fill:#f2a65a}.measured{fill:#d68ad0}"
         ".published{fill:#2a333d;stroke:#8f99a5}}")


def _score(value, what):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError(f"{what} must be finite and between 0 and 1")


def validate_result(result, commit):
    """Reject smoke, partial, dirty or differently configured measurements."""
    try:
        ds = result["dataset"]
        if ds not in DATASETS or result.get("smoke") is not False:
            raise ValueError("expected a full retrieval dataset")
        for mode in MODES:
            env = result["environment"][mode]
            if (env["git_sha"] != commit or env["dirty"] is not False
                    or (env["precision"], env["dimension"], env["depth"], env["lexical_weight"]) != SETTINGS):
                raise ValueError("expected clean commit, q8, 256 dims and default ranking/depth")
            if result["completion"][mode]["queries"] != DATASETS[ds][1]:
                raise ValueError("incomplete query set")
            _score(result["metrics"][mode]["nDCG@10"], "nDCG@10")
        first = result["environment"]["hybrid"]
        for mode in MODES[1:]:
            env = result["environment"][mode]
            for key in ("run_id", "profile_id", "manifest_blake3", "threads"):
                if env[key] != first[key]:
                    raise ValueError(f"mismatched mode provenance: {key}")
    except (KeyError, TypeError) as err:
        raise ValueError(f"malformed retrieval aggregate: missing or invalid {err}") from None


def _candidate(result, commit):
    """True when an aggregate claims the requested commit and chart settings.

    Uses only .get so older-schema or unrelated JSON is skipped, never a KeyError.
    """
    if not isinstance(result, dict) or result.get("dataset") not in DATASETS:
        return False
    env = result.get("environment")
    hybrid = env.get("hybrid") if isinstance(env, dict) else None
    if not isinstance(hybrid, dict) or hybrid.get("git_sha") != commit:
        return False
    return tuple(hybrid.get(k, "missing") for k in ("precision", "dimension", "depth", "lexical_weight")) == SETTINGS


def latest_results(root, commit):
    """One aggregate per dataset for the commit and chart settings.

    Selection never depends on file timestamps. Several matching files are
    allowed only when their charted nDCG@10 values agree; then the first by
    relative path is used. Disagreeing duplicates are an error.
    """
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise ValueError("--commit must be a full 40-character Git SHA")
    candidates = {}
    for path in sorted(root.glob("*/*.json"), key=lambda p: p.relative_to(root).as_posix()):
        try:
            result = json.loads(path.read_text())
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        if not _candidate(result, commit):
            continue
        validate_result(result, commit)
        candidates.setdefault(result["dataset"], []).append((path, result))
    selected = {}
    for ds, found in candidates.items():
        scores = {tuple(r["metrics"][m]["nDCG@10"] for m in MODES) for _, r in found}
        if len(scores) > 1:
            names = ", ".join(p.relative_to(root).as_posix() for p, _ in found)
            raise ValueError(f"ambiguous {ds} aggregates with different scores: {names}; use a dedicated --results directory")
        selected[ds] = found[0][1]
    if set(selected) != set(DATASETS):
        raise ValueError("missing full results for: " + ", ".join(sorted(set(DATASETS) - set(selected))))
    return selected


def load_comparisons(path):
    data = tomllib.loads(path.read_text())
    rows = data["comparison"]
    seen = set()
    for row in rows:
        key = (row["dataset"], row["system"])
        if key in seen or key[0] not in DATASETS:
            raise ValueError("duplicate comparison or unknown dataset")
        seen.add(key)
        _score(row["ndcg_at_10"], "published nDCG@10")
        for field in ("source_url", "source_title", "protocol", "retrieved"):
            if not row.get(field):
                raise ValueError(f"missing comparison provenance: {field}")
        if not row["source_url"].startswith("https://"):
            raise ValueError("comparison source must be an HTTPS URL")
    return rows


def validate_head_to_head(runs):
    """Completed competitor runs measured on Seyk's harness, one per tool/mode/dataset."""
    try:
        seen = set()
        for run in runs:
            key = (run["dataset"], run["tool"], run["mode"])
            if key[0] not in DATASETS or key[1:] not in MODE_NAMES or key in seen:
                raise ValueError(f"duplicate or unknown head-to-head run: {key}")
            seen.add(key)
            if run["comparison_role"] not in ROLES:
                raise ValueError(f"unknown comparison_role: {run['comparison_role']}")
            if run["queries"] != DATASETS[key[0]][1]:
                raise ValueError("incomplete head-to-head query set")
            _score(run["ndcg_at_10"], "head-to-head nDCG@10")
            if not re.fullmatch(r"[0-9a-f]{40}", run["git_sha"]) or not run.get("version"):
                raise ValueError("head-to-head run needs a full adapter commit and tool version")
            empty = run["queries_without_results"]
            if isinstance(empty, bool) or not isinstance(empty, int) or not 0 <= empty <= run["queries"]:
                raise ValueError("invalid queries_without_results")
        if len({run["git_sha"] for run in runs}) > 1:
            raise ValueError("head-to-head runs must share one adapter commit")
    except (KeyError, TypeError) as err:
        raise ValueError(f"malformed head-to-head run: missing or invalid {err}") from None
    return runs


def load_head_to_head(path):
    data = tomllib.loads(path.read_text())
    if data.get("schema") != 1:
        raise ValueError("unsupported head-to-head schema")
    return validate_head_to_head(data["runs"])


def _version(run):
    v = run["version"]
    return v[:7] if re.fullmatch(r"[0-9a-f]{40}", v) else v


def render(results, comparisons, commit, version, machine, head_to_head=()):
    for result in results.values():
        validate_result(result, commit)
    validate_head_to_head(list(head_to_head))
    out = []
    y = 0  # top of the next free line

    def text(x, y, value, cls="", extra=""):
        attrs = f' class="{cls}"' if cls else ""
        out.append(f'<text x="{x}" y="{y}"{attrs}{" " + extra if extra else ""}>{html.escape(str(value))}</text>')

    def line(value, cls="small muted", step=20, baseline=15):
        nonlocal y
        text(LEFT, y + baseline, value, cls)
        y += step

    def note(value, cls="small muted"):
        for part in textwrap.wrap(value, WRAP):
            line(part, cls)

    def bar_row(label, score, cls, source=None):
        nonlocal y
        if source:
            out.append(f'<a href="{html.escape(source["source_url"], quote=True)}"><title>{html.escape(source["source_title"] + "; " + source["protocol"])}</title>')
        text(LEFT, y + 14, label, "muted" if source else "")
        out.append(f'<rect x="{BAR_X}" y="{y}" width="{score * BAR_W:.3f}" height="18" rx="3" class="{cls}"/>')
        text(SCORE_X, y + 14, f"{score:.3f}", "", 'text-anchor="end"')
        if source:
            out.append('</a>')
        y += ROW

    competitors = list(head_to_head)
    y = 16
    line("Retrieval quality", "heading", 28, 20)
    line("nDCG@10 · higher is better · 0–1 scale", "muted", 24)
    note(f"Seyk {version} · {commit[:12]} · q8 · 256 dims · CPU · segmented documents, ranked per resource", "small")
    note(machine)
    for ds, (name, _) in DATASETS.items():
        y += 24
        line(name, "heading", 30, 20)
        top, start = y, len(out)
        line("Seyk", "group", 24)
        for m in MODES:
            bar_row(f"Seyk {m}" + (" (default)" if m == "hybrid" else ""), results[ds]["metrics"][m]["nDCG@10"], m)
        measured = sorted((r for r in competitors if r["dataset"] == ds and r["comparison_role"] == "headline"),
                          key=lambda r: (-r["ndcg_at_10"], MODE_NAMES[r["tool"], r["mode"]]))
        diagnostic = sorted((r for r in competitors if r["dataset"] == ds and r["comparison_role"] != "headline"),
                            key=lambda r: MODE_NAMES[r["tool"], r["mode"]])
        if measured or diagnostic:
            y += 6
            line("Other local tools · measured on Seyk's harness", "group", 24)
            for r in measured:
                bar_row(MODE_NAMES[r["tool"], r["mode"]], r["ndcg_at_10"], "measured")
            for r in diagnostic:
                note(f"† {MODE_NAMES[r['tool'], r['mode']]} {r['ndcg_at_10']:.3f} (diagnostic):"
                     f" {r['queries_without_results']}/{r['queries']} queries empty")
        published = sorted((r for r in comparisons if r["dataset"] == ds), key=lambda r: (-r["ndcg_at_10"], r["system"]))
        y += 6
        if published:
            line("Published · different setup, not rerun", "group", 24)
            for r in published:
                bar_row(r["system"], r["ndcg_at_10"], "published", r)
        else:
            note("No sourced published nDCG@10 for this dataset.")
        # Gridlines go behind this section's bars, so insert them right after its heading.
        out[start:start] = [f'<line x1="{BAR_X + tick * BAR_W / 4:g}" y1="{top}" x2="{BAR_X + tick * BAR_W / 4:g}"'
                            f' y2="{y}" class="grid"/>' for tick in range(5)]
        for tick in range(5):
            text(f"{BAR_X + tick * BAR_W / 4:g}", y + 15, f"{tick / 4:g}", "small muted", 'text-anchor="middle"')
        y += 20
    y += 20
    if competitors:
        tools = sorted({(TOOL_NAMES[r["tool"]], _version(r)) for r in competitors})
        note("Other tools: " + ", ".join(f"{t} {v}" for t, v in tools) + f"; adapter {competitors[0]['git_sha'][:7]}."
             " Same corpora, queries and scorer; each tool uses its own models and chunking. One pass each.")
        if any(r["comparison_role"] != "headline" for r in competitors):
            note("† Not charted: pinned QMD BM25 joins query terms with AND, so most queries return nothing."
                 if any(r["tool"] == "qmd" and r["mode"] == "bm25" for r in competitors)
                 else "† Diagnostic runs are not charted.")
        y += 6
    if not any("embeddinggemma" in row["system"].casefold() for row in comparisons):
        note("EmbeddingGemma: no published SciFact/NFCorpus score.")
    note("Dashed bars are published results from different setups (whole documents, no chunking):"
         " context, not a controlled comparison. Details: docs/benchmarks.md.")
    height = y + 10
    head = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" viewBox="0 0 {WIDTH} {height}" role="img" aria-labelledby="title desc">',
            '<title id="title">Seyk retrieval benchmarks: nDCG@10</title>',
            '<desc id="desc">Full dataset results. Seyk uses segmented documents. Purple bars are other local tools measured on'
            " Seyk's harness with their own models and chunking. Dashed bars are published results from different setups,"
            ' not head-to-head measurements.</desc>',
            f'<style>{STYLE}</style>',
            f'<rect class="bg" width="{WIDTH}" height="{height}" rx="12"/>']
    return "\n".join(head + out + ['</svg>']) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path.home() / ".cache/seyk-bench/results")
    parser.add_argument("--comparisons", type=Path, default=REPO / "docs/benchmarks/comparisons.toml")
    parser.add_argument("--head-to-head", type=Path, default=REPO / "docs/benchmarks/head-to-head.toml")
    parser.add_argument("--output", type=Path, default=REPO / "docs/assets/benchmarks.svg")
    parser.add_argument("--commit", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--machine", required=True, help="Recorded benchmark machine, not the rendering host")
    args = parser.parse_args()
    svg = render(latest_results(args.results, args.commit), load_comparisons(args.comparisons), args.commit,
                 args.version, args.machine, load_head_to_head(args.head_to_head))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg)


if __name__ == "__main__":
    main()
