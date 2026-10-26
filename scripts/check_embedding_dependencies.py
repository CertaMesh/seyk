#!/usr/bin/env python3
"""Guard the reusable crates' dependency boundary on every CI platform."""
import re
import subprocess

FORBIDDEN = {"clap", "rusqlite", "walkdir", "globset", "notify", "rmcp", "tokio", "schemars", "url", "pdf-extract", "lopdf"}

def check(package, extra, forbidden, required=frozenset()):
    command = ["cargo", "tree", "--color", "never", "--locked", "-p", package, *extra]
    tree = subprocess.check_output(command, text=True, encoding="utf-8")
    names = set(re.findall(r"\b([a-z][a-z0-9_-]*) v[0-9]", tree))
    if package not in names:
        raise SystemExit(f"cannot parse dependency tree for {package}")
    missing = required - names
    if missing:
        raise SystemExit(f"{package}: missing dependencies: {sorted(missing)}\n{tree}")
    unexpected = names & forbidden
    if unexpected:
        raise SystemExit(f"{package}: forbidden dependencies: {sorted(unexpected)}\n{tree}")
    print(f"{' '.join(command)}: {len(names) - 1} unique dependency packages; boundary clean")

check("seyk-embed", [], FORBIDDEN)
check("seyk-embeddinggemma", ["-e", "normal"], FORBIDDEN)
check("seyk-embeddinggemma", ["-e", "normal", "--features", "image"], FORBIDDEN)
check("seyk-core", ["-e", "normal"], {"seyk-embeddinggemma", "seyk-stt", "ort", "tokenizers", "hf-hub"})

check("seyk-core", ["-e", "normal", "--features", "pdf"], {"seyk-embeddinggemma", "seyk-stt", "ort", "tokenizers", "hf-hub"})

check("seyk", ["-e", "features", "--no-default-features"], {"symphonia", "symphonia-core", "seyk-stt", "ort", "hf-hub"})
check("seyk", ["-e", "normal", "--no-default-features", "--features", "onnx"], {"symphonia", "seyk-stt"})
check("seyk", ["-e", "normal", "--no-default-features", "--features", "cli"], set(), {"symphonia", "seyk-stt"})
check("seyk-embeddinggemma", ["-e", "normal", "--features", "audio"], FORBIDDEN)
check("seyk-core", ["-e", "normal", "--features", "audio"], {"seyk-embeddinggemma", "seyk-stt", "ort", "tokenizers", "hf-hub"})
