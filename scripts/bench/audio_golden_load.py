#!/usr/bin/env python3
"""Repeat the complete audio_golden suite under CPU load using disposable weights.

Build with cargo test -p seyk-embeddinggemma --features audio --test audio_golden
--no-run. HF_HOME must point to a disposable copy of the pinned HF snapshot:
all resolved snapshot files must stay inside that directory. Never hard-link
the disposable weights to the real HF cache. No downloads or test timeouts.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_binary", type=lambda p: Path(p).resolve())
    parser.add_argument("--hf-home", type=lambda p: Path(p).resolve(), required=True)
    parser.add_argument("--runs", type=int, default=12)
    parser.add_argument("--parallel", type=int, default=3)
    parser.add_argument("--stress-workers", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.runs, args.parallel) < 1 or args.stress_workers < 0:
        parser.error("counts must be positive; stress workers may be zero")
    files = sorted((args.hf_home / "hub").glob("models--*/snapshots/**/*"))
    files = [p for p in files if p.is_file()]
    if not files or any(not p.resolve().is_relative_to(args.hf_home) for p in files):
        parser.error("provide disposable weights entirely inside --hf-home")

    def hashes():
        result = {}
        for path in files:
            with path.open("rb") as stream:
                result[str(path.relative_to(args.hf_home))] = hashlib.file_digest(
                    stream, "sha256"
                ).hexdigest()
        return result

    before = hashes()
    args.output.mkdir(parents=True, exist_ok=True)
    stress = []

    def run(index):
        start = time.monotonic()
        result = subprocess.run(
            [str(args.test_binary), "--include-ignored", "--nocapture"],
            env={**os.environ, "HF_HOME": str(args.hf_home), "HF_HUB_OFFLINE": "1"},
            capture_output=True,
        )
        log = result.stdout + result.stderr
        (args.output / f"run-{index}.log").write_bytes(log)
        if b"running 3 tests\n" not in result.stdout:
            raise RuntimeError(f"run {index}: expected all three golden tests")
        report = {"run": index, "exit": result.returncode,
                  "ort_initialize_exception": b"unknown exception in Initialize()" in log,
                  "seconds": round(time.monotonic() - start, 3)}
        print(json.dumps(report), flush=True)
        return report

    try:
        for _ in range(args.stress_workers):
            stress.append(subprocess.Popen([
                sys.executable, "-c",
                'import hashlib; b=b"x"*1048576\nwhile True: hashlib.sha256(b).digest()',
            ]))
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.parallel) as pool:
            results = list(pool.map(run, range(args.runs)))
        failed = sum(r["exit"] != 0 for r in results)
        ort = sum(r["ort_initialize_exception"] for r in results)
        unchanged = before == hashes()
        summary = {"total": len(results), "failed": failed,
                   "failure_rate": failed / len(results), "ort_exceptions": ort,
                   "fixture_hashes_unchanged": unchanged}
        (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        print(json.dumps(summary), flush=True)
        return int(failed != 0 or not unchanged)
    finally:
        for process in stress:
            process.terminate()
        for process in stress:
            process.wait()


if __name__ == "__main__":
    sys.exit(main())
