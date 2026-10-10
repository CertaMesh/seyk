#!/usr/bin/env python3
"""Repeat the complete audio_golden suite under CPU load using disposable weights.

Build with cargo test -p seyk-embeddinggemma --features audio --test audio_golden
--no-run. --hf-home must point to a disposable copy of the pinned HF snapshot:
the home must be under the system temp directory and all resolved snapshot
files must stay inside that home, with st_nlink == 1. Never hard-link
the disposable weights to the real HF cache. Configured/default HF homes are
rejected even under temp. No downloads or test timeouts.
"""
import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
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
    default_home = (Path.home() / ".cache/huggingface").resolve()
    configured_home = Path(os.environ.get("HF_HOME", str(default_home))).expanduser().resolve()
    if args.hf_home in (default_home, configured_home):
        parser.error("--hf-home must not be the configured or default HF home")
    files = sorted((args.hf_home / "hub").glob("models--*/snapshots/**/*"))
    files = [p for p in files if p.is_file()]
    temporary_root = Path(tempfile.gettempdir()).resolve()
    if (args.hf_home == temporary_root
            or not args.hf_home.is_relative_to(temporary_root)
            or not files
            or any(not p.resolve().is_relative_to(args.hf_home)
                   or p.stat().st_nlink != 1 for p in files)):
        parser.error("provide independent disposable weights inside a system-temp --hf-home (st_nlink == 1)")

    def hashes():
        result = {}
        for path in files:
            with path.open("rb") as stream:
                digest = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
                result[str(path.relative_to(args.hf_home))] = digest.hexdigest()
        return result

    before = hashes()
    args.output.mkdir(parents=True, exist_ok=True)
    stress = []

    def run(index):
        start = time.monotonic()
        try:
            result = subprocess.run(
                [str(args.test_binary), "--include-ignored", "--nocapture"],
                env={**os.environ, "HF_HOME": str(args.hf_home), "HF_HUB_OFFLINE": "1"},
                capture_output=True,
            )
            log = result.stdout + result.stderr
            exit_code = result.returncode
            tests_header = b"running 3 tests\n" in result.stdout
            error = None if tests_header else "expected all three golden tests"
        except OSError as exc:
            log = str(exc).encode()
            exit_code, tests_header, error = None, False, str(exc)
        (args.output / f"run-{index}.log").write_bytes(log)
        report = {"run": index, "exit": exit_code, "tests_header": tests_header,
                  "error": error,
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
        failed = sum(r["exit"] != 0 or r["error"] is not None for r in results)
        ort = sum(r["ort_initialize_exception"] for r in results)
        try:
            unchanged = before == hashes()
            fixture_error = None
        except OSError as exc:
            unchanged, fixture_error = False, str(exc)
        summary = {"total": len(results), "failed": failed,
                   "failure_rate": failed / len(results), "ort_exceptions": ort,
                   "fixture_hashes_unchanged": unchanged, "fixture_error": fixture_error}
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
