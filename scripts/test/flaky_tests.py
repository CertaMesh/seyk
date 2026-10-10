#!/usr/bin/env python3
"""Run a targeted libtest repeatedly under CPU load, optionally with cold macOS CLI launches.

Build the test with cargo test --no-run, then pass the emitted executable path.
Use --binary target/debug/seyk --cold-cli to reproduce first-launch assessment.
No weights are downloaded or changed. JSON lines report each run and the rate.
"""

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("test_binary", type=lambda p: Path(p).resolve())
    parser.add_argument("test_name")
    parser.add_argument("--binary", type=lambda p: Path(p).resolve())
    parser.add_argument("--cold-cli", action="store_true")
    parser.add_argument("--runs", type=int, default=24)
    parser.add_argument("--parallel", type=int, default=8)
    parser.add_argument("--stress-workers", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.cold_cli and (sys.platform != "darwin" or args.binary is None):
        parser.error("--cold-cli requires macOS and --binary")
    if min(args.runs, args.parallel) < 1 or args.stress_workers < 0:
        parser.error("counts must be positive; stress workers may be zero")
    args.output.mkdir(parents=True, exist_ok=True)
    stress = []

    def run(index):
        with tempfile.TemporaryDirectory() as directory:
            env = {**os.environ, "HF_HUB_OFFLINE": "1"}
            if args.binary:
                binary = args.binary
                if args.cold_cli:
                    binary = Path(directory) / "seyk"
                    shutil.copy2(args.binary, binary)
                    # New identity models fresh code after a rebuild, instead
                    # of repeatedly exercising the OS's cached assessment.
                    subprocess.run(
                        ["codesign", "--force", "--sign", "-", "--identifier",
                         f"seyk-flaky-{args.output.name}-{index}", str(binary)],
                        check=True, capture_output=True,
                    )
                env["CARGO_BIN_EXE_seyk"] = str(binary)
            start = time.monotonic()
            result = subprocess.run(
                [str(args.test_binary), args.test_name, "--exact", "--include-ignored", "--nocapture"],
                env=env, capture_output=True,
            )
            (args.output / f"run-{index}.log").write_bytes(result.stdout + result.stderr)
            if b"running 1 test\n" not in result.stdout:
                raise RuntimeError(f"run {index}: expected exactly one matching test; see run log")
            report = {"run": index, "exit": result.returncode,
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
        print(json.dumps({"total": len(results), "failed": failed,
                          "failure_rate": failed / len(results)}), flush=True)
        return int(failed != 0)
    finally:
        for process in stress:
            process.terminate()
        for process in stress:
            process.wait()


if __name__ == "__main__":
    sys.exit(main())
