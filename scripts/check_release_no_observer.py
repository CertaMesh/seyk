#!/usr/bin/env python3
"""Check the actual binary about to be packaged, after release smoke tests."""
import pathlib
import re
import shutil
import subprocess
import sys

binary = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "target/release/seyk")
nm = shutil.which("llvm-nm") or shutil.which("nm")
if nm is None:
    raise SystemExit("nm or llvm-nm is required to inspect the release artifact")
if not binary.is_file():
    raise SystemExit(f"missing release artifact: {binary}")
symbols = subprocess.check_output([nm, str(binary)], text=True)
for symbol in symbols.splitlines():
    if re.search(r"observe_lease|observe_indexing_lease|LeaseEvent|lease_observer", symbol):
        raise SystemExit(f"test observer leaked into {binary}: {symbol}")
print(f"{binary}: observer symbols absent (nm)")
