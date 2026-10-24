#!/usr/bin/env python3
"""Fetch pinned CC0 photos into ~/.cache/seyk-bench/photo-fixture.

SEYK_PHOTO_FIXTURE overrides the destination; --verify never uses the network."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts/fixtures/photo-manifest.json"
DEST = Path(os.environ.get("SEYK_PHOTO_FIXTURE", Path.home() / ".cache/seyk-bench/photo-fixture")).expanduser()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    rows = json.loads(MANIFEST.read_text())
    ids, hashes = set(), set()
    for row in rows:
        ident, digest = row["id"], row["sha256"]
        if not re.fullmatch(r"[a-z0-9-]+", ident) or ident in ids:
            raise ValueError(f"invalid or duplicate id: {ident}")
        if not re.fullmatch(r"[0-9a-f]{64}", digest) or digest in hashes:
            raise ValueError(f"invalid or duplicate SHA-256: {ident}")
        ids.add(ident)
        hashes.add(digest)
        url = urllib.parse.urlsplit(row["url"])
        page = urllib.parse.urlsplit(row["source_page"])
        if (row["licence"] != "CC0-1.0" or url.scheme != "https"
                or url.hostname not in {"thumb.wikimedia.org", "upload.wikimedia.org"}
                or page.scheme != "https" or page.hostname != "commons.wikimedia.org"):
            raise ValueError(f"unapproved source/licence: {ident}")
    DEST.mkdir(parents=True, exist_ok=True)
    for row in rows:
        path = DEST / (row["id"] + ".jpg")
        if path.exists():
            data = path.read_bytes()
        elif args.verify:
            raise FileNotFoundError(path)
        else:
            request = urllib.request.Request(row["url"], headers={"User-Agent": "SeykPhotoEval/1.0"})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
        if hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError(f"SHA-256 mismatch: {path}; refusing to replace or use it")
        if not path.exists():
            pending = path.with_suffix(".part")
            pending.write_bytes(data)
            pending.replace(path)
    print(f"Verified {len(rows)} CC0 photos")


if __name__ == "__main__":
    main()
