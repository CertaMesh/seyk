#!/usr/bin/env python3
"""Fetch pinned CC0 photos into ~/.cache/seyk-bench/photo-fixture.

SEYK_PHOTO_FIXTURE overrides the destination; --verify never uses the network.
--repin accepts validated JPEG changes from the same pinned URL and records provenance."""
import argparse
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import NamedTuple
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "scripts/fixtures/photo-manifest.json"
DEST = Path(os.environ.get("SEYK_PHOTO_FIXTURE", Path.home() / ".cache/seyk-bench/photo-fixture")).expanduser()


class Download(NamedTuple):
    data: bytes
    status: int
    content_type: str
    url: str


class PinnedUrlOnly(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, msg, headers, newurl):
        raise ValueError(f"redirect from pinned URL refused: {request.full_url} -> {newurl}")


OPENER = urllib.request.build_opener(PinnedUrlOnly())


def jpeg_width(data):
    """Decode all pixels with an already-installed decoder; never install one."""
    if not data.startswith(b"\xff\xd8"):
        raise ValueError("re-pin requires JPEG bytes")
    try:
        from PIL import Image
    except ImportError:
        if not shutil.which("sips"):
            raise RuntimeError("re-pin requires an already-installed JPEG decoder (Pillow or macOS sips)")
        with tempfile.TemporaryDirectory(prefix="seyk-jpeg-check-") as directory:
            source = Path(directory) / "input.jpg"
            output = Path(directory) / "decoded.png"
            source.write_bytes(data)
            info = subprocess.run(["sips", "-g", "format", "-g", "pixelWidth", str(source)],
                                  check=True, capture_output=True, text=True).stdout
            if not re.search(r"format: jpeg\s", info):
                raise ValueError("re-pin requires a JPEG image")
            width = re.search(r"pixelWidth: (\d+)", info)
            if not width:
                raise ValueError("cannot read JPEG pixel width")
            subprocess.run(["sips", "-s", "format", "png", str(source), "--out", str(output)],
                           check=True, capture_output=True)
            if not output.is_file():
                raise ValueError("JPEG did not decode")
            return int(width.group(1))
    import io
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG":
            raise ValueError("re-pin requires a JPEG image")
        image.load()
        return image.width


def validate_repin(row, download, original):
    unchanged = tuple(row[key] for key in ("url", "source_page", "licence", "licence_url"))
    if unchanged != original or download.url != row["url"]:
        raise ValueError("re-pin source URL/page/licence changed")
    if download.status != 200 or download.content_type.lower() != "image/jpeg":
        raise ValueError("re-pin requires HTTP 200 and content-type image/jpeg")
    width = jpeg_width(download.data)
    size = re.search(r"/(\d+)px-", urllib.parse.urlsplit(row["url"]).path)
    if size and width != int(size.group(1)):
        raise ValueError(f"re-pin JPEG width {width} differs from pinned URL size {size.group(1)}")


def retry_after(value):
    """Return Retry-After seconds, accepting both HTTP forms."""
    if value is not None:
        try:
            seconds = float(value)
            return max(0.0, seconds) if math.isfinite(seconds) else 30.0
        except ValueError:
            try:
                date = parsedate_to_datetime(value)
                return max(0.0, (date - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                pass
    return 30.0


def fetch(url):
    request = urllib.request.Request(
        url, headers={"User-Agent": "SeykPhotoEval/1.0 (https://github.com/CertaMesh/seyk)"})
    wait = 0.0
    for attempt in range(4):
        time.sleep(1)
        try:
            with OPENER.open(request, timeout=60) as response:
                return Download(response.read(), response.status,
                                response.headers.get_content_type(), response.geturl())
        except urllib.error.HTTPError as error:
            if error.code not in (429, 503):
                raise
            if attempt == 3:
                raise RuntimeError(
                    f"HTTP {error.code} after 3 retries for {url}; stopping without substitutes"
                ) from error
            wait = max(30.0, retry_after(error.headers.get("Retry-After")), wait * 2)
            print(f"HTTP {error.code}; retry {attempt + 1}/3 in {wait:g}s: {url}", flush=True)
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--repin", action="store_true", help="validate drifted JPEGs and update their SHA-256 pins")
    args = parser.parse_args()
    manifest_bytes = MANIFEST.read_bytes()
    rows = json.loads(manifest_bytes)
    originals = {row["id"]: tuple(row[key] for key in ("url", "source_page", "licence", "licence_url")) for row in rows}
    repinned = 0
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
        download = None
        if path.exists():
            data = path.read_bytes()
        elif args.verify:
            raise FileNotFoundError(path)
        else:
            download = fetch(row["url"])
            data = download.data
        digest = hashlib.sha256(data).hexdigest()
        changed = digest != row["sha256"]
        if changed:
            if not args.repin or download is None:
                raise ValueError(f"SHA-256 mismatch: {path}; refusing to replace or use it")
            validate_repin(row, download, originals[row["id"]])
            if digest in hashes:
                raise ValueError(f"re-pin duplicates an existing photo: {row['id']}")
            if MANIFEST.read_bytes() != manifest_bytes:
                raise ValueError("manifest changed during fetch; stopping")
            previous = row["sha256"]
            row["sha256"] = digest
            row["repinned"] = {"date": datetime.now(timezone.utc).date().isoformat(), "previous_sha256": previous}
            hashes.remove(previous)
            hashes.add(digest)
        if not path.exists():
            pending = path.with_suffix(".part")
            pending.write_bytes(data)
            pending.replace(path)
        if changed:
            manifest_bytes = (json.dumps(rows, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
            pending_manifest = MANIFEST.with_suffix(".json.part")
            pending_manifest.write_bytes(manifest_bytes)
            pending_manifest.replace(MANIFEST)
            repinned += 1
            print(f"Re-pinned {row['id']}: {previous} -> {digest}", flush=True)
    print(f"Verified {len(rows)} CC0 photos; re-pinned {repinned}")


if __name__ == "__main__":
    main()
