#!/usr/bin/env python3
"""Build a separate, pinned CC0 calibration set. Default mode estimates only."""
import argparse
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import io
import json
import math
from pathlib import Path
import re
import sys
import urllib.parse

import blake3
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fetch_photo_fixture as fetcher

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "scripts/fixtures/photo-manifest.json"
LICENCE = "https://creativecommons.org/publicdomain/zero/1.0/"
SEED = "seyk-photo-calibration-2026-v1"
MAX_IMAGE_BYTES = 2_000_000
DEFAULT_DOWNLOAD_BYTES = 4_000_000_000
TOPICS = ("animals birds", "plants flowers", "forests mountains", "rivers lakes", "ocean coast",
          "cities streets", "architecture buildings", "interiors furniture", "food cooking",
          "people activities", "sports recreation", "vehicles transport", "tools equipment",
          "art sculpture", "weather sky", "agriculture countryside", "insects wildlife",
          "rocks geology", "boats harbours", "everyday objects")


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(path.suffix + ".part")
    pending.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    pending.replace(path)


def title_key(title):
    return urllib.parse.unquote(title).replace("_", " ").removeprefix("File:").casefold()


def safe_url(url, hosts):
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme != "https" or parsed.hostname not in hosts or parsed.username or parsed.password:
        raise ValueError(f"unapproved source URL: {url}")
    return url


def cc0_url(url):
    parsed = urllib.parse.urlsplit(url)
    return (parsed.scheme in {"http", "https"} and parsed.hostname == "creativecommons.org"
            and parsed.path.rstrip("/") in {"/publicdomain/zero/1.0", "/publicdomain/zero/1.0/deed.en"}
            and not parsed.query and not parsed.fragment and not parsed.username and not parsed.password)


def image_row(page, excluded_titles):
    """Search matches are untrusted; require the licence URL AND licence name."""
    info = page["imageinfo"][0]
    meta = info["extmetadata"]
    licence = meta.get("LicenseShortName", {}).get("value")
    licence_url = meta.get("LicenseUrl", {}).get("value", "")
    if licence != "CC0" or not cc0_url(licence_url):
        raise ValueError("image does not have explicit CC0-1.0 metadata")
    if info.get("mime") != "image/jpeg" or title_key(page["title"]) in excluded_titles:
        raise ValueError("not a separate JPEG photo")
    if info.get("thumbwidth") != 960 or info.get("thumbmime", "image/jpeg") != "image/jpeg":
        raise ValueError("thumbnail is not a 960px JPEG")
    return {"id": f"commons-{page['pageid']}", "title": page["title"],
            "source_page": safe_url(info["descriptionurl"], {"commons.wikimedia.org"}),
            "url": safe_url(info["thumburl"], {"upload.wikimedia.org", "thumb.wikimedia.org"}),
            "licence": "CC0-1.0", "licence_url": LICENCE,
            "original_sha1": info["sha1"], "source_timestamp": info["timestamp"],
            "licence_metadata": meta, "max_bytes": MAX_IMAGE_BYTES}


def split_rows(rows, fit_count):
    """Assign an exact-sized split before any vectors or outcomes are observed."""
    rows = sorted(rows, key=lambda row: blake3.blake3(f"{SEED}:{row['id']}".encode()).digest())
    return [dict(row, split="fit" if i < fit_count else "heldout") for i, row in enumerate(rows)]


def request_json(params):
    url = "https://commons.wikimedia.org/w/api.php?" + urllib.parse.urlencode(
        dict(action="query", format="json", formatversion=2, **params))
    result = json.loads(fetcher.fetch(url).data)
    if "error" in result:
        raise RuntimeError(result["error"])
    return result


def discover(count, excluded):
    """Round-robin broad topic searches; licence is checked for every result."""
    titles = {title_key(row["title"]) for row in excluded}
    ids = {row["id"] for row in excluded}
    rows, seen_hashes = {}, set()
    states = [{} for _ in TOPICS]
    while len(rows) < count and any(state is not None for state in states):
        for i, topic in enumerate(TOPICS):
            if states[i] is None:
                continue
            response = request_json(dict(generator="search", gsrsearch=f'{topic} filetype:bitmap incategory:"CC-zero"',
                                         gsrnamespace=6, gsrlimit=10, prop="imageinfo",
                                         iiprop="url|size|mime|thumbmime|sha1|timestamp|extmetadata",
                                         iiurlwidth=960, iiextmetadatalanguage="en", **states[i]))
            states[i] = response.get("continue")
            for page in response.get("query", {}).get("pages", []):
                try:
                    row = image_row(page, titles)
                except (KeyError, ValueError):
                    continue
                if row["id"] in ids or row["id"] in rows or row["original_sha1"] in seen_hashes:
                    continue
                row["topic"] = topic
                rows[row["id"]] = row
                seen_hashes.add(row["original_sha1"])
                if len(rows) == count:
                    return list(rows.values())
    raise RuntimeError(f"only {len(rows)} eligible photos found; refusing a smaller set")


def verify_source(row):
    response = request_json(dict(pageids=row["id"].removeprefix("commons-"), prop="imageinfo",
                                 iiprop="url|size|mime|thumbmime|sha1|timestamp|extmetadata",
                                 iiurlwidth=960, iiextmetadatalanguage="en"))
    pages = response.get("query", {}).get("pages", [])
    if len(pages) != 1:
        raise ValueError("selected source disappeared")
    current = image_row(pages[0], set())
    for field in ("id", "title", "url", "source_page", "original_sha1", "source_timestamp"):
        if current[field] != row[field]:
            raise ValueError(f"selected source changed: {field}")
    return current["licence_metadata"]


def download(row):
    """Bound bytes read even if Content-Length is absent or incorrect."""
    request = fetcher.urllib.request.Request(row["url"], headers={
        "User-Agent": "SeykPhotoCalibration/1.0 (https://github.com/CertaMesh/seyk)"})
    wait = 0.0
    for attempt in range(4):
        fetcher.time.sleep(1)
        try:
            # The pinned-URL opener refuses redirects on every attempt.
            with fetcher.OPENER.open(request, timeout=60) as response:
                if response.status != 200 or response.headers.get_content_type() != "image/jpeg":
                    raise ValueError("expected HTTP 200 image/jpeg")
                data = response.read(MAX_IMAGE_BYTES + 1)
            if len(data) > MAX_IMAGE_BYTES:
                raise ValueError("photo exceeds the 2 MB limit; do not substitute it")
            validate_jpeg(data)
            return data
        except fetcher.urllib.error.HTTPError as error:
            if error.code not in (429, 503):
                raise
            if attempt == 3:
                raise RuntimeError(f"HTTP {error.code} after 3 retries; stopping without substitutes") from error
            wait = max(30.0, fetcher.retry_after(error.headers.get("Retry-After")), wait * 2)
            print(f"HTTP {error.code}; retry {attempt + 1}/3 in {wait:g}s: {row['url']}", flush=True)
            fetcher.time.sleep(wait)


def validate_jpeg(data):
    with Image.open(io.BytesIO(data)) as image:
        if image.format != "JPEG" or image.width != 960:
            raise ValueError("expected a decodable 960px JPEG")
        image.load()


def verify(directory, excluded=None):
    excluded = json.loads(EVAL.read_text()) if excluded is None else excluded
    manifest = json.loads((directory / "manifest.json").read_text())
    rows = manifest["photos"]
    if manifest["seed"] != SEED or len(rows) != manifest["count"]:
        raise ValueError("invalid calibration manifest")
    if split_rows([{k: v for k, v in row.items() if k != "split"} for row in rows], manifest["fit_count"]) != rows:
        raise ValueError("calibration split changed")
    titles = {title_key(row["title"]) for row in excluded}
    ids = {row["id"] for row in excluded}
    hashes = {row["sha256"] for row in excluded}
    originals = set()
    for row in rows:
        if not re.fullmatch(r"commons-[0-9]+", row["id"]) or row["id"] in ids or title_key(row["title"]) in titles:
            raise ValueError("duplicate or overlapping photo")
        if row["licence"] != "CC0-1.0" or row["licence_url"] != LICENCE:
            raise ValueError("unapproved licence")
        meta = row["licence_metadata"]
        if meta["LicenseShortName"]["value"] != "CC0" or not cc0_url(meta["LicenseUrl"]["value"]):
            raise ValueError("missing per-image licence evidence")
        safe_url(row["url"], {"upload.wikimedia.org", "thumb.wikimedia.org"})
        safe_url(row["source_page"], {"commons.wikimedia.org"})
        data = (directory / "images" / f"{row['id']}.jpg").read_bytes()
        if len(data) > MAX_IMAGE_BYTES or hashlib.sha256(data).hexdigest() != row["sha256"]:
            raise ValueError("photo hash/size mismatch")
        if row["sha256"] in hashes or row["original_sha1"] in originals:
            raise ValueError("duplicate or overlapping photo bytes")
        validate_jpeg(data)
        hashes.add(row["sha256"])
        ids.add(row["id"])
        originals.add(row["original_sha1"])
    return manifest


def download_estimate(count, approved_gb=None):
    if approved_gb is not None and (not math.isfinite(approved_gb) or approved_gb <= 0):
        raise ValueError("approved download bound must be positive and finite")
    budget = Decimal(DEFAULT_DOWNLOAD_BYTES) if approved_gb is None else Decimal(str(approved_gb)) * 1_000_000_000
    upper = count * MAX_IMAGE_BYTES
    return {"photos": count, "maximum_image_bytes": MAX_IMAGE_BYTES,
            "download_upper_bound_gb": upper / 1e9,
            "download_budget_gb": float(budget / 1_000_000_000),
            "approval_required": upper > budget}


def enforce_download_budget(count, approved_gb=None):
    estimate = download_estimate(count, approved_gb)
    if estimate["approval_required"]:
        raise ValueError(f"download upper bound {estimate['download_upper_bound_gb']:g} GB exceeds "
                         f"{estimate['download_budget_gb']:g} GB budget; obtain approval and pass --approved-download-gb")


def build(directory, count, approved_gb):
    enforce_download_budget(count, approved_gb)
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "manifest.json").exists():
        manifest = verify(directory)
        if manifest["count"] != count:
            raise ValueError("existing set has a different count")
        return manifest
    selection = directory / "selection.json"
    excluded = json.loads(EVAL.read_text())
    if selection.exists():
        plan = json.loads(selection.read_text())
        if plan["count"] != count or plan["seed"] != SEED or plan["eval_sha256"] != hashlib.sha256(EVAL.read_bytes()).hexdigest():
            raise ValueError("selection inputs changed; use a separate destination")
    else:
        plan = {"count": count, "seed": SEED, "fit_count": count * 3 // 4,
                "eval_sha256": hashlib.sha256(EVAL.read_bytes()).hexdigest(),
                "photos": split_rows(discover(count, excluded), count * 3 // 4)}
        write_json(selection, plan)
    (directory / "images").mkdir(exist_ok=True)
    hashes = {row["sha256"] for row in excluded}
    for row in plan["photos"]:
        path = directory / "images" / f"{row['id']}.jpg"
        if not path.exists():
            row["licence_metadata"] = verify_source(row)
            row["licence_checked_at"] = datetime.now(timezone.utc).isoformat()
        data = path.read_bytes() if path.exists() else download(row)
        validate_jpeg(data)
        if len(data) > MAX_IMAGE_BYTES:
            raise ValueError("photo exceeds byte limit")
        digest = hashlib.sha256(data).hexdigest()
        if digest in hashes or ("sha256" in row and row["sha256"] != digest):
            raise ValueError("duplicate/overlapping or changed photo bytes")
        hashes.add(digest)
        row["sha256"] = digest
        row["bytes"] = len(data)
        if not path.exists():
            pending = path.with_suffix(".part")
            pending.write_bytes(data)
            pending.replace(path)
        write_json(selection, plan)
    plan["verified_at"] = datetime.now(timezone.utc).isoformat()
    write_json(directory / "manifest.json", plan)
    return verify(directory)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path.home() / ".cache/seyk-bench/photo-calibration")
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--approved-download-gb", type=float)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--download", action="store_true")
    mode.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.count < 1000:
        parser.error("at least 1,000 separate calibration photos are required")
    print(json.dumps(download_estimate(args.count, args.approved_download_gb)), flush=True)
    if args.download:
        build(args.directory, args.count, args.approved_download_gb)
    elif args.verify:
        verify(args.directory)


if __name__ == "__main__":
    main()
