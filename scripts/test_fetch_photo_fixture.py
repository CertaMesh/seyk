"""Offline tests for the pinned fixture fetcher's retry policy."""
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import io
import hashlib
import json
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import zlib
import unittest
from unittest.mock import patch, MagicMock
import urllib.error

import fetch_photo_fixture as fixture


class FetchTests(unittest.TestCase):
    def error(self, status, retry=None):
        headers = {} if retry is None else {"Retry-After": retry}
        return urllib.error.HTTPError("https://example.invalid/pinned", status, "busy", headers, io.BytesIO())

    def test_bounded_retries_grow_and_keep_the_same_request(self):
        errors = [self.error(429, "45"), self.error(503), self.error(429, "2"), self.error(429)]
        with patch.object(fixture.OPENER, "open", side_effect=errors) as opened, patch.object(fixture.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "after 3 retries"):
                fixture.fetch("https://example.invalid/pinned")
        self.assertEqual(opened.call_count, 4)
        self.assertEqual([c.args[0] for c in sleep.call_args_list], [1, 45, 1, 90, 1, 180, 1])
        requests = [call.args[0] for call in opened.call_args_list]
        self.assertTrue(all(request is requests[0] for request in requests))
        self.assertIn("https://github.com/CertaMesh/seyk", requests[0].get_header("User-agent"))

    def test_success_after_retry_and_nonretryable_failure(self):
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b"pinned bytes"
        with patch.object(fixture.OPENER, "open", side_effect=[self.error(503), response]), patch.object(fixture.time, "sleep"):
            self.assertEqual(fixture.fetch("https://example.invalid/pinned").data, b"pinned bytes")
        with patch.object(fixture.OPENER, "open", side_effect=self.error(404)) as opened, patch.object(fixture.time, "sleep"):
            with self.assertRaises(urllib.error.HTTPError):
                fixture.fetch("https://example.invalid/pinned")
            self.assertEqual(opened.call_count, 1)

    def test_retry_after_accepts_seconds_dates_and_bad_headers(self):
        self.assertEqual(fixture.retry_after("120"), 120)
        self.assertEqual(fixture.retry_after(None), 30)
        self.assertEqual(fixture.retry_after("bad"), 30)
        self.assertEqual(fixture.retry_after("inf"), 30)
        self.assertEqual(fixture.retry_after("nan"), 30)
        future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=120))
        self.assertTrue(118 <= fixture.retry_after(future) <= 120)


class RepinTests(unittest.TestCase):
    def row(self):
        return {"id": "pinned-photo", "sha256": hashlib.sha256(b"old").hexdigest(),
                "url": "https://thumb.wikimedia.org/wikipedia/commons/thumb/a/a0/photo.jpg/960px-photo.jpg",
                "source_page": "https://commons.wikimedia.org/wiki/File:Photo.jpg",
                "licence": "CC0-1.0", "licence_url": "https://creativecommons.org/publicdomain/zero/1.0/"}

    def original(self, row):
        return tuple(row[key] for key in ("url", "source_page", "licence", "licence_url"))

    def test_status_type_url_width_and_source_are_required(self):
        row = self.row()
        download = fixture.Download(b"jpeg", 200, "image/jpeg", row["url"])
        with patch.object(fixture, "jpeg_width", return_value=960):
            fixture.validate_repin(row, download, self.original(row))
            for bad in [download._replace(status=206), download._replace(content_type="text/html"),
                        download._replace(url="https://other.invalid/photo.jpg")]:
                with self.assertRaises(ValueError):
                    fixture.validate_repin(row, bad, self.original(row))
            original = self.original(row)
            for key in ("source_page", "licence", "licence_url"):
                altered = dict(row, **{key: "changed"})
                with self.assertRaises(ValueError):
                    fixture.validate_repin(altered, download, original)
        with patch.object(fixture, "jpeg_width", return_value=959):
            with self.assertRaisesRegex(ValueError, "width"):
                fixture.validate_repin(row, download, self.original(row))
        with self.assertRaisesRegex(ValueError, "JPEG bytes"):
            fixture.jpeg_width(b"<html>not an image</html>")
        with self.assertRaisesRegex(ValueError, "redirect"):
            fixture.PinnedUrlOnly().redirect_request(
                fixture.urllib.request.Request(row["url"]), None, 302, "redirect", {}, "https://other.invalid/photo.jpg")

    def test_repin_updates_only_hash_and_provenance_and_verify_is_offline(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            dest = root / "photos"
            row = self.row()
            manifest.write_text(json.dumps([row]))
            data = b"\xff\xd8new JPEG bytes"
            download = fixture.Download(data, 200, "image/jpeg", row["url"])
            with patch.object(fixture, "MANIFEST", manifest), patch.object(fixture, "DEST", dest):
                with patch.object(fixture, "fetch", return_value=download), patch("sys.argv", ["fetcher"]):
                    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                        fixture.main()
                self.assertFalse((dest / "pinned-photo.jpg").exists())
                with patch.object(fixture, "fetch", return_value=download), patch.object(fixture, "jpeg_width", return_value=960), patch("sys.argv", ["fetcher", "--repin"]), patch.object(fixture, "datetime", wraps=datetime) as clock:
                    clock.now.return_value = datetime(2031, 1, 2, 0, 5, tzinfo=timezone.utc)
                    fixture.main()
                    clock.now.assert_called_once_with(timezone.utc)
                changed = json.loads(manifest.read_text())[0]
                self.assertEqual(changed.pop("repinned"), {"date": "2031-01-02", "previous_sha256": row["sha256"]})
                self.assertEqual(changed.pop("sha256"), hashlib.sha256(data).hexdigest())
                self.assertEqual(changed, {k: v for k, v in row.items() if k != "sha256"})
                with patch.object(fixture, "fetch", side_effect=AssertionError("network forbidden")), patch("sys.argv", ["fetcher", "--verify"]):
                    fixture.main()
                (dest / "pinned-photo.jpg").write_bytes(b"corrupt cached bytes")
                with patch.object(fixture, "fetch", side_effect=AssertionError("network forbidden")), patch("sys.argv", ["fetcher", "--repin"]):
                    with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                        fixture.main()

    @unittest.skipUnless(shutil.which("sips"), "native JPEG decoder test requires macOS sips")
    def test_native_decoder_reads_all_pixels(self):
        def chunk(kind, data):
            return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
        png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 3, 2, 8, 2, 0, 0, 0))
        png += chunk(b"IDAT", zlib.compress((b"\x00" + b"\xff\x00\x00" * 3) * 2)) + chunk(b"IEND", b"")
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.png"
            jpeg = Path(directory) / "source.jpg"
            source.write_bytes(png)
            subprocess.run(["sips", "-s", "format", "jpeg", str(source), "--out", str(jpeg)], check=True, capture_output=True)
            self.assertEqual(fixture.jpeg_width(jpeg.read_bytes()), 3)


if __name__ == "__main__":
    unittest.main()
