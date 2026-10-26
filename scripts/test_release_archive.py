#!/usr/bin/env python3
"""Check archive validation against Unix and Windows notice representations."""
import contextlib
import html
import io
from pathlib import Path
import tarfile
import tempfile
import unittest
import zipfile

from check_release_archive import check_archive


class ArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.files = {name: Path(name).read_bytes() for name in ("LICENSE", "THIRD_PARTY.md")}
        for source in Path("third_party/onnxruntime").iterdir():
            if source.is_file():
                self.files[source.as_posix()] = source.read_bytes()
        native = self.files["third_party/onnxruntime/ThirdPartyNotices.txt"].decode("utf-8")
        self.files["THIRD_PARTY_LICENSES.html"] = ("<html>" + html.escape(native) + "</html>").encode()

    def check(self, suffix):
        archive = Path(self.temp.name) / f"seyk-fixture{suffix}"
        if suffix == ".zip":
            with zipfile.ZipFile(archive, "w") as bundle:
                for name, data in self.files.items():
                    bundle.writestr(f"seyk-fixture/{name}", data)
        else:
            with tarfile.open(archive, "w:gz") as bundle:
                for name, data in self.files.items():
                    entry = tarfile.TarInfo(f"seyk-fixture/{name}")
                    entry.size = len(data)
                    bundle.addfile(entry, io.BytesIO(data))
        with contextlib.redirect_stdout(io.StringIO()):
            check_archive(archive)

    def test_unix_tar_and_windows_zip(self):
        self.check(".tar.gz")
        self.files["THIRD_PARTY_LICENSES.html"] = self.files["THIRD_PARTY_LICENSES.html"].replace(b"\n", b"\r\n")
        self.check(".zip")

    def test_changed_native_notice_is_rejected(self):
        self.files["third_party/onnxruntime/ThirdPartyNotices.txt"] += b"changed"
        for suffix in (".tar.gz", ".zip"):
            with self.subTest(suffix=suffix), self.assertRaises(AssertionError):
                self.check(suffix)

    def test_missing_licence_bundle_is_rejected(self):
        del self.files["THIRD_PARTY_LICENSES.html"]
        for suffix in (".tar.gz", ".zip"):
            with self.subTest(suffix=suffix), self.assertRaises(KeyError):
                self.check(suffix)

    def test_missing_embedded_notices_are_rejected(self):
        self.files["THIRD_PARTY_LICENSES.html"] = b"<html>" + b"x" * 2000 + b"</html>"
        for suffix in (".tar.gz", ".zip"):
            with self.subTest(suffix=suffix), self.assertRaisesRegex(AssertionError, "embedded ORT"):
                self.check(suffix)


if __name__ == "__main__":
    unittest.main()
