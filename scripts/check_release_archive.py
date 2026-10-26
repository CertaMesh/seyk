#!/usr/bin/env python3
"""Verify notices in the actual distributable, including byte-exact ORT files."""
import hashlib
import html
from pathlib import Path
import sys
import tarfile
import zipfile


def check_archive(archive):
    archive = Path(archive)
    root = archive.name.removesuffix(".tar.gz").removesuffix(".zip")
    if archive.suffix == ".zip":
        bundle = zipfile.ZipFile(archive)
        read = bundle.read
    else:
        bundle = tarfile.open(archive)

        def read(name):
            return bundle.extractfile(name).read()

    with bundle:
        for name in ("LICENSE", "THIRD_PARTY.md"):
            assert read(f"{root}/{name}") == Path(name).read_bytes(), name
        # The generator writes platform-native HTML newlines on Windows.
        content = read(f"{root}/THIRD_PARTY_LICENSES.html").decode("utf-8").replace("\r\n", "\n")
        assert len(content) > 1000, "empty dependency licence bundle"
        notices = Path("third_party/onnxruntime")
        for source in notices.iterdir():
            if source.is_file():
                assert read(f"{root}/{source.as_posix()}") == source.read_bytes(), source
        native = (notices / "ThirdPartyNotices.txt").read_bytes()
        expected = (notices / "SHA256SUMS").read_text().split()[0]
        assert hashlib.sha256(native).hexdigest() == expected, "ORT notice checksum"
        assert html.escape(native.decode("utf-8")) in content, "embedded ORT notices"
    print(f"Verified licence notices in {archive}")


if __name__ == "__main__":
    check_archive(sys.argv[1])
