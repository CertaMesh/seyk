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
            if read(f"{root}/{name}") != Path(name).read_bytes():
                raise ValueError(f"licence file differs: {name}")
        # The generator writes platform-native HTML newlines on Windows.
        content = read(f"{root}/THIRD_PARTY_LICENSES.html").decode("utf-8").replace("\r\n", "\n")
        if len(content) <= 1000:
            raise ValueError("empty dependency licence bundle")
        notices = Path("third_party/onnxruntime")
        for source in notices.iterdir():
            if source.is_file():
                if read(f"{root}/{source.as_posix()}") != source.read_bytes():
                    raise ValueError(f"native notice file differs: {source}")
        native = (notices / "ThirdPartyNotices.txt").read_bytes()
        expected = (notices / "SHA256SUMS").read_text().split()[0]
        if hashlib.sha256(native).hexdigest() != expected:
            raise ValueError("ORT notice checksum mismatch")
        if html.escape(native.decode("utf-8")) not in content:
            raise ValueError("missing embedded ORT notices")
    print(f"Verified licence notices in {archive}")


if __name__ == "__main__":
    try:
        check_archive(sys.argv[1])
    except (ValueError, KeyError, OSError, tarfile.TarError, zipfile.BadZipFile) as error:
        raise SystemExit(f"Release archive verification failed: {error}")
