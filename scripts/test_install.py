#!/usr/bin/env python3
"""Exercise the installer with public-download fixtures and no installed gh."""
import hashlib
import io
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

INSTALLER = Path(__file__).with_name("install.sh").resolve()
NAME = "seyk-0.2.0-x86_64-unknown-linux-gnu"


class InstallTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.bin = self.root / "bin"
        self.bin.mkdir()
        self.install_dir = self.root / "installed"
        self.assets = self.root / "assets"
        self.assets.mkdir()
        archive = self.assets / f"{NAME}.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            binary = b'#!/bin/sh\necho "seyk 0.2.0"\n'
            entry = tarfile.TarInfo(f"{NAME}/seyk")
            entry.size = len(binary)
            entry.mode = 0o755
            bundle.addfile(entry, io.BytesIO(binary))
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (self.assets / f"{NAME}.tar.gz.sha256").write_text(f"{digest}  {NAME}.tar.gz\n")
        for command in ("sed", "awk", "mktemp", "rm", "tar", "gzip", "mkdir", "install", "shasum"):
            source = shutil.which(command)
            self.assertIsNotNone(source, command)
            (self.bin / command).symlink_to(source)
        # Force the same supported platform even on macOS test hosts.
        self.write_command("uname", '#!/bin/sh\ncase "$1" in -s) echo Linux ;; -m) echo x86_64 ;; esac\n')
        self.env = dict(os.environ, PATH=str(self.bin), SEYK_INSTALL_DIR=str(self.install_dir),
                        FIXTURE_ASSETS=str(self.assets), FIXTURE_LOG=str(self.root / "requests"))
        self.env.pop("SEYK_VERSION", None)
        self.env.pop("SEYK_DOWNLOADER", None)

    def write_command(self, name, content):
        path = self.bin / name
        path.write_text(content)
        path.chmod(0o755)

    def downloader(self, name):
        self.write_command(name, f'#!{sys.executable}\n' + '''
import os, pathlib, shutil, sys
args = sys.argv[1:]
assets = pathlib.Path(os.environ["FIXTURE_ASSETS"])
with open(os.environ["FIXTURE_LOG"], "a") as log:
    log.write(" ".join(args) + "\\n")
if os.environ.get("FIXTURE_FAIL"):
    sys.exit(22)
if args[0] == "api":
    print("v0.2.0")
elif args[0] == "release":
    dest = pathlib.Path(args[args.index("--dir") + 1])
    for index, arg in enumerate(args):
        if arg == "--pattern":
            shutil.copyfile(assets / args[index + 1], dest / args[index + 1])
else:
    url = next(arg for arg in args if arg.startswith("https://"))
    output = pathlib.Path(args[args.index("-o" if "-o" in args else "-O") + 1])
    if url.endswith("/latest"):
        output.write_text('{\\n  "tag_name": "v0.2.0",\\n}\\n')
    else:
        shutil.copyfile(assets / url.rsplit("/", 1)[1], output)
''')

    def run_install(self):
        return subprocess.run(["/bin/sh", str(INSTALLER)], env=self.env, text=True, capture_output=True)

    def test_latest_without_gh(self):
        for name in ("curl", "wget"):
            with self.subTest(downloader=name):
                self.downloader(name)
                result = self.run_install()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertTrue((self.install_dir / "seyk").is_file())
                self.assertIn("seyk 0.2.0", result.stdout)
                self.assertIn("/releases/latest", (self.root / "requests").read_text())
                (self.bin / name).unlink()

    def test_pinned_and_optional_gh(self):
        for name in ("curl", "wget", "gh"):
            with self.subTest(downloader=name):
                self.downloader(name)
                self.env.update(SEYK_DOWNLOADER=name, SEYK_VERSION="v0.2.0")
                result = self.run_install()
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertNotIn("/releases/latest", (self.root / "requests").read_text())
                (self.bin / name).unlink()

    def test_corrupt_archive_does_not_replace_existing_binary(self):
        self.downloader("curl")
        self.install_dir.mkdir()
        binary = self.install_dir / "seyk"
        binary.write_text("existing")
        with (self.assets / f"{NAME}.tar.gz").open("ab") as archive:
            archive.write(b"corruption")
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Checksum mismatch", result.stderr)
        self.assertEqual(binary.read_text(), "existing")

    def test_download_failure(self):
        self.downloader("curl")
        self.env["FIXTURE_FAIL"] = "1"
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Download failed", result.stderr)
        self.assertFalse(self.install_dir.exists())

    def test_missing_downloader(self):
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Install curl or wget", result.stderr)

    def test_missing_checksum_tool(self):
        self.downloader("curl")
        (self.bin / "shasum").unlink()
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("SHA-256 verification requires", result.stderr)

    def test_invalid_checksum(self):
        self.downloader("wget")
        (self.assets / f"{NAME}.tar.gz.sha256").write_text("bad checksum\n")
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Invalid SHA-256", result.stderr)
        self.assertFalse(self.install_dir.exists())


if __name__ == "__main__":
    unittest.main()
