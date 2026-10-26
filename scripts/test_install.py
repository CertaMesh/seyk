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
        self.create_archive()
        for command in ("sed", "awk", "mktemp", "rm", "tar", "gzip", "mkdir", "install", "mv", "shasum"):
            source = shutil.which(command)
            self.assertIsNotNone(source, command)
            (self.bin / command).symlink_to(source)
        # Force the same supported platform even on macOS test hosts.
        self.write_command("uname", '#!/bin/sh\ncase "$1" in -s) echo Linux ;; -m) echo x86_64 ;; esac\n')
        self.env = dict(os.environ, PATH=str(self.bin), SEYK_INSTALL_DIR=str(self.install_dir),
                        FIXTURE_ASSETS=str(self.assets), FIXTURE_LOG=str(self.root / "requests"))
        self.env.pop("SEYK_VERSION", None)
        self.env.pop("SEYK_DOWNLOADER", None)

    def create_archive(self, binary=b'#!/bin/sh\necho "seyk 0.2.0"\n', kind=tarfile.REGTYPE):
        archive = self.assets / f"{NAME}.tar.gz"
        with tarfile.open(archive, "w:gz") as bundle:
            entry = tarfile.TarInfo(f"{NAME}/seyk")
            entry.type = kind
            entry.mode = 0o755
            if kind == tarfile.REGTYPE:
                entry.size = len(binary)
                bundle.addfile(entry, io.BytesIO(binary))
            else:
                entry.linkname = "/bin/sh"
                bundle.addfile(entry)
        digest = hashlib.sha256(archive.read_bytes()).hexdigest()
        (self.assets / f"{NAME}.tar.gz.sha256").write_text(f"{digest}  {NAME}.tar.gz\n")

    def existing_binary(self):
        self.install_dir.mkdir()
        binary = self.install_dir / "seyk"
        binary.write_text("existing")
        return binary

    def assert_no_staged_binary(self):
        self.assertEqual(list(self.install_dir.glob(".seyk.*")), [])

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
                requests = (self.root / "requests").read_text()
                self.assertIn("/releases/latest", requests)
                if name == "curl":
                    self.assertIn("--proto =https --proto-redir =https --tlsv1.2", requests)
                else:
                    self.assertIn("--https-only", requests)
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
        self.assertIn("set SEYK_VERSION=vX.Y.Z", result.stderr)
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

    def test_truncated_pipe_does_not_execute_installer(self):
        self.downloader("curl")
        source = INSTALLER.read_text()
        invocation = source.rindex('main "$@"')
        for size in (len(source) // 3, len(source) // 2, invocation):
            with self.subTest(size=size):
                subprocess.run(["/bin/sh"], input=source[:size], env=self.env,
                               text=True, capture_output=True)
                self.assertFalse((self.root / "requests").exists())
                self.assertFalse(self.install_dir.exists())

    def test_failed_version_preserves_previous_installation(self):
        self.downloader("curl")
        binary = self.existing_binary()
        self.create_archive(b"#!/bin/sh\nexit 42\n")
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Downloaded binary cannot run", result.stderr)
        self.assertEqual(binary.read_text(), "existing")
        self.assert_no_staged_binary()

    def test_partial_install_failure_is_cleaned_up(self):
        self.downloader("curl")
        binary = self.existing_binary()
        (self.bin / "install").unlink()
        self.write_command("install", '#!/bin/sh\necho partial > "$4"\nexit 1\n')
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot install", result.stderr)
        self.assertEqual(binary.read_text(), "existing")
        self.assert_no_staged_binary()

    def test_smoke_check_precedes_atomic_replacement(self):
        self.downloader("wget")
        binary = self.existing_binary()
        self.env["FIXTURE_SMOKE"] = str(self.root / "smoke")
        self.create_archive(b'#!/bin/sh\necho "$0" > "$FIXTURE_SMOKE"\necho "seyk 0.2.0"\n')
        result = self.run_install()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(str(self.install_dir / ".seyk."), (self.root / "smoke").read_text())
        self.assertIn("seyk 0.2.0", binary.read_text())
        self.assert_no_staged_binary()

    def test_failed_rename_preserves_previous_installation(self):
        self.downloader("curl")
        binary = self.existing_binary()
        (self.bin / "mv").unlink()
        self.write_command("mv", "#!/bin/sh\nexit 1\n")
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Cannot replace", result.stderr)
        self.assertEqual(binary.read_text(), "existing")
        self.assert_no_staged_binary()

    def test_destination_directory_is_rejected(self):
        self.downloader("curl")
        (self.install_dir / "seyk").mkdir(parents=True)
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("destination is a directory", result.stderr)
        self.assertEqual(list((self.install_dir / "seyk").iterdir()), [])
        self.assert_no_staged_binary()

    def test_non_regular_binary_is_rejected(self):
        self.downloader("curl")
        binary = self.existing_binary()
        for kind in (tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.DIRTYPE, tarfile.FIFOTYPE):
            with self.subTest(kind=kind):
                self.create_archive(kind=kind)
                result = self.run_install()
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("one regular seyk binary", result.stderr)
                self.assertEqual(binary.read_text(), "existing")
                self.assert_no_staged_binary()

    def test_empty_gh_checksum_is_rejected(self):
        self.downloader("gh")
        self.env["SEYK_DOWNLOADER"] = "gh"
        (self.assets / f"{NAME}.tar.gz.sha256").write_text("")
        result = self.run_install()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Missing or empty checksum", result.stderr)
        self.assertFalse(self.install_dir.exists())

    def test_sha256sum(self):
        source = shutil.which("sha256sum")
        if source is None:
            self.skipTest("sha256sum is not installed")
        (self.bin / "sha256sum").symlink_to(source)
        (self.bin / "shasum").unlink()
        self.downloader("curl")
        result = self.run_install()
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
