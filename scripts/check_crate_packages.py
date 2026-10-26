#!/usr/bin/env python3
"""Review crates.io file lists and built archives. Run packaging in CI first."""

import hashlib
import json
from pathlib import Path, PurePosixPath
import subprocess
import shutil
import tarfile
import tomllib

# Dependencies precede consumers. Cargo's workspace dry-run uses the same graph.
CRATES = ("seyk-embed", "seyk-embeddinggemma", "seyk-core", "seyk-stt", "seyk")
FORBIDDEN = {"tests", "fixtures", "benches", "examples", "target", ".github", "scripts"}
MAX_FILE = 512_000
MAX_ARCHIVE = 1_000_000
MAX_UNPACKED = 3_000_000


def review_files(name, files):
    required = {"Cargo.toml", "Cargo.lock", "README.md", "LICENSE", "NOTICE"}
    if name == "seyk-embed":
        required.add("LICENSE-CC-BY-4.0")
    if name == "seyk-stt":
        required.add("LICENSE-MIT")
    assert required <= set(files), (name, "missing files", required - set(files))
    for file in files:
        assert not FORBIDDEN.intersection(PurePosixPath(file).parts), (name, file)
        assert file.startswith(("src/", "assets/", "LICENSE", "NOTICE")) or file in {
            "Cargo.toml", "Cargo.toml.orig", "Cargo.lock", ".cargo_vcs_info.json",
            "README.md", "build.rs",
        }, (name, "unexpected file", file)
    assert not any("golden" in file or "witness" in file for file in files), name


def main():
    version = tomllib.loads(Path("Cargo.toml").read_text())["workspace"]["package"]["version"]
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    sizes = {}
    for name in CRATES:
        output = subprocess.check_output(
            ["cargo", "package", "--list", "-p", name, "--locked"], text=True
        )
        print(f"=== {name} cargo package --list ===\n{output}", flush=True)
        files = output.splitlines()
        review_files(name, files)
        archive = Path("target/package") / f"{name}-{version}.crate"
        # Workspace publication moves archives into Cargo's staging registry.
        # Copy those exact verified bytes to the artifact directory.
        if not archive.exists():
            staged = list(Path("target/package").rglob(archive.name))
            assert staged, (name, "missing staged archive")
            # Cargo retains copies in both tmp-crate and tmp-registry. Compare
            # actual contents instead of assuming there is only one path.
            hashes = {hashlib.sha256(path.read_bytes()).hexdigest() for path in staged}
            assert len(hashes) == 1, (name, "staged archive contents differ", staged)
            shutil.copyfile(staged[0], archive)
        with tarfile.open(archive) as package:
            members = package.getmembers()
            prefix = f"{name}-{version}/"
            assert all(member.isfile() and member.name.startswith(prefix) for member in members)
            packaged_files = [member.name.removeprefix(prefix) for member in members]
            review_files(name, packaged_files)
            assert set(files) == set(packaged_files), (name, "list/archive mismatch")
            assert all(member.size <= MAX_FILE for member in members), name
            unpacked = sum(member.size for member in members)
            assert unpacked <= MAX_UNPACKED, (name, unpacked)
            vcs = json.load(package.extractfile(prefix + ".cargo_vcs_info.json"))
            assert vcs["git"]["sha1"] == commit and not vcs["git"].get("dirty", False), name
            readme = package.extractfile(prefix + "README.md").read().decode()
            assert "https://github.com/CertaMesh/seyk" in readme
        compressed = archive.stat().st_size
        assert compressed <= MAX_ARCHIVE, (name, compressed)
        sizes[name] = {"archive_bytes": compressed, "unpacked_bytes": unpacked}
        print(f"SIZE {name}={compressed / 1000:.1f}KB unpacked={unpacked / 1000:.1f}KB", flush=True)
    Path("target/package/sizes.json").write_text(json.dumps(sizes, indent=2) + "\n")


if __name__ == "__main__":
    main()
