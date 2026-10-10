#!/usr/bin/env python3
"""Exercise the load runner with synthetic files and fake test executables."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

RUNNER = Path(__file__).with_name("audio_golden_load.py")


class AudioGoldenLoadTests(unittest.TestCase):
    def fixture(self, root):
        home = root / "hf"
        weight = home / "hub/models--fixture/snapshots/pinned/weight"
        weight.parent.mkdir(parents=True)
        weight.write_bytes(b"synthetic weights")
        return home, weight

    def run_fixture(self, root, home, binary, env=None):
        return subprocess.run(
            [sys.executable, str(RUNNER), str(binary), "--hf-home", str(home),
             "--runs", "3", "--parallel", "2", "--stress-workers", "0",
             "--output", str(root / "results")], capture_output=True, text=True, env=env,
        )

    def test_early_aborts_continue_and_still_check_fixture_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home, weight = self.fixture(root)
            binary = root / "fake-test"
            binary.write_text("#!/bin/sh\necho 'unknown exception in Initialize()' >&2\nexit 7\n")
            binary.chmod(0o700)
            result = self.run_fixture(root, home, binary)
            self.assertEqual(result.returncode, 1, result.stderr)
            summary = json.loads((root / "results/summary.json").read_text())
            self.assertEqual(summary["total"], 3)
            self.assertEqual(summary["failed"], 3)
            self.assertEqual(summary["ort_exceptions"], 3)
            self.assertTrue(summary["fixture_hashes_unchanged"])
            self.assertEqual(weight.read_bytes(), b"synthetic weights")
            self.assertEqual(len(list((root / "results").glob("run-*.log"))), 3)

    def test_launch_failures_are_recorded_without_aborting_the_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home, _ = self.fixture(root)
            result = self.run_fixture(root, home, root / "absent-binary")
            self.assertEqual(result.returncode, 1, result.stderr)
            summary = json.loads((root / "results/summary.json").read_text())
            self.assertEqual(summary["failed"], 3)
            self.assertTrue(summary["fixture_hashes_unchanged"])

    def test_configured_and_default_hf_homes_are_rejected_before_running(self):
        for configured in [True, False]:
            with self.subTest(configured=configured), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                home = root / ("configured-hf" if configured else ".cache/huggingface")
                weight = home / "hub/models--fixture/snapshots/pinned/weight"
                weight.parent.mkdir(parents=True)
                weight.write_bytes(b"synthetic weights")
                env = {**os.environ, "HOME": str(root)}
                env.pop("HF_HOME", None)
                if configured:
                    env["HF_HOME"] = str(home)
                result = self.run_fixture(root, home, root / "absent-binary", env)
                self.assertEqual(result.returncode, 2, result.stderr)
                self.assertIn("configured or default HF home", result.stderr)
                self.assertFalse((root / "results").exists())
                self.assertEqual(weight.read_bytes(), b"synthetic weights")

    def test_hard_linked_fixtures_are_rejected_before_running(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            home, weight = self.fixture(root)
            os.link(weight, root / "shared-source")
            result = self.run_fixture(root, home, root / "absent-binary")
            self.assertEqual(result.returncode, 2)
            self.assertIn("st_nlink == 1", result.stderr)
            self.assertFalse((root / "results").exists())
            self.assertEqual(weight.read_bytes(), b"synthetic weights")


if __name__ == "__main__":
    unittest.main()
