"""Synthetic/offline tests. Never fetch images or load model weights."""
import io
import copy
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
from PIL import Image

import build_set as builder
import measure
import run


class BuilderTests(unittest.TestCase):
    def page(self):
        return {"pageid": 123, "title": "File:Test photo.jpg", "imageinfo": [{
            "extmetadata": {"LicenseShortName": {"value": "CC0"}, "LicenseUrl": {"value": builder.LICENCE}},
            "mime": "image/jpeg", "thumbmime": "image/jpeg", "thumbwidth": 960,
            "descriptionurl": "https://commons.wikimedia.org/wiki/File:Test_photo.jpg",
            "thumburl": "https://upload.wikimedia.org/test.jpg", "sha1": "a" * 40,
            "timestamp": "2026-01-01T00:00:00Z"}]}

    def test_per_image_licence_and_source_verification(self):
        page = self.page()
        self.assertEqual(builder.image_row(page, set())["licence"], "CC0-1.0")
        for key, value in (("LicenseShortName", "CC BY 4.0"), ("LicenseUrl", "https://evil.invalid/CC0")):
            altered = self.page()
            altered["imageinfo"][0]["extmetadata"][key]["value"] = value
            with self.assertRaisesRegex(ValueError, "explicit CC0"):
                builder.image_row(altered, set())
        with self.assertRaisesRegex(ValueError, "separate JPEG"):
            builder.image_row(page, {builder.title_key("Test_photo.jpg")})
        page["imageinfo"][0]["thumburl"] = "https://evil.invalid/photo.jpg"
        with self.assertRaisesRegex(ValueError, "source URL"):
            builder.image_row(page, set())

    def test_licence_url_normalization_stays_on_the_exact_CC0_document(self):
        self.assertTrue(builder.cc0_url("http://creativecommons.org/publicdomain/zero/1.0/deed.en"))
        self.assertFalse(builder.cc0_url("https://creativecommons.org.evil.invalid/publicdomain/zero/1.0/"))
        self.assertFalse(builder.cc0_url("https://creativecommons.org/licenses/by/4.0/"))
        row = builder.image_row(self.page(), set())
        with patch.object(builder, "request_json", return_value={"query": {"pages": [self.page()]}}):
            builder.verify_source(row)
        changed = self.page()
        changed["imageinfo"][0]["sha1"] = "changed"
        with patch.object(builder, "request_json", return_value={"query": {"pages": [changed]}}):
            with self.assertRaisesRegex(ValueError, "source changed"):
                builder.verify_source(row)

    def test_split_is_fixed_before_fitting_and_independent_of_input_order(self):
        rows = [{"id": f"commons-{i}"} for i in range(2000)]
        first = builder.split_rows(rows, 1500)
        self.assertEqual(first, builder.split_rows(list(reversed(rows)), 1500))
        self.assertEqual(sum(r["split"] == "fit" for r in first), 1500)
        self.assertEqual(sum(r["split"] == "heldout" for r in first), 500)

    def test_download_budget_boundaries_precede_discovery(self):
        cases = ((2000, None, True), (2001, None, False),
                 (2001, 4.002, True), (2001, 4.0, False),
                 (2000, 3.999, False), (2000, 4.0, True))
        with tempfile.TemporaryDirectory() as directory:
            for count, approved, allowed in cases:
                with self.subTest(count=count, approved=approved):
                    with patch.object(builder, "discover", side_effect=RuntimeError("discovery reached")) as discover:
                        if allowed:
                            with self.assertRaisesRegex(RuntimeError, "discovery reached"):
                                builder.build(Path(directory), count, approved)
                            discover.assert_called_once()
                        else:
                            with self.assertRaisesRegex(ValueError, "exceeds .* GB budget"):
                                builder.build(Path(directory), count, approved)
                            discover.assert_not_called()
            for invalid in (float("nan"), float("inf"), 0, -1):
                with patch.object(builder, "discover") as discover:
                    with self.assertRaisesRegex(ValueError, "positive and finite"):
                        builder.build(Path(directory), 2000, invalid)
                    discover.assert_not_called()

    def test_both_entry_points_report_the_effective_budget_and_estimate_offline(self):
        for module, action in ((builder, "--download"), (run, "--execute")):
            for count, approved, required in ((2000, None, False), (2001, None, True),
                                              (2001, "4.002", False), (2000, "3.999", True)):
                args = ["entry", "--count", str(count)]
                if approved is not None:
                    args += ["--approved-download-gb", approved]
                with self.subTest(entry=module.__name__, count=count, approved=approved):
                    with patch.object(module.sys, "argv", args), patch.object(module, "build", side_effect=AssertionError("network")), patch.object(run.subprocess, "run", side_effect=AssertionError("model")), patch("sys.stdout", new_callable=io.StringIO) as output:
                        module.main()
                    estimate = json.loads(output.getvalue())
                    self.assertEqual(estimate["approval_required"], required)
                    self.assertEqual(estimate["download_budget_gb"], 4.0 if approved is None else float(approved))
                    self.assertEqual(estimate["download_upper_bound_gb"], count * 2_000_000 / 1e9)
                    if required:
                        with patch.object(module.sys, "argv", args + [action]), patch.object(builder, "discover") as discover, patch.object(run.subprocess, "run") as subprocess, patch("sys.stdout", new_callable=io.StringIO):
                            with self.assertRaisesRegex(ValueError, "exceeds .* GB budget"):
                                module.main()
                            discover.assert_not_called()
                            subprocess.assert_not_called()

    def test_download_is_bounded_and_bad_responses_fail(self):
        response = MagicMock()
        response.status = 200
        response.headers.get_content_type.return_value = "image/jpeg"
        response.read.return_value = b"x" * (builder.MAX_IMAGE_BYTES + 1)
        with patch.object(builder.fetcher.OPENER, "open") as opened, patch.object(builder.fetcher.time, "sleep"):
            opened.return_value.__enter__.return_value = response
            with self.assertRaisesRegex(ValueError, "2 MB"):
                builder.download({"url": "https://upload.wikimedia.org/a.jpg"})
            response.read.assert_called_once_with(builder.MAX_IMAGE_BYTES + 1)
            response.status = 206
            with self.assertRaisesRegex(ValueError, "HTTP 200"):
                builder.download({"url": "https://upload.wikimedia.org/a.jpg"})

    def http_error(self, status, retry=None):
        headers = {} if retry is None else {"Retry-After": retry}
        return builder.fetcher.urllib.error.HTTPError("https://upload.wikimedia.org/a.jpg", status,
                                                     "busy", headers, io.BytesIO())

    def response(self):
        response = MagicMock()
        response.status = 200
        response.headers.get_content_type.return_value = "image/jpeg"
        response.read.return_value = b"JPEG bytes"
        response.__enter__.return_value = response
        return response

    def test_image_retries_are_bounded_and_honour_retry_after(self):
        errors = [self.http_error(429, "45"), self.http_error(503, "120"),
                  self.http_error(429), self.http_error(503)]
        with patch.object(builder.fetcher.OPENER, "open", side_effect=errors) as opened, patch.object(builder.fetcher.time, "sleep") as sleep:
            with self.assertRaisesRegex(RuntimeError, "after 3 retries"):
                builder.download({"url": "https://upload.wikimedia.org/a.jpg"})
            self.assertEqual(opened.call_count, 4)
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [1, 45, 1, 120, 1, 240, 1])
            requests = [call.args[0] for call in opened.call_args_list]
            self.assertTrue(all(request is requests[0] for request in requests))
            self.assertEqual(requests[0].full_url, "https://upload.wikimedia.org/a.jpg")

    def test_retry_accepts_http_dates_and_keeps_validation(self):
        future = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=120))
        response = self.response()
        with patch.object(builder.fetcher.OPENER, "open", side_effect=[self.http_error(429, future), response]), patch.object(builder.fetcher.time, "sleep") as sleep, patch.object(builder, "validate_jpeg") as validate:
            self.assertEqual(builder.download({"url": "https://upload.wikimedia.org/a.jpg"}), b"JPEG bytes")
            self.assertEqual([call.args[0] for call in sleep.call_args_list][::2], [1, 1])
            self.assertTrue(110 <= sleep.call_args_list[1].args[0] <= 120)
            response.read.assert_called_once_with(builder.MAX_IMAGE_BYTES + 1)
            validate.assert_called_once_with(b"JPEG bytes")
        response = self.response()
        response.headers.get_content_type.return_value = "text/html"
        with patch.object(builder.fetcher.OPENER, "open", side_effect=[self.http_error(503), response]) as opened, patch.object(builder.fetcher.time, "sleep"):
            with self.assertRaisesRegex(ValueError, "HTTP 200 image/jpeg"):
                builder.download({"url": "https://upload.wikimedia.org/a.jpg"})
            self.assertEqual(opened.call_count, 2)
        for error in (self.http_error(404), ValueError("redirect refused")):
            with patch.object(builder.fetcher.OPENER, "open", side_effect=error) as opened, patch.object(builder.fetcher.time, "sleep"):
                with self.assertRaises(type(error)):
                    builder.download({"url": "https://upload.wikimedia.org/a.jpg"})
                self.assertEqual(opened.call_count, 1)

    def test_verify_rejects_eval_hash_duplicates_and_changed_bytes_offline(self):
        output = io.BytesIO()
        Image.new("RGB", (960, 1), "red").save(output, format="JPEG")
        data = output.getvalue()
        row = builder.image_row(self.page(), set())
        row["sha256"] = builder.hashlib.sha256(data).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)
            (path / "images").mkdir()
            image = path / "images/commons-123.jpg"
            image.write_bytes(data)
            manifest = {"seed": builder.SEED, "count": 1, "fit_count": 1,
                        "photos": builder.split_rows([row], 1)}
            builder.write_json(path / "manifest.json", manifest)
            with patch.object(builder, "download", side_effect=AssertionError("network")):
                builder.verify(path, [])
                with self.assertRaisesRegex(ValueError, "overlapping photo bytes"):
                    builder.verify(path, [{"id": "other", "title": "other", "sha256": row["sha256"]}])
                image.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError, "hash/size"):
                    builder.verify(path, [])


class CacheTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get("SEYK_CALIBRATION_EXPORTER"), "CI provides the model-free exporter")
    def test_same_size_changed_weights_invalidate_cached_vectors(self):
        exporter = str(Path(os.environ["SEYK_CALIBRATION_EXPORTER"]).resolve())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "model.onnx").write_bytes(b"synthetic ONNX stub; never loaded")
            weights = model / "model.onnx_data"
            weights.write_bytes(b"old-weight-bytes")
            size = weights.stat().st_size
            digest_map = root / "model-files.json"
            cache = root / "vectors"
            cache.mkdir()

            def key(hashes):
                builder.write_json(digest_map, hashes)
                return run.subprocess.run([exporter, "--cache-key", str(digest_map)],
                                          check=True, capture_output=True, text=True).stdout.strip()

            before = run.model_files(model)
            old_key = key(before)
            cached = cache / f"{old_key}.json"
            cached.write_text('{"synthetic": "old vector"}')
            self.assertEqual(key(run.model_files(model)), old_key)
            self.assertTrue(cached.exists())
            weights.write_bytes(b"new-weight-bytes")
            self.assertEqual(weights.stat().st_size, size)
            after = run.model_files(model)
            self.assertEqual(before["model.onnx"], after["model.onnx"])
            self.assertNotEqual(before["model.onnx_data"], after["model.onnx_data"])
            new_key = key(after)
            self.assertNotEqual(new_key, old_key)
            self.assertFalse((cache / f"{new_key}.json").exists())
            self.assertTrue(cached.exists())
            # The entire map is bound, including non-ONNX preprocessing assets.
            (model / "processor.json").write_text('{}')
            self.assertNotEqual(key(run.model_files(model)), new_key)
            self.assertEqual(key(dict(reversed(list(after.items())))), new_key)


class NumericTests(unittest.TestCase):
    def test_encoding_ties_even_and_high_nibble(self):
        anchor = np.zeros(8, dtype=np.float32)
        table = np.array([[0] * 8, [1] * 8], dtype=np.float32)
        image = np.array([[-1, .5, 1.5, 2.5, 3.5, 14.5, 15.5, 99]], dtype=np.float32)
        self.assertEqual(measure.encode(image, anchor, table).tolist(), [[0, 34, 78, 255]])
        table[1, 0] = 0
        with self.assertRaisesRegex(ValueError, "invalid quantization"):
            measure.encode(image, anchor, table)

    def test_rank_ties_and_rerank_excludes_approximate_tail(self):
        np.testing.assert_array_equal(measure.order(np.array([.1, .1, .2])), [2, 0, 1])
        np.testing.assert_array_equal(measure.rerank(np.array([.9, .8, .7]), np.array([.1, .2, .99]), 2), [1, 0])
        self.assertEqual(measure.spearman(np.array([1, 1, 2]), np.array([1, 1, 2])), 1)

    def test_fit_uses_only_A_and_M3_preserves_non_outliers(self):
        rng = np.random.default_rng(4714)
        images = rng.uniform(-.25, .25, (80, 8)).astype(np.float32)
        images[:, 0] = 2
        table = np.array([[-1] * 8, [2 / 15] * 8], dtype=np.float32)
        before = images.copy()
        result = measure.fit_tables(images, np.ones((10, 8), np.float32), np.zeros(8, np.float32), table)
        np.testing.assert_array_equal(images, before)
        np.testing.assert_array_equal(result["M3"][0][:, 1:], table[:, 1:])
        self.assertEqual(result["M3"][1]["widened_coordinates"], [0])
        self.assertGreater(result["M3"][0][1, 0], table[1, 0])
        for candidate, _ in result.values():
            self.assertTrue(np.isfinite(candidate).all())
            self.assertTrue((candidate[1] > 0).all())

    def test_original_python_goldens_unchanged(self):
        def vector(dim, seed):
            values = np.zeros(dim, np.float32)
            if seed == 3:
                values[0] = 1
            else:
                for i in range(256):
                    values[(i * 5 + seed * 11) % dim] = 1 / 16 if (i * 17 + seed * 13) % 7 < 3 else -1 / 16
            return values
        for row in json.loads((measure.ASSETS / "golden.json").read_text()):
            dim = row["dim"]
            anchor, table = measure.asset(dim)
            codes = measure.encode(vector(dim, row["seed"])[None, :], anchor, table)
            self.assertEqual(codes[0].tolist(), row["codes"])
            score = measure.lut_scores(vector(dim, 2), codes, anchor, table)[0]
            self.assertLess(abs(float(score) - row["score"]), 1e-5)

    @unittest.skipUnless(os.environ.get("SEYK_CALIBRATION_REFERENCE"), "CI provides model-free production Rust references")
    def test_production_scores_and_ranking_are_bit_identical(self):
        for row in json.loads(Path(os.environ["SEYK_CALIBRATION_REFERENCE"]).read_text()):
            images = np.asarray(row["images"], np.float32)
            anchor, table = measure.asset(row["dim"])
            codes = measure.encode(images, anchor, table)
            np.testing.assert_array_equal(codes, row["codes"])
            for query in row["queries"]:
                q = np.asarray(query["vector"], np.float32)
                for actual, expected in ((measure.lut_scores(q, codes, anchor, table), query["q4"]),
                                         (measure.dot_scores(q, images), query["f32"])):
                    expected = np.asarray(expected, np.float32)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    np.testing.assert_array_equal(measure.order(actual), measure.order(expected))

    @unittest.skipUnless(os.environ.get("SEYK_CALIBRATION_REFERENCE"), "CI provides model-free production Rust references")
    def test_complete_offline_measurement_and_stale_export_rejection(self):
        references = json.loads(Path(os.environ["SEYK_CALIBRATION_REFERENCE"]).read_text())
        images = []
        for i in range(65):
            images.append({"id": f"photo-{i}", "split": "fit" if i < 40 else "heldout" if i < 45 else "eval",
                           "vectors": {str(r["dim"]): r["images"][i] for r in references},
                           "codes": {str(r["dim"]): r["codes"][i] for r in references}})
        queries = []
        for split in ("fit", "eval"):
            for i in range(3):
                queries.append({"text": f"{split}-{i}", "split": split,
                                "vectors": {str(r["dim"]): r["queries"][i]["vector"] for r in references},
                                "scores": {str(r["dim"]): {name: r["queries"][i][name][40:] for name in ("q4", "f32")} for r in references}})
        job = {"calibration_manifest_sha256": "synthetic-only", "images": images,
               "queries": [dict(q, relevant={r["id"]: 1 for r in images[45:]}) for q in queries]}
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            builder.write_json(root / "job.json", job)
            export = {"job_blake3": builder.blake3.blake3((root / "job.json").read_bytes()).hexdigest(),
                      "images": images, "queries": queries, "identity": "synthetic-only"}
            builder.write_json(root / "export.json", export)
            evidence = measure.measure(root / "export.json", root / "job.json", root / "candidates", {})
            self.assertFalse(evidence["decision"]["merge_eligible"])
            self.assertEqual(len(list((root / "candidates").glob("*.bin"))), 8)
            for dim in (256, 768):
                self.assertEqual((root / "candidates" / f"quant4_{dim}.v1.f32.bin").read_bytes(),
                                 (measure.ASSETS / f"quant4_{dim}.f32.bin").read_bytes())
            corrupted = copy.deepcopy(export)
            corrupted["queries"][3]["scores"]["256"]["q4"][0] = 42
            builder.write_json(root / "export.json", corrupted)
            with self.assertRaisesRegex(ValueError, "score bits differ"):
                measure.measure(root / "export.json", root / "job.json", root / "candidates", {})
            (root / "job.json").write_text("{}")
            with self.assertRaisesRegex(ValueError, "another job"):
                measure.measure(root / "export.json", root / "job.json", root / "candidates", {})

    def test_stop_rules_and_M3_preference(self):
        def quality(recall, mae):
            return {"quality": {"recall": {f"k{k}_R{r}": recall for k in (1, 5, 10) for r in (k, 2*k, 50)},
                                "mae": mae, "identical_rerank": True, "maximum_ndcg_drop_rerank50": 0}}
        results = {str(d): {"v1": quality(.8, .1), "M1": quality(.825, .09),
                            "M2": quality(.82, .09), "M3": quality(.822, .09)} for d in (256, 768)}
        decision = measure.select(results)
        self.assertEqual(decision["candidate"], "M3")
        self.assertFalse(decision["keep_v1"])
        self.assertFalse(decision["merge_eligible"])
        results["256"]["M3"]["quality"]["recall"]["k1_R1"] = .79
        self.assertTrue(measure.select(results)["keep_v1"])

    def test_generic_queries_are_disjoint_and_model_free(self):
        queries = json.loads((builder.ROOT / "scripts/fixtures/photo-queries.json").read_text())
        self.assertEqual(len(run.generic_queries(queries)), 200)
        with self.assertRaisesRegex(ValueError, "disjoint"):
            run.generic_queries([{"text": run.generic_queries([])[0]["text"]}])


if __name__ == "__main__":
    unittest.main()
