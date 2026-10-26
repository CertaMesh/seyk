#!/usr/bin/env python3
"""Run the calibration experiment after download and inference authorization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from build_set import EVAL, MAX_IMAGE_BYTES, ROOT, build, write_json
from measure import measure

SUBJECTS = ("a bicycle", "a wooden table", "a crowded market", "a kitchen", "a railway station",
            "a flowering tree", "a sailing boat", "a stone bridge", "a mountain trail", "a sandy beach",
            "a garden", "a library", "a farm", "a sculpture", "a lake", "a city square",
            "a bird", "a cat", "a bus", "a waterfall")
SETTINGS = ("in morning light", "under an overcast sky", "seen from above", "with people nearby",
            "in the distance", "with detailed textures", "in summer", "at sunset",
            "in a wide view", "in a close view")


def generic_queries(eval_queries):
    texts = [f"A photo of {subject} {setting}." for subject in SUBJECTS for setting in SETTINGS]
    if len(set(texts)) != 200 or set(texts) & {q["text"] for q in eval_queries}:
        raise ValueError("generic captions must be unique and disjoint from eval queries")
    return [{"text": text, "split": "fit"} for text in texts]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while block := file.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def model_files(directory):
    paths = sorted(path for path in directory.rglob("*") if path.is_file())
    if not paths or not any(path.suffix == ".onnx" for path in paths):
        raise ValueError("--model must contain already-provisioned q8 ONNX weights")
    return {str(path.relative_to(directory)): sha256_file(path) for path in paths}


def prepare_job(directory, fixture, manifest, model_hashes):
    eval_photos = json.loads(EVAL.read_text())
    eval_queries = json.loads((ROOT / "scripts/fixtures/photo-queries.json").read_text())
    return {"calibration_manifest_sha256": sha256_file(directory / "manifest.json"),
            "eval_manifest_sha256": sha256_file(EVAL), "model_files_sha256": model_hashes,
            "images": [{"id": row["id"], "path": str((directory / "images" / f"{row['id']}.jpg").resolve()),
                        "split": row["split"], "sha256": row["sha256"]} for row in manifest["photos"]] +
                       [{"id": row["id"], "path": str((fixture / f"{row['id']}.jpg").resolve()),
                         "split": "eval", "sha256": row["sha256"]} for row in eval_photos],
            "queries": generic_queries(eval_queries) + [dict(q, split="eval") for q in eval_queries]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true", help="download calibration images and run inference")
    parser.add_argument("--directory", type=Path, default=Path.home() / ".cache/seyk-bench/photo-calibration")
    parser.add_argument("--fixture", type=Path, default=Path(os.environ.get("SEYK_PHOTO_FIXTURE", str(Path.home() / ".cache/seyk-bench/photo-fixture"))))
    parser.add_argument("--count", type=int, default=2000)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--exporter", type=Path, default=ROOT / "target/release/examples/photo_calibration_export")
    parser.add_argument("--approved-download-gb", type=float)
    parser.add_argument("--score-only", action="store_true", help="recompute from verified exported vectors without loading a model")
    args = parser.parse_args()
    if args.count < 1000:
        parser.error("at least 1,000 calibration photos are required")
    print(json.dumps({"download_upper_bound_gb": args.count * MAX_IMAGE_BYTES / 1e9,
                      "approval_required": args.count * MAX_IMAGE_BYTES > 5e9,
                      "eval_download": False, "model_download": False}), flush=True)
    if not args.execute and not args.score_only:
        return
    directory, fixture = args.directory.expanduser().resolve(), args.fixture.expanduser().resolve()
    if directory.is_relative_to(ROOT):
        parser.error("store photos, vectors and candidate tables outside the repository")
    if fixture == directory:
        parser.error("calibration and eval destinations must be separate")
    if args.score_only:
        from build_set import verify
        verify(directory)
        job = directory / "job.json"
        previous = json.loads(job.read_text())
        manifest = json.loads((directory / "manifest.json").read_text())
        current = prepare_job(directory, fixture, manifest, previous["model_files_sha256"])
        if previous != current:
            raise ValueError("experiment inputs changed; regenerate export")
        for image in current["images"]:
            if sha256_file(Path(image["path"])) != image["sha256"]:
                raise ValueError("experiment image changed")
        hashes = previous["model_files_sha256"]
    else:
        if args.model is None or not args.exporter.is_file():
            parser.error("provide --model and a prebuilt --exporter; this command never builds or downloads models")
        env = dict(os.environ, SEYK_PHOTO_FIXTURE=str(fixture))
        subprocess.run([sys.executable, str(ROOT / "scripts/fetch_photo_fixture.py"), "--verify"],
                       cwd=ROOT, env=env, check=True)
        hashes = model_files(args.model.expanduser().resolve())
        manifest = build(directory, args.count, args.approved_download_gb)
        job = directory / "job.json"
        write_json(job, prepare_job(directory, fixture, manifest, hashes))
        subprocess.run([str(args.exporter.resolve()), str(job), str(directory / "export.json"),
                        str(directory / "vectors"), str(args.model.expanduser().resolve())], cwd=ROOT, check=True)
    evidence = measure(directory / "export.json", job, directory / "candidates", hashes)
    print(json.dumps(evidence["decision"], indent=2))
    print(f"Evidence: {directory / 'candidates/evidence.json'}")


if __name__ == "__main__":
    main()
