#![cfg(all(feature = "cli", feature = "pdf"))]
mod common;
#[path = "../crates/seyk-core/tests/support/pdf_fixture.rs"]
mod fixture;
use fixture::{Kind, pdf};
use seyk::{Dimension, SearchEngine};
use std::process::{Command, Output};

fn no_leak(output: &Output) {
    for bytes in [&output.stdout, &output.stderr] {
        let s = String::from_utf8_lossy(bytes);
        assert!(
            !s.contains(fixture::MARKER),
            "unexpected document marker in CLI output"
        );
        assert!(!s.contains("Operation {"), "panic payload escaped");
    }
}

#[test]
fn real_binary_extract_contains_panics_and_preserves_public_text() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.pdf");
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "pdf"))
        .build()
        .unwrap();
    for kind in [
        Kind::Plain,
        Kind::Tint,
        Kind::InvalidUtf8,
        Kind::Panic,
        Kind::MissingContents,
        Kind::MissingArrayContents,
        Kind::TruncatedContents,
        Kind::BadFlate,
    ] {
        std::fs::write(&path, pdf(&[fixture::PUBLIC], Kind::Plain)).unwrap();
        let mut result = engine.search("basil").paths([&path]).execute().unwrap();
        let bytes = pdf(&[fixture::PUBLIC], kind);
        std::fs::write(&path, &bytes).unwrap();
        result.hits[0].metadata.content_hash = blake3::hash(&bytes).to_hex().to_string();
        result.hits[0].metadata.size_bytes = bytes.len() as u64;
        let mut envelope = serde_json::to_value(result).unwrap();
        envelope["schema_version"] = 3.into();
        let json = dir.path().join("hit.json");
        std::fs::write(&json, serde_json::to_vec(&envelope).unwrap()).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_seyk"))
            .args(["extract", "--from-json"])
            .arg(&json)
            .output()
            .unwrap();
        no_leak(&output);
        assert!(!String::from_utf8_lossy(&output.stderr).contains("internal failure"));
        if matches!(kind, Kind::Plain | Kind::Tint) {
            assert!(
                output.status.success(),
                "{}",
                String::from_utf8_lossy(&output.stderr)
            );
            assert_eq!(output.stdout, fixture::PUBLIC.as_bytes());
        } else {
            assert!(!output.status.success());
            assert!(output.stdout.is_empty());
            let reason = String::from_utf8_lossy(&output.stderr);
            assert!(reason.contains("PDF parse") || reason.contains("PDF parser"));
        }
    }
}

#[test]
#[ignore = "requires local Gemma weights; no downloads"]
fn real_binary_index_preview_and_extract_never_leak_hidden_markers() {
    let dir = tempfile::tempdir().unwrap();
    let cache = dir.path().join("cache.db");
    for kind in [
        Kind::Plain,
        Kind::Tint,
        Kind::InvalidUtf8,
        Kind::Panic,
        Kind::MissingContents,
        Kind::MissingArrayContents,
        Kind::TruncatedContents,
        Kind::BadFlate,
    ] {
        let path = dir.path().join("fixture.pdf");
        std::fs::write(&path, pdf(&[fixture::PUBLIC], kind)).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_seyk"))
            .arg("index")
            .arg(&path)
            .args(["--dimension", "128", "--text-only", "--cache-path"])
            .arg(&cache)
            .arg("--json")
            .output()
            .unwrap();
        no_leak(&output);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        if !matches!(kind, Kind::Plain | Kind::Tint) {
            assert_eq!(report["sync"]["skipped"].as_array().unwrap().len(), 1);
            continue;
        }
        let output = Command::new(env!("CARGO_BIN_EXE_seyk"))
            .args(["search", "basil"])
            .arg(&path)
            .args(["--dimension", "128", "--text-only", "--cache-path"])
            .arg(&cache)
            .arg("--json")
            .output()
            .unwrap();
        no_leak(&output);
        assert!(output.status.success());
        let json = dir.path().join("hits.json");
        std::fs::write(&json, &output.stdout).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_seyk"))
            .args(["extract", "--from-json"])
            .arg(json)
            .output()
            .unwrap();
        no_leak(&output);
        assert!(output.status.success());
        assert_eq!(output.stdout, fixture::PUBLIC.as_bytes());
    }
}

#[test]
fn real_binary_page_extract_returns_explicit_ocr_unavailable_without_a_model() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.pdf");
    std::fs::write(&path, pdf(&[fixture::PUBLIC], Kind::Plain)).unwrap();
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "pdf"))
        .build()
        .unwrap();
    let mut result = engine.search("basil").paths([&path]).execute().unwrap();
    result.hits[0].locator = seyk::ResourceLocator::Page { page: 1 };
    result.hits[0].preview = None;
    let mut envelope = serde_json::to_value(result).unwrap();
    envelope["schema_version"] = 3.into();
    let json = dir.path().join("hit.json");
    std::fs::write(&json, serde_json::to_vec(&envelope).unwrap()).unwrap();
    let output = Command::new(env!("CARGO_BIN_EXE_seyk"))
        .args(["extract", "--from-json"])
        .arg(json)
        .env("HF_HOME", dir.path().join("no-models"))
        .output()
        .unwrap();
    assert!(!output.status.success());
    assert!(output.stdout.is_empty());
    assert!(
        String::from_utf8_lossy(&output.stderr)
            .contains("readable text for scanned pages needs OCR, which is not available yet")
    );
    no_leak(&output);
}

#[cfg(feature = "pdf-render")]
#[path = "../crates/seyk-core/tests/support/scan_fixture.rs"]
mod scan_fixture;

#[cfg(feature = "pdf-render")]
#[test]
#[ignore = "requires local Gemma vision weights; no downloads"]
fn real_binary_scan_render_path_never_leaks_hidden_markers() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("scan.pdf");
    for (bytes, diagnostic) in [
        (scan_fixture::broken_image(fixture::MARKER), true),
        (pdf(&[""], Kind::Tint), false),
    ] {
        std::fs::write(&path, bytes).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_seyk"))
            .arg("index")
            .arg(&path)
            .args(["--dimension", "256", "--cache-path"])
            .arg(dir.path().join("cache.db"))
            .arg("--json")
            .output()
            .unwrap();
        no_leak(&output);
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let report: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(report["sync"]["skipped"].as_array().unwrap().is_empty());
        if diagnostic {
            assert_eq!(
                report["sync"]["page_diagnostics"][0]["reason"],
                "scanned page could not be rendered"
            );
            assert_eq!(report["sync"]["segments_embedded"], 0);
        } else {
            assert_eq!(report["sync"]["segments_embedded"], 1);
        }
    }
}

#[test]
fn real_binary_cmap_panics_have_no_hook_stderr_spam() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("fixture.pdf");
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "cmap-cli"))
        .build()
        .unwrap();
    for bad_length in [false, true] {
        std::fs::write(&path, pdf(&[fixture::PUBLIC], Kind::Plain)).unwrap();
        let mut result = engine.search("basil").paths([&path]).execute().unwrap();
        let bytes = fixture::malformed_cmap(bad_length);
        std::fs::write(&path, &bytes).unwrap();
        result.hits[0].metadata.content_hash = blake3::hash(&bytes).to_hex().to_string();
        result.hits[0].metadata.size_bytes = bytes.len() as u64;
        let mut envelope = serde_json::to_value(result).unwrap();
        envelope["schema_version"] = 6.into();
        let json = dir.path().join("hit.json");
        std::fs::write(&json, serde_json::to_vec(&envelope).unwrap()).unwrap();
        let output = Command::new(env!("CARGO_BIN_EXE_seyk"))
            .args(["extract", "--from-json"])
            .arg(&json)
            .output()
            .unwrap();
        no_leak(&output);
        assert!(!output.status.success());
        assert!(output.stdout.is_empty());
        let stderr = String::from_utf8_lossy(&output.stderr);
        assert!(stderr.contains("PDF parser panicked"));
        assert!(!stderr.contains("internal failure"));
        assert!(!stderr.contains("panicked at"));
        assert!(!stderr.contains("hexstring"));
        assert!(!stderr.contains("unicode map"));
    }
}
