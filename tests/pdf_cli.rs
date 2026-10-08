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
    for kind in [Kind::Plain, Kind::Tint, Kind::InvalidUtf8, Kind::Panic] {
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
    for kind in [Kind::Plain, Kind::Tint, Kind::InvalidUtf8, Kind::Panic] {
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
        if matches!(kind, Kind::InvalidUtf8 | Kind::Panic) {
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
