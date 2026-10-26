#![cfg(feature = "cli")]
use serde_json::{Value, json};
use std::path::Path;
use std::process::{Command, Output};

const SENTINEL: &str = "SYNTHETIC_PRIVATE_SENTINEL";

fn envelope(path: &Path, bytes: &[u8]) -> Value {
    let results = seyk::SearchResults {
        degraded: None,
        kind_counts: None,
        queries: vec![SENTINEL.into()],
        mode: seyk::SearchMode::Semantic,
        stats: Default::default(),
        hits: vec![seyk::SearchHit {
            resource: seyk::ResourceId::from_path(path.canonicalize().unwrap()).unwrap(),
            locator: seyk::ResourceLocator::Text {
                byte_start: 0,
                byte_end: bytes.len() as u64,
                line_start: Some(1),
                line_end: Some(1),
            },
            metadata: seyk::ResourceMetadata {
                media_type: Some("text/plain".into()),
                modality: seyk::Modality::Text,
                size_bytes: bytes.len() as u64,
                modified_ns: None,
                content_hash: blake3::hash(bytes).to_hex().to_string(),
            },
            score: 1.0,
            signals: Default::default(),
            r#match: Default::default(),
            heading: None,
            preview: None,
            query_index: 0,
        }],
    };
    let mut value = serde_json::to_value(results).unwrap();
    value["schema_version"] = 10.into();
    value
}

fn run(input: &Path, args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_seyk"))
        .args(["extract", "--from-json"])
        .arg(input)
        .args(args)
        .env("HF_HUB_OFFLINE", "1")
        .output()
        .unwrap()
}

fn denied(out: Output, exit: i32, code: &str) {
    assert_eq!(out.status.code(), Some(exit), "{out:?}");
    assert!(out.stdout.is_empty(), "{out:?}");
    assert_eq!(out.stderr, format!("error: {code}\n").as_bytes());
}

#[test]
fn typed_input_failures_are_one_static_line() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("public.txt");
    std::fs::write(&source, "public").unwrap();
    let valid = envelope(&source, b"public");
    let input = dir.path().join("hits.json");
    for schema in [
        Value::Null,
        json!(1),
        json!(10),
        json!("9"),
        json!(9.5),
        json!(true),
    ] {
        let mut value = valid.clone();
        value["schema_version"] = schema;
        std::fs::write(&input, value.to_string()).unwrap();
        denied(run(&input, &["--typed-errors"]), 3, "invalid_input");
    }
    for data in [
        format!("{{{SENTINEL}"),
        json!({"schema_version":10,"hits":SENTINEL}).to_string(),
    ] {
        std::fs::write(&input, data).unwrap();
        denied(
            run(&input, &["--typed-errors", "--json"]),
            3,
            "invalid_input",
        );
    }
    let mut unversioned = valid.clone();
    unversioned
        .as_object_mut()
        .unwrap()
        .remove("schema_version");
    std::fs::write(&input, unversioned.to_string()).unwrap();
    denied(run(&input, &["--typed-errors"]), 3, "invalid_input");
    let mut malformed_hit = valid.clone();
    malformed_hit["hits"][0]["locator"]["byte_start"] = SENTINEL.into();
    std::fs::write(&input, malformed_hit.to_string()).unwrap();
    std::fs::remove_file(&source).unwrap();
    denied(run(&input, &["--typed-errors"]), 3, "invalid_input");
    std::fs::write(&input, valid.to_string()).unwrap();
    denied(
        run(&input, &["--typed-errors", "--hit", "1"]),
        3,
        "invalid_input",
    );
}

#[test]
fn success_is_identical_and_legacy_errors_stay_prose() {
    let dir = tempfile::tempdir().unwrap();
    let bytes = "Exact café 🙂\r\n".as_bytes();
    let source = dir.path().join("public.txt");
    std::fs::write(&source, bytes).unwrap();
    let input = dir.path().join("hits.json");
    let mut value = envelope(&source, bytes);
    for version in 2..=10 {
        value["schema_version"] = version.into();
        std::fs::write(&input, value.to_string()).unwrap();
        for args in [vec![], vec!["--json"]] {
            let legacy = run(&input, &args);
            let mut typed_args = args;
            typed_args.push("--typed-errors");
            let typed = run(&input, &typed_args);
            assert_eq!(legacy.status.code(), Some(0));
            assert_eq!(typed.status.code(), Some(0));
            assert_eq!(legacy.stdout, typed.stdout);
            assert!(legacy.stderr.is_empty() && typed.stderr.is_empty());
            if typed_args.contains(&"--json") {
                let extracted: Value = serde_json::from_slice(&typed.stdout).unwrap();
                assert_eq!(extracted["schema_version"], 10);
                assert_eq!(extracted["text"].as_str().unwrap().as_bytes(), bytes);
            } else {
                assert_eq!(typed.stdout, bytes);
            }
        }
    }
    let out = run(&input, &["--hit", "99"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    assert_eq!(out.stderr, b"error: no hit at index 99\n");
    std::fs::write(&source, b"changed").unwrap();
    let out = run(&input, &[]);
    assert_eq!(out.status.code(), Some(1));
    assert!(out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("resource changed since search:"));
}

#[test]
fn generated_evidence_failures_are_typed() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("public.txt");
    let input = dir.path().join("hits.json");
    std::fs::write(&source, b"public").unwrap();
    let valid = envelope(&source, b"public");
    let check = |value: &Value, exit, code| {
        std::fs::write(&input, value.to_string()).unwrap();
        for args in [vec!["--typed-errors"], vec!["--typed-errors", "--json"]] {
            denied(run(&input, &args), exit, code);
        }
    };
    std::fs::write(&source, b"changed").unwrap();
    check(&valid, 4, "resource_changed");
    std::fs::write(&source, b"public").unwrap();
    let mut value = valid.clone();
    value["hits"][0]["locator"] = json!({"kind":"whole_resource"});
    check(&value, 5, "unsupported_extraction");
    value = valid.clone();
    value["hits"][0]["locator"]["byte_end"] = 999.into();
    check(&value, 6, "invalid_evidence");
    std::fs::write(&source, [0xff, 0xfe]).unwrap();
    check(&envelope(&source, &[0xff, 0xfe]), 6, "invalid_evidence");
    std::fs::remove_file(&source).unwrap();
    check(&valid, 7, "io_failed");
    denied(
        run(&dir.path().join("missing.json"), &["--typed-errors"]),
        7,
        "io_failed",
    );
    let secret = dir.path().join(".env");
    std::fs::write(&secret, b"public").unwrap();
    check(&envelope(&secret, b"public"), 8, "policy_denied");
}

#[test]
fn clap_still_owns_usage_help_and_version() {
    for args in [
        vec!["extract", "--typed-errors"],
        vec![
            "extract",
            "--typed-errors",
            "--from-json",
            "-",
            "--hit",
            "bad",
        ],
        vec![
            "extract",
            "--typed-errors",
            "--from-json",
            "-",
            "--hit",
            "-1",
        ],
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_seyk"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(2));
        assert!(out.stdout.is_empty());
        assert!(String::from_utf8_lossy(&out.stderr).contains("error:"));
    }
    for args in [
        vec!["extract", "--typed-errors", "--help"],
        vec!["--version"],
    ] {
        let out = Command::new(env!("CARGO_BIN_EXE_seyk"))
            .args(args)
            .output()
            .unwrap();
        assert_eq!(out.status.code(), Some(0));
        assert!(!out.stdout.is_empty());
        assert!(out.stderr.is_empty());
    }
}

#[cfg(unix)]
#[test]
fn closed_stdout_is_io_failure_for_raw_and_json() {
    use std::os::fd::OwnedFd;
    use std::os::unix::net::UnixStream;
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("public.txt");
    std::fs::write(&source, b"public").unwrap();
    let input = dir.path().join("hits.json");
    std::fs::write(&input, envelope(&source, b"public").to_string()).unwrap();
    for json in [false, true] {
        let (writer, reader) = UnixStream::pair().unwrap();
        drop(reader);
        let mut command = Command::new(env!("CARGO_BIN_EXE_seyk"));
        command
            .args(["extract", "--typed-errors", "--from-json"])
            .arg(&input)
            .stdout(std::process::Stdio::from(OwnedFd::from(writer)));
        if json {
            command.arg("--json");
        }
        denied(command.output().unwrap(), 7, "io_failed");
    }
}

#[cfg(not(feature = "audio-stt"))]
#[test]
fn audio_without_transcriber_is_unsupported() {
    let dir = tempfile::tempdir().unwrap();
    let source = dir.path().join("public.wav");
    // No audio feature dispatches directly to UnsupportedExtraction. With the
    // audio feature enabled, verify and decode a generated valid WAV first.
    let mut bytes = Vec::new();
    bytes.extend(b"RIFF");
    bytes.extend(32036u32.to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(16u32.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(16000u32.to_le_bytes());
    bytes.extend(32000u32.to_le_bytes());
    bytes.extend(2u16.to_le_bytes());
    bytes.extend(16u16.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend(32000u32.to_le_bytes());
    bytes.resize(32044, 0);
    std::fs::write(&source, &bytes).unwrap();
    let mut value = envelope(&source, &bytes);
    value["hits"][0]["metadata"]["modality"] = "audio".into();
    value["hits"][0]["metadata"]["media_type"] = "audio/wav".into();
    value["hits"][0]["locator"] = json!({"kind":"time", "start_ms":0, "end_ms":1000});
    let input = dir.path().join("hits.json");
    std::fs::write(&input, value.to_string()).unwrap();
    denied(
        run(&input, &["--typed-errors"]),
        5,
        "unsupported_extraction",
    );
}
