#![cfg(feature = "cli")]
//! Keeps examples/agent-decisions working without model weights: an empty
//! HF_HOME makes search fall back to keywords and file names.

use assert_cmd::Command;
use serde_json::Value;
use std::path::Path;
use tempfile::TempDir;

fn binary(hf_home: &Path) -> Command {
    let mut cmd = Command::cargo_bin("seyk").unwrap();
    // A local SEYK_MODEL directory would bypass the empty HF_HOME.
    cmd.env_remove("SEYK_MODEL")
        .env("HF_HOME", hf_home)
        .env("HF_HUB_OFFLINE", "1");
    cmd
}

fn copy_fixture(to: &Path) {
    let from = Path::new(env!("CARGO_MANIFEST_DIR")).join("examples/agent-decisions/decisions");
    std::fs::create_dir_all(to).unwrap();
    for entry in std::fs::read_dir(from).unwrap() {
        let path = entry.unwrap().path();
        std::fs::copy(&path, to.join(path.file_name().unwrap())).unwrap();
    }
}

fn search(hf_home: &Path, query: &str, dir: &Path) -> Vec<u8> {
    binary(hf_home)
        .args(["search", query, "--no-cache", "--json", "-n", "3"])
        .arg(dir)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone()
}

/// Extracts the top hit and checks it is exactly the expected source lines.
fn assert_top_hit_extracts(query: &str, file: &str, lines: (u64, u64), passage: &str) {
    let temp = TempDir::new().unwrap();
    let dir = temp.path().join("decisions");
    copy_fixture(&dir);

    let json = search(temp.path(), query, &dir);
    let response: Value = serde_json::from_slice(&json).unwrap();
    assert_eq!(
        response["degraded"],
        serde_json::json!({"mode":"lexical_only","reason":"no_model"})
    );
    let hit = &response["hits"][0];
    assert!(
        hit["resource"]
            .as_str()
            .unwrap()
            .ends_with(&format!("/{file}")),
        "{hit}"
    );
    assert_eq!(
        hit["locator"]["line_start"].as_u64(),
        Some(lines.0),
        "{hit}"
    );
    assert_eq!(hit["locator"]["line_end"].as_u64(), Some(lines.1), "{hit}");

    let extracted = binary(temp.path())
        .args(["extract", "--from-json", "-"])
        .write_stdin(json)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let source = std::fs::read(dir.join(file)).unwrap();
    let start = hit["locator"]["byte_start"].as_u64().unwrap() as usize;
    let end = hit["locator"]["byte_end"].as_u64().unwrap() as usize;
    assert_eq!(extracted, &source[start..end]);
    let extracted = String::from_utf8(extracted).unwrap();
    assert!(extracted.contains(passage), "{extracted}");
}

#[test]
fn decision_passage_is_found_and_extracted_exactly() {
    assert_top_hit_extracts(
        "store tickets in a single SQLite database file",
        "0007-storage-sqlite.md",
        (13, 18),
        "We store tickets in a single SQLite database file in write-ahead-log mode,",
    );
}

/// The walk-through in examples/agent-decisions/README.md.
#[test]
fn readme_walkthrough_finds_the_consequences() {
    assert_top_hit_extracts(
        "how do background jobs write to the database?",
        "0007-storage-sqlite.md",
        (20, 25),
        "so background jobs queue their writes through the main service",
    );
}

#[test]
fn changed_decision_record_is_refused() {
    let temp = TempDir::new().unwrap();
    let dir = temp.path().join("decisions");
    copy_fixture(&dir);

    let json = search(temp.path(), "why did we pick SQLite over Postgres", &dir);
    let path = dir.join("0007-storage-sqlite.md");
    let edited = std::fs::read_to_string(&path)
        .unwrap()
        .replace("Status: Accepted", "Status: Superseded");
    std::fs::write(&path, edited).unwrap();

    binary(temp.path())
        .args(["extract", "--from-json", "-", "--typed-errors"])
        .write_stdin(json)
        .assert()
        .code(4)
        .stderr("error: resource_changed\n");
}
