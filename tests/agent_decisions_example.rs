#![cfg(feature = "cli")]
//! Keeps examples/agent-decisions working without model weights: an empty
//! HF_HOME makes search fall back to keywords and file names.

use assert_cmd::Command;
use serde_json::Value;
use std::path::Path;
use tempfile::TempDir;

fn binary(hf_home: &Path) -> Command {
    let mut cmd = Command::cargo_bin("seyk").unwrap();
    cmd.env("HF_HOME", hf_home).env("HF_HUB_OFFLINE", "1");
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

#[test]
fn decision_record_is_found_and_extracted_exactly() {
    let temp = TempDir::new().unwrap();
    let dir = temp.path().join("decisions");
    copy_fixture(&dir);

    let json = search(temp.path(), "why did we pick SQLite over Postgres", &dir);
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
            .ends_with("/0007-storage-sqlite.md"),
        "{hit}"
    );

    let extracted = binary(temp.path())
        .args(["extract", "--from-json", "-"])
        .write_stdin(json)
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let source = std::fs::read(dir.join("0007-storage-sqlite.md")).unwrap();
    let start = hit["locator"]["byte_start"].as_u64().unwrap() as usize;
    let end = hit["locator"]["byte_end"].as_u64().unwrap() as usize;
    assert_eq!(extracted, &source[start..end]);
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
