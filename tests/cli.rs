use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use tempfile::TempDir;

fn binary() -> Command {
    Command::cargo_bin("gemma-search").unwrap()
}

#[test]
fn help_lists_all_top_level_commands() {
    binary().arg("--help").assert().success().stdout(
        predicate::str::contains("search")
            .and(predicate::str::contains("index"))
            .and(predicate::str::contains("cache"))
            .and(predicate::str::contains("model")),
    );
}

#[test]
fn cache_stats_json_and_clear_work_without_a_model() {
    let temp = TempDir::new().unwrap();
    let cache = temp.path().join("cache.sqlite3");
    let output = binary()
        .args([
            "cache",
            "--cache-path",
            cache.to_str().unwrap(),
            "stats",
            "--json",
        ])
        .output()
        .unwrap();
    assert!(output.status.success());
    let stats: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(stats["profiles"], serde_json::json!([]));

    binary()
        .args(["cache", "--cache-path", cache.to_str().unwrap(), "clear"])
        .assert()
        .success();
}

#[test]
fn missing_model_is_a_helpful_failure_and_invalid_dimensions_are_rejected() {
    let temp = TempDir::new().unwrap();
    binary()
        .env("HF_HOME", temp.path())
        .env("HF_HUB_OFFLINE", "1")
        .args(["search", "alpha", "--no-cache"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("model").and(predicate::str::contains("download")));

    binary()
        .args(["search", "alpha", "--dimension", "300", "--no-cache"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unsupported dimension 300"));
}
