mod common;

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use tempfile::TempDir;

fn binary() -> Command {
    Command::cargo_bin("seyk").unwrap()
}

#[test]
fn help_lists_all_top_level_commands() {
    binary().arg("--help").assert().success().stdout(
        predicate::str::contains("search")
            .and(predicate::str::contains("index"))
            .and(predicate::str::contains("extract"))
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
    assert_eq!(stats["schema_version"], 2);
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

/// The MCP server answers the handshake and lists its tool before any model is
/// loaded, so clients start fast and this runs without weights.
#[cfg(feature = "mcp")]
#[test]
fn mcp_server_lists_search_tool_without_loading_the_model() {
    let temp = TempDir::new().unwrap();
    let requests = [
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
    ]
    .join("\n")
        + "\n";
    let output = binary()
        .args(["mcp", "--standalone", temp.path().to_str().unwrap()])
        .env("SEYK_MODEL", temp.path().join("no-model"))
        .write_stdin(requests)
        .timeout(std::time::Duration::from_secs(20))
        .output()
        .unwrap();
    let responses: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(responses[0]["result"]["serverInfo"]["name"], "seyk");
    let tools = &responses[1]["result"]["tools"];
    assert_eq!(tools[0]["name"], "search");
    assert!(
        tools[0]["inputSchema"]["required"]
            .as_array()
            .unwrap()
            .contains(&Value::from("queries"))
    );
}

/// `seyk mcp` starts the shared daemon on first use and relays through it; the
/// daemon reports its folders and stops on request. No model weights needed.
#[cfg(all(feature = "mcp", unix))]
#[test]
fn mcp_relays_through_a_daemon_it_starts() {
    let temp = TempDir::new().unwrap();
    let notes = temp.path().join("notes");
    std::fs::create_dir(&notes).unwrap();
    let env = |cmd: &mut Command| {
        cmd.env("SEYK_DAEMON_DIR", temp.path())
            .env("SEYK_MODEL", temp.path().join("no-model"))
            .env("SEYK_CACHE", temp.path().join("cache.sqlite3"));
    };
    let requests = [
        r#"{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"test","version":"0"}}}"#,
        r#"{"jsonrpc":"2.0","method":"notifications/initialized"}"#,
        r#"{"jsonrpc":"2.0","id":2,"method":"tools/list"}"#,
    ]
    .join("\n")
        + "\n";
    let mut relay = binary();
    env(&mut relay);
    let output = relay
        .args(["mcp", notes.to_str().unwrap()])
        .write_stdin(requests)
        .timeout(std::time::Duration::from_secs(30))
        .output()
        .unwrap();
    let responses: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(responses[0]["result"]["serverInfo"]["name"], "seyk");
    assert_eq!(responses[1]["result"]["tools"][0]["name"], "search");

    let mut status = binary();
    env(&mut status);
    let status: Value = serde_json::from_slice(
        &status
            .args(["daemon", "status", "--json"])
            .output()
            .unwrap()
            .stdout,
    )
    .unwrap();
    let folder = &status["folders"][0];
    assert!(folder[0].as_str().unwrap().ends_with("notes"));
    // Without weights the watcher cannot start; that must show, not hang.
    assert_eq!(folder[1]["state"], "stopped");

    let mut stop = binary();
    env(&mut stop);
    stop.args(["daemon", "stop"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Stopped."));
}

#[test]
fn extract_round_trip_without_a_model() {
    use seyk::{Dimension, ResourceLocator, SearchEngine};
    let temp = TempDir::new().unwrap();
    let path = temp.path().join("café note.md");
    let text = "# Alpha\r\n\r\nExact **café** [文書](https://example.invalid) 🙂\r\n";
    std::fs::write(&path, text).unwrap();
    let engine = SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(Dimension::D128, "cli-extract"))
        .build()
        .unwrap();
    let results = engine
        .search("alpha")
        .paths([&path])
        .previews(None)
        .execute()
        .unwrap();
    let mut value = serde_json::to_value(&results).unwrap();
    value["schema_version"] = 2.into();
    assert!(value["hits"][0].get("preview").unwrap().is_null());
    let input = serde_json::to_string(&value).unwrap();
    let ResourceLocator::Text {
        byte_start,
        byte_end,
        ..
    } = results.hits[0].locator
    else {
        panic!("text locator")
    };
    let expected = &text.as_bytes()[byte_start as usize..byte_end as usize];
    let run = |args: &[&str], data: &str| {
        binary()
            .args(args)
            .env("SEYK_MODEL", temp.path().join("no-model"))
            .env("HF_HUB_OFFLINE", "1")
            .write_stdin(data)
            .output()
            .unwrap()
    };
    let raw = run(&["extract", "--from-json", "-"], &input);
    assert!(
        raw.status.success(),
        "{}",
        String::from_utf8_lossy(&raw.stderr)
    );
    assert_eq!(raw.stdout, expected);
    let json = run(&["extract", "--from-json", "-", "--json"], &input);
    assert!(json.status.success());
    let extracted: Value = serde_json::from_slice(&json.stdout).unwrap();
    assert_eq!(extracted["schema_version"], 2);
    assert_eq!(extracted["text"].as_str().unwrap().as_bytes(), expected);
    assert_eq!(extracted["source"], value["hits"][0]["resource"]);
    assert_eq!(extracted["locator"], value["hits"][0]["locator"]);
    assert_eq!(extracted["metadata"], value["hits"][0]["metadata"]);
    let input_path = temp.path().join("hits.json");
    std::fs::write(&input_path, &input).unwrap();
    binary()
        .args(["extract", "--from-json"])
        .arg(&input_path)
        .assert()
        .success()
        .stdout(expected);
    let missing = run(&["extract", "--from-json", "-", "--hit", "999"], &input);
    assert!(!missing.status.success());
    assert!(missing.stdout.is_empty());
    std::fs::write(&path, text.replace("Exact", "Other")).unwrap();
    let stale = run(&["extract", "--from-json", "-", "--json"], &input);
    assert!(!stale.status.success());
    assert!(stale.stdout.is_empty());
    assert!(String::from_utf8_lossy(&stale.stderr).contains("resource changed"));
    for data in [
        "not json".to_string(),
        "{}".into(),
        input.replace("\"schema_version\":2", "\"schema_version\":3"),
    ] {
        let bad = run(&["extract", "--from-json", "-"], &data);
        assert!(!bad.status.success());
        assert!(bad.stdout.is_empty());
    }
    value["hits"] = serde_json::json!([]);
    let empty = run(&["extract", "--from-json", "-"], &value.to_string());
    assert!(!empty.status.success());
    assert!(String::from_utf8_lossy(&empty.stderr).contains("no hit at index 0"));
}
