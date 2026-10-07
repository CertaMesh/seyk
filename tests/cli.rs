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
        .args(["mcp", temp.path().to_str().unwrap()])
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
