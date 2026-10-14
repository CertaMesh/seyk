#![cfg(feature = "cli")]

mod common;

use assert_cmd::Command;
use predicates::prelude::*;
use serde_json::Value;
use tempfile::TempDir;

fn binary() -> Command {
    // A freshly linked executable can spend tens of seconds in the OS loader
    // (notably macOS code-signature assessment) before main even runs. Verify
    // process readiness once, before starting any command's operation deadline.
    // Once also prevents parallel tests from competing on that first launch.
    static READY: std::sync::Once = std::sync::Once::new();
    READY.call_once(|| {
        let output = Command::cargo_bin("seyk")
            .unwrap()
            .arg("--version")
            .timeout(std::time::Duration::from_secs(120))
            .ok()
            .expect("CLI readiness probe (--version) must succeed within 120 s");
        assert_eq!(output.stdout, b"seyk 0.1.0\n");
    });
    Command::cargo_bin("seyk").unwrap()
}

#[test]
fn version_matches_workspace_package() {
    binary()
        .arg("--version")
        .assert()
        .success()
        .stdout("seyk 0.1.0\n");
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
    let cache = temp.path().join("dedicated/cache.sqlite3");
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
    assert_eq!(stats["schema_version"], 6);
    assert_eq!(stats["profiles"], serde_json::json!([]));
    assert_eq!(stats["in_synced_folder"], false);
    #[cfg(target_os = "macos")]
    assert_eq!(stats["backup_excluded"], true);
    #[cfg(not(target_os = "macos"))]
    assert!(stats["backup_excluded"].is_null());
    assert!(temp.path().join("dedicated/CACHEDIR.TAG").is_file());

    binary()
        .args(["cache", "--cache-path", cache.to_str().unwrap(), "clear"])
        .assert()
        .success();
}

#[test]
fn relative_cache_path_does_not_exclude_unrelated_siblings() {
    let temp = TempDir::new().unwrap();
    let sibling = temp.path().join("source.md");
    std::fs::write(&sibling, "important source").unwrap();
    #[cfg(unix)]
    let permissions = {
        use std::os::unix::fs::PermissionsExt;
        std::fs::metadata(temp.path()).unwrap().permissions().mode()
    };
    let output = binary()
        .current_dir(temp.path())
        .args(["cache", "--cache-path", "cache.sqlite3", "stats", "--json"])
        .output()
        .unwrap();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stats: Value = serde_json::from_slice(&output.stdout).unwrap();
    #[cfg(target_os = "macos")]
    assert_eq!(stats["backup_excluded"], false);
    #[cfg(not(target_os = "macos"))]
    assert!(stats["backup_excluded"].is_null());
    assert!(!temp.path().join("CACHEDIR.TAG").exists());
    assert_eq!(
        std::fs::read_to_string(sibling).unwrap(),
        "important source"
    );
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        assert_eq!(
            std::fs::metadata(temp.path()).unwrap().permissions().mode(),
            permissions
        );
    }
}

#[test]
fn cache_scrub_preserves_live_records_and_stats_json_shape_without_a_model() {
    let temp = TempDir::new().unwrap();
    let notes = temp.path().join("notes");
    std::fs::create_dir(&notes).unwrap();
    let file = notes.join("note.txt");
    std::fs::write(&file, "syntheticprivatequartzsentinel old").unwrap();
    let cache = temp.path().join("cache.sqlite3");
    let engine = seyk::SearchEngine::builder()
        .embedder(common::FakeEmbedder::new(
            seyk::Dimension::D128,
            "scrub-cli",
        ))
        .cache(seyk::SqliteCache::open(&cache).unwrap())
        .build()
        .unwrap();
    engine.index().paths([&notes]).execute().unwrap();
    std::fs::write(&file, "new ordinary synthetic passage").unwrap();
    engine.index().paths([&notes]).execute().unwrap();
    drop(engine);
    binary()
        .args(["cache", "--cache-path", cache.to_str().unwrap(), "stats"])
        .assert()
        .success()
        .stdout(predicate::str::contains("cache scrub pending"));
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
    let before: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert!(before["schema_version"].is_u64());
    assert!(before.get("scrub_pending").is_none());
    assert!(before.get("cleanup_pending").is_none());
    binary()
        .args(["cache", "--cache-path", cache.to_str().unwrap(), "scrub"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Scrubbed"));
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
    let after: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(before["schema_version"], after["schema_version"]);
    assert_eq!(
        before["profiles"], after["profiles"],
        "scrub preserves live cache records"
    );
    assert_eq!(
        before.as_object().unwrap().keys().collect::<Vec<_>>(),
        after.as_object().unwrap().keys().collect::<Vec<_>>()
    );
    use seyk::EmbeddingCache;
    assert!(
        !seyk::SqliteCache::open(&cache)
            .unwrap()
            .stats()
            .unwrap()
            .scrub_pending
    );
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

/// Exercise an MCP session as a client: initialize, await readiness, then list
/// tools. Defensively keep stdin open until the pending replies arrive.
#[cfg(feature = "mcp")]
fn mcp_handshake(command: &mut std::process::Command) -> [Value; 2] {
    use std::io::{BufRead, BufReader, Write};
    use std::process::{Child, Stdio};
    use std::time::{Duration, Instant};

    struct Session(Child);
    impl Drop for Session {
        fn drop(&mut self) {
            // Also reap on assertion failure or timeout.
            let _ = self.0.kill();
            let _ = self.0.wait();
        }
    }
    let deadline = Instant::now() + Duration::from_secs(20);
    let mut session = Session(
        command
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit())
            .spawn()
            .unwrap(),
    );
    let mut stdin = session.0.stdin.take().unwrap();
    let stdout = session.0.stdout.take().unwrap();
    let (send, receive) = std::sync::mpsc::channel();
    let reader = std::thread::spawn(move || {
        for line in BufReader::new(stdout).lines() {
            if send.send(line).is_err() {
                break;
            }
        }
    });
    let mut response = |id| {
        use std::sync::mpsc::RecvTimeoutError;
        let line = match receive.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
            Ok(line) => line.expect("reading MCP child stdout failed"),
            Err(RecvTimeoutError::Disconnected) => panic!(
                "MCP child exited or closed stdout before response {id}; status: {:?}",
                session
                    .0
                    .try_wait()
                    .expect("checking MCP child status failed")
            ),
            Err(RecvTimeoutError::Timeout) => {
                if let Some(status) = session
                    .0
                    .try_wait()
                    .expect("checking MCP child status failed")
                {
                    panic!("MCP child exited before response {id}: {status}");
                }
                panic!("MCP response {id} timed out within the 20 s operation deadline");
            }
        };
        let value: Value = serde_json::from_str(&line).unwrap();
        assert_eq!(value["id"], id);
        assert!(value.get("error").is_none(), "{value}");
        value
    };
    writeln!(stdin, r#"{{"jsonrpc":"2.0","id":1,"method":"initialize","params":{{"protocolVersion":"2025-06-18","capabilities":{{}},"clientInfo":{{"name":"test","version":"0"}}}}}}"#).unwrap();
    stdin.flush().unwrap();
    let initialized = response(1);
    writeln!(
        stdin,
        r#"{{"jsonrpc":"2.0","method":"notifications/initialized"}}"#
    )
    .unwrap();
    writeln!(stdin, r#"{{"jsonrpc":"2.0","id":2,"method":"tools/list"}}"#).unwrap();
    stdin.flush().unwrap();
    let tools = response(2);
    drop(stdin);
    // Both responses are received; close/reap the child without another
    // wall-clock assertion on OS process teardown.
    drop(session);
    reader.join().unwrap();
    [initialized, tools]
}

#[cfg(all(feature = "mcp", unix))]
#[test]
#[should_panic(expected = "MCP child exited")]
fn mcp_early_child_exit_has_a_distinct_diagnostic() {
    let mut command = std::process::Command::new("/bin/sh");
    // Consume initialize before exiting so the failure is the missing reply,
    // rather than a write to an already closed stdin pipe.
    command.args(["-c", "read -r initialize; exit 7"]);
    mcp_handshake(&mut command);
}

/// The MCP server answers the handshake and lists its tool before any model is
/// loaded, so clients start fast and this runs without weights.
#[cfg(feature = "mcp")]
#[test]
fn mcp_server_lists_search_tool_without_loading_the_model() {
    let temp = TempDir::new().unwrap();
    let mut command = std::process::Command::new(binary().get_program());
    command
        .args(["mcp", "--standalone", temp.path().to_str().unwrap()])
        .env("SEYK_MODEL", temp.path().join("no-model"));
    let responses = mcp_handshake(&mut command);
    assert_eq!(responses[0]["result"]["serverInfo"]["name"], "seyk");
    let tools = &responses[1]["result"]["tools"];
    let search = tools
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["name"] == "search")
        .unwrap();
    #[cfg(feature = "audio-stt")]
    assert!(
        tools
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["name"] == "extract")
    );
    assert!(
        search["inputSchema"]["required"]
            .as_array()
            .unwrap()
            .contains(&Value::from("queries"))
    );
}

#[cfg(all(feature = "mcp", unix))]
#[test]
fn local_content_advertises_fixed_policy_without_daemon_or_model() {
    let temp = tempfile::Builder::new()
        .prefix("local-content-cli-")
        .tempdir()
        .unwrap();
    let root = temp.path().canonicalize().unwrap();
    let daemon = root.join("daemon-not-used");
    let mut command = std::process::Command::new(binary().get_program());
    command
        .args(["mcp", "--local-content", "--no-watch"])
        .arg(&root)
        .env("SEYK_DAEMON_DIR", &daemon)
        .env("SEYK_MODEL", root.join("no-model"))
        .env("SEYK_CACHE", root.join("no-cache.sqlite3"));
    let responses = mcp_handshake(&mut command);
    assert_eq!(
        responses[0]["result"]["serverInfo"]["name"],
        "seyk-local-content-v1"
    );
    assert!(!daemon.exists());
    assert!(!root.join("no-cache.sqlite3").exists());
    assert!(
        responses[1]["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|t| t["name"] == "search")
    );
}

#[cfg(all(feature = "mcp", unix))]
#[test]
fn local_content_refuses_missing_scope_and_symlink_root_before_model_load() {
    let temp = tempfile::Builder::new()
        .prefix("local-content-cli-")
        .tempdir()
        .unwrap();
    let root = temp.path().canonicalize().unwrap();
    let alias = root.join("alias");
    std::os::unix::fs::symlink(&root, &alias).unwrap();
    binary()
        .args(["mcp", "--local-content", "--no-watch"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "requires explicit directory roots",
        ));
    binary()
        .args(["mcp", "--local-content", "--no-watch"])
        .arg(alias)
        .env("SEYK_MODEL", root.join("no-model"))
        .assert()
        .failure()
        .stderr(predicate::str::contains("unsupported type"));
}

#[cfg(all(feature = "mcp", unix))]
#[test]
fn local_content_refuses_secret_root_before_model_load() {
    let temp = tempfile::Builder::new()
        .prefix("local-secret-cli-")
        .tempdir()
        .unwrap();
    let root = temp.path().canonicalize().unwrap().join("vault.OPVAULT");
    std::fs::create_dir(&root).unwrap();
    binary()
        .args(["mcp", "--local-content", "--no-watch"])
        .arg(&root)
        .env("SEYK_MODEL", temp.path().join("no-model"))
        .assert()
        .failure()
        .stderr(predicate::str::contains("refusing secret file"));
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
    assert!(
        responses[1]["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "search")
    );

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
    for version in 2..=6 {
        let mut document = value.clone();
        document["schema_version"] = version.into();
        let data = document.to_string();
        let raw = run(&["extract", "--from-json", "-"], &data);
        assert!(
            raw.status.success(),
            "schema {version}: {}",
            String::from_utf8_lossy(&raw.stderr)
        );
        assert_eq!(raw.stdout, expected);
        let json = run(&["extract", "--from-json", "-", "--json"], &data);
        assert!(json.status.success());
        let extracted: Value = serde_json::from_slice(&json.stdout).unwrap();
        assert_eq!(extracted["text"].as_str().unwrap().as_bytes(), expected);
    }
    for version in [1, 7] {
        let mut document = value.clone();
        document["schema_version"] = version.into();
        let bad = run(&["extract", "--from-json", "-"], &document.to_string());
        assert!(!bad.status.success(), "schema {version}");
        assert!(bad.stdout.is_empty());
    }
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
    assert_eq!(extracted["schema_version"], 6);
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
        input.replace("\"schema_version\":2", "\"schema_version\":99"),
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

#[test]
fn synced_cache_warns_on_stderr_and_still_returns_json() {
    let temp = TempDir::new().unwrap();
    std::fs::create_dir(temp.path().join(".stfolder")).unwrap();
    let cache = temp.path().join("nested/cache.sqlite3");
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
    assert_eq!(stats["in_synced_folder"], true);
    let stderr = String::from_utf8(output.stderr).unwrap();
    assert_eq!(stderr.matches("inside a known synced folder").count(), 1);
}

#[test]
fn explicit_secret_paths_are_refused_even_without_a_model() {
    let temp = TempDir::new().unwrap();
    let secret = temp.path().join("private.pem");
    std::fs::write(&secret, "synthetic secret sentinel").unwrap();
    for command in ["index", "search"] {
        let mut cmd = binary();
        cmd.arg(command);
        if command == "search" {
            cmd.arg("sentinel");
        }
        cmd.arg(&secret)
            .args(["--include", "*", "--hidden"])
            .env("SEYK_MODEL", temp.path().join("no-model"))
            .assert()
            .failure()
            .stderr(predicate::str::contains("refusing secret file"))
            .stdout(predicate::str::contains("synthetic secret sentinel").not());
    }
}

#[test]
#[ignore = "requires local Gemma weights; no downloads"]
fn secret_include_globs_cannot_bypass_cli_policy() {
    let temp = TempDir::new().unwrap();
    let notes = temp.path().join("notes");
    std::fs::create_dir(&notes).unwrap();
    for name in ["id_rsa.txt", ".env.local.toml", "vault.opvault/notes.md"] {
        let path = notes.join(name);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, "synthetic secret sentinel").unwrap();
    }
    for command in ["index", "search"] {
        let mut cmd = binary();
        cmd.arg(command);
        if command == "search" {
            cmd.arg("sentinel");
        }
        let output = cmd
            .arg(&notes)
            .args([
                "--include",
                "*",
                "--include",
                "**/*",
                "--hidden",
                "--text-only",
                "--threads",
                "1",
                "--json",
                "--cache-path",
            ])
            .arg(temp.path().join("cache.db"))
            .output()
            .unwrap();
        assert!(
            output.status.success(),
            "{}",
            String::from_utf8_lossy(&output.stderr)
        );
        let json: Value = serde_json::from_slice(&output.stdout).unwrap();
        let sync = if command == "index" {
            &json["sync"]
        } else {
            &json["stats"]["sync"]
        };
        assert_eq!(sync["resources_discovered"], 0);
        if command == "search" {
            assert_eq!(json["hits"], serde_json::json!([]));
        }
    }
}
