#![cfg(feature = "cli")]

use std::process::Command;

fn setup(args: &[&str]) -> std::process::Output {
    let home = tempfile::tempdir().unwrap();
    Command::new(env!("CARGO_BIN_EXE_seyk"))
        .arg("setup")
        .args(args)
        .env("HOME", home.path())
        .env("HF_HOME", home.path().join("hf"))
        .env("XDG_CACHE_HOME", home.path().join("cache"))
        .env("HF_HUB_CACHE", home.path().join("hf/hub"))
        .env("HF_ENDPOINT", "http://127.0.0.1:0")
        .output()
        .unwrap()
}

#[test]
fn setup_status_is_offline_and_lists_every_model() {
    let output = setup(&["--json"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let status: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(status["schema_version"], 9);
    assert_eq!(status["action"], "status");
    let names: Vec<_> = status["models"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["model"].as_str().unwrap())
        .collect();
    assert_eq!(names, ["text", "audio", "stt"]);
    assert_eq!(status["models"][0]["verification"], "presence");
    assert_eq!(status["models"][1]["verification"], "sha256");
}

#[test]
fn audio_plan_names_pins_and_total_without_network() {
    let output = setup(&["audio", "--dry-run", "--json"]);
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let plan: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(plan["schema_version"], 9);
    assert_eq!(plan["action"], "plan");
    assert_eq!(plan["models"].as_array().unwrap().len(), 2);
    assert_eq!(plan["models"][0]["bytes"], 340307954u64);
    assert_eq!(
        plan["models"][0]["revision"],
        "daa72c51243991dfcaf9f9137d2c573d8f7790c0"
    );
    assert_eq!(
        plan["models"][1]["revision"],
        "5332fcc35e32a33b86612b9a57a89be7906102b1"
    );
    assert!(plan["total_bytes"].as_u64().unwrap() > 600_000_000);
    assert_eq!(plan["download_bytes"], plan["total_bytes"]);
    assert!(plan["next_step"].as_str().unwrap().contains("seyk search"));
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(stderr.contains("Total:") && stderr.contains("Whisper tiny"));
}

#[test]
fn human_status_and_plan_include_next_steps() {
    let output = setup(&[]);
    assert!(output.status.success());
    let status = String::from_utf8_lossy(&output.stdout);
    assert!(status.contains("text:") && status.contains("audio:") && status.contains("stt:"));
    assert!(status.contains("run seyk setup audio"));
    let output = setup(&["audio", "--dry-run"]);
    assert!(output.status.success());
    assert!(String::from_utf8_lossy(&output.stdout).contains("./recordings"));
}

#[test]
fn repeating_setup_plan_is_idempotent_and_creates_no_model_files() {
    let home = tempfile::tempdir().unwrap();
    let plan = || {
        Command::new(env!("CARGO_BIN_EXE_seyk"))
            .args(["setup", "audio", "--dry-run", "--json"])
            .env("HOME", home.path())
            .env("HF_HOME", home.path().join("hf"))
            .env("XDG_CACHE_HOME", home.path().join("cache"))
            .env("HF_ENDPOINT", "http://127.0.0.1:0")
            .output()
            .unwrap()
    };
    let first = plan();
    let second = plan();
    assert!(first.status.success() && second.status.success());
    assert_eq!(first.stdout, second.stdout);
    assert_eq!(first.stderr, second.stderr);
    assert_eq!(std::fs::read_dir(home.path()).unwrap().count(), 0);
}
