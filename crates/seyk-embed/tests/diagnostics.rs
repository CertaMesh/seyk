use seyk_embed::diagnostics::{Phase, counts};

#[test]
fn diagnostic_child() {
    if std::env::var_os("SEYK_DIAGNOSTIC_CHILD").is_none() {
        return;
    }
    let _phase = Phase::new("test_phase");
    counts("test_shape", &[("rows", 4), ("max_tokens", 1024)]);
}

#[test]
fn diagnostics_require_opt_in_and_emit_only_phase_and_numeric_metadata() {
    for enabled in [false, true] {
        let mut child = std::process::Command::new(std::env::current_exe().unwrap());
        child
            .args(["--exact", "diagnostic_child", "--nocapture"])
            .env("SEYK_DIAGNOSTIC_CHILD", "1");
        if enabled {
            child.env("SEYK_MEMORY_DIAGNOSTICS", "1");
        } else {
            child.env_remove("SEYK_MEMORY_DIAGNOSTICS");
        }
        let output = child.output().unwrap();
        assert!(output.status.success());
        let stderr = String::from_utf8(output.stderr).unwrap();
        if enabled {
            let lines: Vec<_> = stderr.lines().collect();
            assert_eq!(lines.len(), 3);
            assert!(lines[0].contains("event=test_phase state=begin"));
            assert!(lines[1].contains("event=test_shape state=count rows=4 max_tokens=1024"));
            assert!(lines[2].contains("event=test_phase state=end"));
            assert!(
                lines
                    .iter()
                    .all(|line| line.starts_with("seyk_memory pid="))
            );
        } else {
            assert!(stderr.is_empty());
        }
    }
}
