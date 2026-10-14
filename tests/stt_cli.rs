#![cfg(all(feature = "cli", feature = "audio-stt"))]
use seyk::{
    Modality, ResourceId, ResourceLocator, ResourceMetadata, SearchHit, SearchMode, SearchResults,
};
use std::{
    path::Path,
    process::{Command, Output},
};
fn input(dir: &Path, end_ms: u64, bytes: &[u8]) -> std::path::PathBuf {
    let wav = dir.join("public.wav");
    std::fs::write(&wav, bytes).unwrap();
    let result = SearchResults {
        queries: vec!["public fixture".into()],
        mode: SearchMode::Semantic,
        stats: Default::default(),
        hits: vec![SearchHit {
            resource: ResourceId::from_path(&wav).unwrap(),
            locator: ResourceLocator::Time {
                start_ms: 0,
                end_ms,
            },
            metadata: ResourceMetadata {
                media_type: Some("audio/wav".into()),
                modality: Modality::Audio,
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
    let mut value = serde_json::to_value(result).unwrap();
    value["schema_version"] = 4.into();
    let path = dir.join("hit.json");
    std::fs::write(&path, serde_json::to_vec(&value).unwrap()).unwrap();
    path
}
fn chirp() -> Vec<u8> {
    let n = 160000u32;
    let mut wav = Vec::new();
    wav.extend(b"RIFF");
    wav.extend((36 + n * 2).to_le_bytes());
    wav.extend(b"WAVEfmt ");
    wav.extend(16u32.to_le_bytes());
    wav.extend(1u16.to_le_bytes());
    wav.extend(1u16.to_le_bytes());
    wav.extend(16000u32.to_le_bytes());
    wav.extend(32000u32.to_le_bytes());
    wav.extend(2u16.to_le_bytes());
    wav.extend(16u16.to_le_bytes());
    wav.extend(b"data");
    wav.extend((n * 2).to_le_bytes());
    for i in 0..n {
        let t = i as f64 / 16000.0;
        wav.extend(
            ((0.2 * (std::f64::consts::TAU * (170.0 * t + 20.0 * t * t)).sin() * 32767.0) as i16)
                .to_le_bytes(),
        );
    }
    wav
}
fn extract(input: &Path, model: &Path, json: bool) -> Output {
    let mut command = Command::new(env!("CARGO_BIN_EXE_seyk"));
    command
        .args(["extract", "--from-json"])
        .arg(input)
        .arg("--stt-model-dir")
        .arg(model)
        .env("HF_HUB_OFFLINE", "1")
        .env("SEYK_MODEL", model.join("not-a-gemma-model"));
    if json {
        command.arg("--json");
    }
    command.output().unwrap()
}
#[test]
fn missing_model_cap_and_stale_hash_fail_without_content() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing");
    let file = input(dir.path(), 10000, &chirp());
    let out = extract(&file, &missing, false);
    assert!(!out.status.success() && out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("seyk model download --stt"));
    let file = input(dir.path(), 30001, &chirp());
    let out = extract(&file, &missing, false);
    assert!(!out.status.success() && out.stdout.is_empty());
    assert!(String::from_utf8_lossy(&out.stderr).contains("30-second"));
    let file = input(dir.path(), 10000, &chirp());
    std::fs::write(dir.path().join("public.wav"), b"changed").unwrap();
    let out = extract(&file, &missing, false);
    assert!(!out.status.success() && out.stdout.is_empty());
    assert!(!String::from_utf8_lossy(&out.stderr).contains("model download"));
}
/// Gate: SEYK_STT_SPEECH_WAV=$PWD/tmp/stt-speech.wav SEYK_STT_MODEL_DIR=<asset source> cargo test --features audio-stt --test stt_cli -- --include-ignored
#[test]
#[ignore = "requires local pinned Whisper files; no downloads"]
fn human_label_and_json_provenance_with_real_model() {
    let source = std::path::PathBuf::from(std::env::var_os("SEYK_STT_MODEL_DIR").unwrap());
    let dir = tempfile::tempdir().unwrap();
    let model = seyk::stt::manifest::install_with(&dir.path().join("model"), |asset| {
        Ok(source.join(&asset.path))
    })
    .unwrap();
    let speech =
        std::fs::read(std::env::var_os("SEYK_STT_SPEECH_WAV").expect("set SEYK_STT_SPEECH_WAV"))
            .unwrap();
    let file = input(dir.path(), 10000, &speech);
    let human = extract(&file, &model, false);
    assert!(
        human.status.success(),
        "{}",
        String::from_utf8_lossy(&human.stderr)
    );
    assert!(
        human
            .stdout
            .starts_with(b"Machine transcript (may contain errors)\n")
    );
    let out = extract(&file, &model, true);
    assert!(out.status.success());
    let value: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    assert_eq!(value["schema_version"], 6);
    assert_eq!(value["provenance"]["method"], "speech_to_text");
    let p = &value["provenance"]["speech"];
    assert_eq!(p["fidelity"], "machine_transcription");
    assert_eq!(p["sample_start"], 0);
    assert_eq!(p["sample_end"], 160000);
    assert_eq!(p["language_detected"], "en");
    assert_eq!(p["no_speech"], false);
    assert_eq!(p["manifest_sha256"], seyk::stt::manifest::manifest_sha256());
    assert!(out.stderr.is_empty());
    assert!(human.stderr.is_empty());
}

#[test]
fn download_rejects_extraction_only_options() {
    for option in ["--stt-model-dir", "--stt-language"] {
        let output = Command::new(env!("CARGO_BIN_EXE_seyk"))
            .args(["model", "download", "--stt", option, "unused"])
            .output()
            .unwrap();
        assert!(!output.status.success());
        assert!(
            String::from_utf8_lossy(&output.stderr)
                .contains("configure extraction, not model download")
        );
    }
}
