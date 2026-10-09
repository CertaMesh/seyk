use super::*;
#[test]
fn suppression_is_exact_and_context_is_bounded() {
    let mut logits = vec![0.0; VOCAB];
    suppress(&mut logits, true);
    assert!(logits[220].is_infinite() && logits[EOT].is_infinite());
    let mut logits = vec![0.0; VOCAB];
    suppress(&mut logits, false);
    assert!(logits[220].is_finite() && logits[EOT].is_finite());
    assert!(logits[EOT + 1..].iter().all(|x| *x == f32::NEG_INFINITY));
    let missing = Whisper::new("/nonexistent/stt");
    assert!(matches!(
        missing.transcribe(&vec![0.0; 480001], None),
        Err(Error::AudioIntervalTooLong)
    ));
    assert!(matches!(
        missing.transcribe(&[0.1], None),
        Err(Error::SttModel { .. })
    ));
}

fn f32s(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
#[test]
fn rust_mel_matches_pinned_openai_reference() {
    use std::io::Read;
    let input = f32s(include_bytes!("../tests/fixtures/mel-input.f32le"));
    let mut expected = Vec::new();
    flate2::read::GzDecoder::new(
        include_bytes!("../tests/fixtures/mel-reference.f32le.gz").as_slice(),
    )
    .read_to_end(&mut expected)
    .unwrap();
    let expected = f32s(&expected);
    let actual = mel::features(&input).unwrap();
    assert_eq!(actual.len(), 240000);
    assert_eq!(expected.len(), actual.len());
    let error = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max);
    // Full [80,3000] tensor, including reflected left edge and padded right tail.
    assert!(error <= 1e-4, "max absolute mel error {error}");
}
fn fixture(seconds: usize) -> Vec<f32> {
    let wav = std::fs::read(
        std::env::var_os("SEYK_STT_SPEECH_WAV")
            .expect("set SEYK_STT_SPEECH_WAV to locally generated speech"),
    )
    .unwrap();
    assert_eq!(&wav[36..40], b"data");
    let mut base: Vec<f32> = wav[44..]
        .as_chunks::<2>()
        .0
        .iter()
        .map(|b| i16::from_le_bytes(*b) as f32 / 32768.0)
        .collect();
    base.extend(vec![0.0; 8000]);
    base.into_iter().cycle().take(seconds * 16000).collect()
}
const EXPECTED_10: &str = " the morning train arrives at the station. Please bring a blue notebook and a pencil. The library opens at nine. Today we will walk through the garden and look at the flowers. A small bird is";
const EXPECTED_30: &str = " the morning train arrives at the station. Please bring a blue notebook and a pencil. The library opens at 9. Today we will walk through the garden and look at the flowers. A small bird is singing near the window. After lunch we will read a story together. The morning train arrives at the station. Please bring a blue notebook and a pencil. The library opens at 9. Today we will walk through the garden and look at the flowers. A small bird is singing near the window. After lunch we will read a story together. The morning train arrives at the station.";

/// Gate: SEYK_STT_SPEECH_WAV=$PWD/tmp/stt-speech.wav SEYK_STT_MODEL_DIR=<pinned directory> cargo test -p seyk-stt --release -- --include-ignored
#[test]
#[ignore = "requires explicitly provisioned Xenova FP32 model; no network in test"]
fn model_backed_transcript_cache_parity_silence_and_installation() {
    let directory =
        PathBuf::from(std::env::var_os("SEYK_STT_MODEL_DIR").expect("set SEYK_STT_MODEL_DIR"));
    let provisioned = tempfile::tempdir().unwrap();
    let directory = manifest::install_with(&provisioned.path().join("model"), |asset| {
        Ok(directory.join(&asset.path))
    })
    .unwrap();
    manifest::verify_directory(&directory).unwrap();
    let mut state = State::load(&directory, 1).unwrap();
    let (short, _, short_tokens) = state.decode(&fixture(10), None, false).unwrap();
    assert_eq!(short.text, EXPECTED_10);
    assert_eq!(short.language.as_deref(), Some("en"));
    assert!(!short.no_speech);
    let (long, error, long_tokens) = state.decode(&fixture(30), None, true).unwrap();
    assert_eq!(long.text, EXPECTED_30);
    assert!(error < 1e-4, "max cached/uncached error {error}");
    let golden: serde_json::Value =
        serde_json::from_str(include_str!("../tests/fixtures/token-goldens.json")).unwrap();
    assert_eq!(
        serde_json::to_value(short_tokens).unwrap(),
        golden["seconds_10"]
    );
    assert_eq!(
        serde_json::to_value(long_tokens).unwrap(),
        golden["seconds_30"]
    );
    drop(state);
    let speech = Whisper::new(&directory);
    for language in [None, Some("en")] {
        let silence = speech.transcribe(&vec![0.0; 160000], language).unwrap();
        assert!(silence.no_speech && silence.text.is_empty() && silence.language.is_none());
    }
    drop(speech);
    let temp = tempfile::tempdir().unwrap();
    let installed = temp.path().join("installed");
    manifest::install_with(&installed, |asset| Ok(directory.join(&asset.path))).unwrap();
    let graph = installed.join("onnx/encoder_model.onnx");
    // Equal-length tampering of an owned copy: never mutate the user's model.
    let mut bytes = std::fs::read(&graph).unwrap();
    bytes[0] ^= 1;
    std::fs::write(&graph, &bytes).unwrap();
    assert!(matches!(
        Whisper::new(&installed).transcribe(&[0.1; 1600], None),
        Err(Error::SttModel { .. })
    ));
    manifest::install_with(&installed, |asset| Ok(directory.join(&asset.path))).unwrap();
    std::fs::remove_file(&graph).unwrap();
    assert!(matches!(
        Whisper::new(&installed).transcribe(&[0.1; 1600], None),
        Err(Error::SttModel { .. })
    ));
    manifest::install_with(&installed, |asset| Ok(directory.join(&asset.path))).unwrap();
    manifest::verify_directory(&installed).unwrap();
    std::fs::write(installed.join("unexpected"), b"extra").unwrap();
    assert!(matches!(
        Whisper::new(&installed).transcribe(&[0.1; 1600], None),
        Err(Error::SttModel { .. })
    ));
}

#[test]
fn no_speech_threshold_includes_equality_and_eot_denominator() {
    assert!(classify_no_speech(0.600001, -2.0, 1));
    assert!(!classify_no_speech(0.6, -2.0, 1));
    assert!(!classify_no_speech(0.9, -1.999999, 1));
    assert!(classify_no_speech(0.9, -5.0, 4));
}

#[test]
fn manifest_pins_bundled_assets_and_policy_scalars() {
    use sha2::{Digest, Sha256};
    let value: serde_json::Value = serde_json::from_str(manifest::MANIFEST).unwrap();
    assert_eq!(manifest::manifest_sha256(), manifest::MANIFEST_SHA256);
    for (name, bytes) in [
        (
            "OPENAI-MIT.txt",
            include_bytes!("../assets/OPENAI-MIT.txt").as_slice(),
        ),
        (
            "APACHE-2.0.txt",
            include_bytes!("../assets/APACHE-2.0.txt").as_slice(),
        ),
        (
            "XENOVA-CARD.md",
            include_bytes!("../assets/XENOVA-CARD.md").as_slice(),
        ),
        (
            "mel-80.f32le",
            include_bytes!("../assets/mel-80.f32le").as_slice(),
        ),
    ] {
        let pin = if name == "mel-80.f32le" {
            &value["mel"]
        } else {
            value["notices"]
                .as_array()
                .unwrap()
                .iter()
                .find(|n| n["file"] == name)
                .unwrap()
        };
        assert_eq!(
            format!("{:x}", Sha256::digest(bytes)),
            pin["sha256"].as_str().unwrap(),
            "{name}"
        );
    }
    for (name, id) in [
        ("sot", SOT),
        ("eot", EOT),
        ("transcribe", TRANSCRIBE),
        ("no_timestamps", NO_TIMESTAMPS),
        ("no_speech_token", NO_SPEECH),
    ] {
        assert_eq!(value["policy"][name].as_u64().unwrap(), id as u64);
    }
    assert_eq!(
        value["policy"]["begin_suppress"],
        serde_json::json!([220, EOT])
    );
    assert_eq!(value["policy"]["suppress_above_eot"], true);
}
