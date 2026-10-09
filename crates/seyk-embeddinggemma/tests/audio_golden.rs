#![cfg(feature = "audio")]
use seyk_embeddinggemma::{Embedder, EmbeddingGemma2, Modality, audio};
#[path = "support/audio_signal.rs"]
mod fixture;
fn floats(bytes: &[u8]) -> Vec<f32> {
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect()
}
#[test]
fn reference_features_and_token_mask() {
    assert_eq!(
        fixture::signal(),
        floats(include_bytes!("fixtures/audio/audio_signal.f32"))
    );
    for (name, samples, expected, reference) in cases() {
        let pre = audio::preprocess(&samples).unwrap();
        assert_eq!(pre.values.len(), expected.len());
        assert_eq!(
            pre.mask,
            serde_json::from_value::<Vec<bool>>(reference["mask"].clone()).unwrap()
        );
        assert_eq!(
            pre.mask.len(),
            reference["features_shape"][0].as_u64().unwrap() as usize
        );
        let max = pre
            .values
            .iter()
            .zip(expected)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max <= 1e-5, "max_abs_diff={max}");
        let tokens = reference["input_ids"]
            .as_array()
            .unwrap()
            .iter()
            .filter(|v| v.as_u64() == Some(258881))
            .count();
        assert_eq!(pre.mask.iter().step_by(4).filter(|v| **v).count(), tokens);
        println!(
            "AUDIO_FEATURE_PARITY case={name} max_abs_diff={max} frames={} tokens={tokens}",
            pre.mask.len()
        );
    }
}

#[test]
#[ignore = "requires pinned local q8 text/audio weights; no downloads"]
fn reference_embedding_and_text_cache_identity() {
    let model = EmbeddingGemma2::builder()
        .dimension(768)
        .vision(false)
        .audio(true)
        .threads(1)
        .build()
        .unwrap();
    let text = EmbeddingGemma2::builder()
        .dimension(768)
        .vision(false)
        .audio(false)
        .threads(1)
        .build()
        .unwrap();
    assert_eq!(model.identity(), text.identity());
    assert_eq!(
        model.embed_query("unchanged text input").unwrap(),
        text.embed_query("unchanged text input").unwrap()
    );
    assert_ne!(
        model.input_identity(Modality::Audio),
        text.input_identity(Modality::Audio)
    );
    for (name, samples, _, reference) in cases() {
        let count = audio::preprocess(&samples)
            .unwrap()
            .mask
            .iter()
            .step_by(4)
            .filter(|v| **v)
            .count();
        let ids = model
            .token_ids(&format!("<|audio>{}<audio|>", "<|audio|>".repeat(count)))
            .unwrap();
        assert_eq!(
            ids,
            serde_json::from_value::<Vec<u32>>(reference["input_ids"].clone()).unwrap()
        );
        let expected: Vec<f32> = serde_json::from_value(reference["embedding"].clone()).unwrap();
        let embedding = model.embed_audio(&samples, 16_000).unwrap();
        let dot: f32 = embedding
            .as_slice()
            .iter()
            .zip(&expected)
            .map(|(a, b)| a * b)
            .sum();
        let norm = expected.iter().map(|v| v * v).sum::<f32>().sqrt();
        let cosine = dot / norm;
        println!("AUDIO_EMBEDDING_PARITY case={name} cosine={cosine}");
        assert!(cosine >= 0.99);
    }
}

#[test]
#[ignore = "requires pinned local q8 text weights; exercises missing optional encoder without downloads"]
fn missing_audio_files_leave_text_available() {
    let mut config = seyk_embeddinggemma::GemmaConfig::default();
    config.vision = false;
    let source = seyk_embeddinggemma::locate(&config).unwrap();
    let dir = tempfile::tempdir().unwrap();
    for file in config.files() {
        let target = dir.path().join(&file);
        std::fs::create_dir_all(target.parent().unwrap()).unwrap();
        std::fs::hard_link(source.join(&file).canonicalize().unwrap(), target).unwrap();
    }
    let missing = EmbeddingGemma2::builder()
        .model_path(dir.path())
        .vision(false)
        .audio(true)
        .threads(1)
        .build()
        .unwrap();
    let text = EmbeddingGemma2::builder()
        .model_path(dir.path())
        .vision(false)
        .audio(false)
        .threads(1)
        .build()
        .unwrap();
    assert_eq!(missing.identity(), text.identity());
    assert!(missing.embed_query("text remains searchable").is_ok());
    assert!(matches!(
        missing.embed_audio(&fixture::signal(), 16000),
        Err(seyk_embeddinggemma::Error::ModelNotFound(_))
    ));
    // Installing only audio weights changes the audio identity, never text's.
    for file in audio::FILES {
        std::fs::hard_link(
            source.join(file).canonicalize().unwrap(),
            dir.path().join(file),
        )
        .unwrap();
    }
    let installed = EmbeddingGemma2::builder()
        .model_path(dir.path())
        .vision(false)
        .audio(true)
        .threads(1)
        .build()
        .unwrap();
    assert_eq!(missing.identity(), installed.identity());
    assert!(matches!(
        missing.embed_audio(&fixture::signal(), 16000),
        Err(seyk_embeddinggemma::Error::ModelNotFound(_))
    ));
    // A same-length edit between construction and lazy load must not cache a
    // vector under the old identity. Replace a hard link, never mutate HF data.
    let graph = dir.path().join(audio::FILES[0]);
    let mut bytes = std::fs::read(&graph).unwrap();
    bytes[0] ^= 1;
    std::fs::remove_file(&graph).unwrap();
    std::fs::write(&graph, bytes).unwrap();
    let error = installed
        .embed_audio(&fixture::signal(), 16000)
        .unwrap_err();
    assert!(error.to_string().contains("audio weights changed"));
    assert_ne!(
        missing.input_identity(Modality::Audio),
        installed.input_identity(Modality::Audio)
    );
}

fn cases() -> impl Iterator<Item = (&'static str, Vec<f32>, Vec<f32>, serde_json::Value)> {
    [
        (
            "one-second",
            16000,
            include_bytes!("fixtures/audio/features.f32").as_slice(),
            include_str!("fixtures/audio/reference.json"),
        ),
        (
            "partial-frame",
            16001,
            include_bytes!("fixtures/audio/tail-features.f32").as_slice(),
            include_str!("fixtures/audio/tail.json"),
        ),
        (
            "full-window",
            160000,
            include_bytes!("fixtures/audio/full-features.f32").as_slice(),
            include_str!("fixtures/audio/full.json"),
        ),
    ]
    .into_iter()
    .map(|(name, len, features, reference)| {
        (
            name,
            fixture::signal_len(len),
            floats(features),
            serde_json::from_str(reference).unwrap(),
        )
    })
}
