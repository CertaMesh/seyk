#![cfg(feature = "audio")]
use seyk_core::{ResourceId, audio};

/// Baseline captured at 7cab7ee before changing preparation, with the same files.
/// Source hashes distinguish a decoder change from a regenerated fixture change.
#[test]
#[ignore = "requires scripts/audio/timing.py fixtures from the recorded baseline"]
fn input_hashes_remain_identical() {
    let root = std::path::PathBuf::from(std::env::var("SEYK_AUDIO_TIMING").unwrap());
    let mut rows = std::collections::BTreeMap::new();
    for entry in std::fs::read_dir(&root).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().to_string();
        if !matches!(
            path.extension().and_then(|s| s.to_str()),
            Some("wav" | "flac" | "mp3" | "m4a")
        ) || matches!(name.as_str(), "multi.m4a" | "he-aac.m4a")
        {
            continue;
        }
        let bytes = std::fs::read(&path).unwrap();
        let source_hash = blake3::hash(&bytes).to_hex().to_string();
        let id = ResourceId::from_path(&path).unwrap();
        let mut windows = Vec::new();
        audio::windows(bytes, &id, |locator, samples| {
            let mut hash = blake3::Hasher::new();
            hash.update(b"audio-input-v1");
            hash.update(audio::PREPARATION.as_bytes());
            let identity = b"fixture-audio-identity";
            hash.update(&(identity.len() as u64).to_le_bytes());
            hash.update(identity);
            hash.update(&(samples.len() as u64).to_le_bytes());
            for sample in &samples { hash.update(&sample.to_le_bytes()); }
            let source = audio::VerifiedReader::new(
                Box::new(std::fs::File::open(&path).unwrap()), &id,
                seyk_core::discovery::audio_byte_limit(7200).unwrap(),
            ).unwrap();
            let slice = audio::slice_verified(source, &id, &locator, 7200).unwrap();
            let real_len = if name == "one-sample.wav" { 1 } else { samples.len() };
            assert_eq!(slice.len(), real_len, "{name}: unpadded extent");
            let actual: Vec<_> = slice.iter().map(|f| f.to_bits()).collect();
            let expected: Vec<_> = samples[..real_len].iter().map(|f| f.to_bits()).collect();
            assert_eq!(actual, expected, "{name}: extraction changed model samples");
            windows.push(serde_json::json!({"locator":locator,"input_hash":hash.finalize().to_hex().to_string()}));
            Ok(())
        }).unwrap();
        rows.insert(
            name,
            serde_json::json!({"source_hash":source_hash,"windows":windows}),
        );
    }
    let actual = serde_json::to_value(rows).unwrap();
    let expected: serde_json::Value =
        serde_json::from_slice(include_bytes!("fixtures/audio-input-hashes.json")).unwrap();
    assert_eq!(
        actual.as_object().unwrap().len(),
        expected.as_object().unwrap().len()
    );
    for (name, row) in actual.as_object().unwrap() {
        // Containers may carry new creation timestamps after regeneration.
        // The committed source hashes retain baseline provenance; the gate is
        // the exact model-input hash and locator, not container metadata.
        assert_eq!(
            row["windows"], expected[name]["windows"],
            "{name}: baseline input changed"
        );
    }
}
