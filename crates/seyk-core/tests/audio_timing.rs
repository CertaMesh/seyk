#![cfg(feature = "audio")]
use seyk_core::{Error, ResourceId, ResourceLocator, audio};
use std::path::Path;
fn decode(path: &Path) -> Vec<f32> {
    let id = ResourceId::from_path(path).unwrap();
    let mut pcm = Vec::new();
    audio::windows(std::fs::read(path).unwrap(), &id, |locator, samples| {
        let ResourceLocator::Time { start_ms, .. } = locator else {
            unreachable!()
        };
        let start = start_ms as usize * 16;
        if pcm.len() < start + samples.len() {
            pcm.resize(start + samples.len(), 0.0);
        }
        pcm[start..start + samples.len()].copy_from_slice(&samples);
        Ok(())
    })
    .unwrap();
    pcm
}
#[test]
#[ignore = "requires scripts/audio/timing.py fixtures in SEYK_AUDIO_TIMING; no weights"]
fn codec_presentation_timing() {
    let root = std::path::PathBuf::from(std::env::var("SEYK_AUDIO_TIMING").unwrap());
    let records: Vec<serde_json::Value> =
        serde_json::from_slice(&std::fs::read(root.join("manifest.json")).unwrap()).unwrap();
    for record in records {
        let name = record["file"].as_str().unwrap();
        let actual = decode(&root.join(name));
        let expected = decode(&root.join(record["reference"].as_str().unwrap()));
        assert!(
            actual.len().abs_diff(expected.len()) <= 1,
            "{name}: length {} expected {}",
            actual.len(),
            expected.len()
        );
        // Cross-correlation over the interior; broad search also catches AAC
        // priming and MP3 delay errors, not merely a near-zero local maximum.
        let lag = (-1000i32..=1000)
            .map(|lag| {
                let score = (1100..expected.len().min(actual.len()) - 1100)
                    .step_by(3)
                    .map(|i| expected[i] as f64 * actual[(i as i32 + lag) as usize] as f64)
                    .sum::<f64>();
                (lag, score)
            })
            .max_by(|a, b| a.1.total_cmp(&b.1))
            .unwrap()
            .0;
        println!(
            "AUDIO_TIMING file={name} lag_samples={lag} samples={} reference_samples={}",
            actual.len(),
            expected.len()
        );
        assert!(lag.abs() <= 1, "{name}: lag {lag}");
    }
    let id: ResourceId = "mem://audio/unsupported".parse().unwrap();
    for name in ["multi.m4a", "he-aac.m4a"] {
        let error = audio::windows(std::fs::read(root.join(name)).unwrap(), &id, |_, _| Ok(()))
            .unwrap_err();
        assert!(matches!(error, Error::UnsupportedAudio { .. }));
        println!("AUDIO_REJECT file={name} error={error}");
    }
}
