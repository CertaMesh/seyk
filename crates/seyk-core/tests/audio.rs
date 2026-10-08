#![cfg(feature = "audio")]
mod common;
#[cfg(feature = "pdf")]
#[path = "support/pdf_fixture.rs"]
mod pdf_fixture;
use seyk_core::{
    Dimension, Embedder, Embedding, EmbeddingRequest, EmbeddingTask, Error, Modality,
    ModelIdentity, ResourceId, ResourceInput, ResourceLocator, SearchEngine, SearchMode,
    SqliteCache, audio,
};
use std::sync::{
    Arc, Mutex,
    atomic::{AtomicUsize, Ordering},
};
fn wav(samples: usize, rate: u32) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend(b"RIFF");
    bytes.extend((36 + samples as u32 * 2).to_le_bytes());
    bytes.extend(b"WAVEfmt ");
    bytes.extend(16u32.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(1u16.to_le_bytes());
    bytes.extend(rate.to_le_bytes());
    bytes.extend((rate * 2).to_le_bytes());
    bytes.extend(2u16.to_le_bytes());
    bytes.extend(16u16.to_le_bytes());
    bytes.extend(b"data");
    bytes.extend((samples as u32 * 2).to_le_bytes());
    for i in 0..samples {
        bytes.extend(((i.wrapping_mul(7919) % 32767) as i16).to_le_bytes());
    }
    bytes
}
struct Model {
    identity: ModelIdentity,
    policy: &'static str,
    calls: Mutex<Vec<usize>>,
    fail_after: AtomicUsize,
}
impl Model {
    fn new(policy: &'static str) -> Self {
        Self {
            identity: common::FakeEmbedder::new(Dimension::D128, "audio")
                .identity()
                .clone(),
            policy,
            calls: Mutex::new(Vec::new()),
            fail_after: AtomicUsize::new(usize::MAX),
        }
    }
}
impl Embedder for Model {
    fn identity(&self) -> &ModelIdentity {
        &self.identity
    }
    fn input_identity(&self, m: Modality) -> &str {
        if m == Modality::Audio {
            self.policy
        } else {
            ""
        }
    }
    fn supports(&self, _: Modality) -> bool {
        true
    }
    fn embed_batch(&self, requests: &[EmbeddingRequest<'_>]) -> seyk_embed::Result<Vec<Embedding>> {
        for r in requests {
            if let ResourceInput::Audio {
                samples,
                sample_rate,
            } = &r.input
            {
                assert_eq!(*sample_rate, 16000);
                assert!(samples.len() <= 160000);
                let mut calls = self.calls.lock().unwrap();
                calls.push(samples.len());
                if calls.len() > self.fail_after.load(Ordering::Relaxed) {
                    return Err(seyk_embed::Error::ModelNotFound(
                        "synthetic missing audio weights".into(),
                    ));
                }
            }
        }
        requests
            .iter()
            .map(|r| {
                let mut vector = vec![0.0; 128];
                vector[usize::from(r.task == EmbeddingTask::SearchDocument)] = 1.0;
                Embedding::from_raw(vector, Dimension::D128)
            })
            .collect()
    }
}
#[test]
fn window_tails_round_outward_and_corruption_is_typed() {
    let id: ResourceId = "mem://audio/wave".parse().unwrap();
    let mut rows = Vec::new();
    audio::windows(wav(256001, 16000), &id, |locator, pcm| {
        rows.push((locator, pcm.len()));
        Ok(())
    })
    .unwrap();
    assert_eq!(
        rows,
        vec![
            (
                ResourceLocator::Time {
                    start_ms: 0,
                    end_ms: 10000
                },
                160000
            ),
            (
                ResourceLocator::Time {
                    start_ms: 8000,
                    end_ms: 16001
                },
                128001
            ),
            (
                ResourceLocator::Time {
                    start_ms: 16000,
                    end_ms: 16001
                },
                320
            )
        ]
    );
    assert!(matches!(
        audio::windows(vec![0; 128], &id, |_, _| Ok(())),
        Err(Error::UnsupportedAudio { .. })
    ));
}
#[test]
fn bounded_batches_atomic_binding_and_no_audio_at_rest() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clip.wav");
    let db = dir.path().join("cache.db");
    std::fs::write(&path, wav(16000 * 49, 16000)).unwrap();
    let model = Arc::new(Model::new("policy1"));
    let engine = SearchEngine::builder()
        .embedder(model.clone())
        .cache(SqliteCache::open(&db).unwrap())
        .batch_size(2)
        .build()
        .unwrap();
    let result = engine
        .search("query")
        .paths([&path])
        .mode(SearchMode::Semantic)
        .limit(20)
        .execute()
        .unwrap();
    assert!(result.stats.sync.skipped.is_empty());
    assert_eq!(result.hits.len(), 7);
    assert!(
        result
            .hits
            .iter()
            .all(|h| h.preview.is_none() && h.metadata.media_type.as_deref() == Some("audio/wav"))
    );
    let error = engine.extract(&result.hits[0]).unwrap_err();
    assert!(matches!(
        error,
        Error::UnsupportedExtraction {
            modality: Modality::Audio,
            ..
        }
    ));
    assert!(error.to_string().contains("STT not available until 5b"));
    // Different file content: two successful windows then a failure. Binding
    // must remain entirely the previous seven windows, never a partial version.
    model
        .fail_after
        .store(model.calls.lock().unwrap().len() + 2, Ordering::Relaxed);
    let mut changed = wav(16000 * 51, 16000);
    for byte in &mut changed[44..] {
        *byte ^= 0x55;
    }
    std::fs::write(&path, changed).unwrap();
    let failed = engine.index().paths([&path]).execute().unwrap();
    assert_eq!(failed.sync.skipped.len(), 1);
    let after = engine
        .search("query")
        .paths([&path])
        .sync(false)
        .mode(SearchMode::Semantic)
        .limit(20)
        .execute()
        .unwrap();
    assert_eq!(
        after.hits.iter().map(|h| &h.locator).collect::<Vec<_>>(),
        result.hits.iter().map(|h| &h.locator).collect::<Vec<_>>()
    );
    drop(engine);
    let data = std::fs::read(db).unwrap();
    let source = wav(200, 16000);
    let floats: Vec<u8> = source[44..]
        .as_chunks::<2>()
        .0
        .iter()
        .flat_map(|b| (i16::from_le_bytes(*b) as f32 / 32768.0).to_le_bytes())
        .collect();
    assert!(!data.windows(64).any(|w| w == &floats[40..104]));
}
#[test]
fn modality_policy_change_reprepares_audio_without_invalidating_text() {
    let dir = tempfile::tempdir().unwrap();
    let audio_path = dir.path().join("audio.wav");
    let text_path = dir.path().join("text.txt");
    let db = dir.path().join("cache.db");
    std::fs::write(&audio_path, wav(20000, 16000)).unwrap();
    std::fs::write(&text_path, "unchanged text").unwrap();
    for (policy, updated, unchanged) in [("timing1", 2, 0), ("timing2", 1, 1)] {
        let model = Arc::new(Model::new(policy));
        let engine = SearchEngine::builder()
            .embedder(model.clone())
            .cache(SqliteCache::open(&db).unwrap())
            .build()
            .unwrap();
        let stats = engine
            .index()
            .paths([&audio_path, &text_path])
            .execute()
            .unwrap()
            .sync;
        assert_eq!(stats.resources_updated, updated);
        assert_eq!(stats.resources_unchanged, unchanged);
        assert_eq!(model.calls.lock().unwrap().len(), 1);
        use seyk_core::{EmbeddingCache, ResourceWrite, SegmentWrite};
        let profile = engine.profile_id().to_string();
        drop(engine);
        let id = ResourceId::from_path(std::fs::canonicalize(&audio_path).unwrap()).unwrap();
        let mut cache = SqliteCache::open(&db).unwrap();
        let mut cached = cache.resource(&profile, &id).unwrap().unwrap();
        if policy == "timing1" {
            // Seed the evidence a previous timing policy would have produced,
            // with identical file bytes/version. The next descriptor must force
            // reconstruction of locators as well as embedding input hashes.
            cached.segments[0].locator = ResourceLocator::Time {
                start_ms: 20,
                end_ms: 1270,
            };
            cache
                .put_resource(
                    &profile,
                    &id,
                    &ResourceWrite {
                        metadata: &cached.metadata,
                        version: &cached.version,
                        descriptor_hash: cached.descriptor_hash.as_deref(),
                        segments: cached
                            .segments
                            .iter()
                            .map(|segment| SegmentWrite {
                                segment,
                                lexical_body: "",
                            })
                            .collect(),
                    },
                )
                .unwrap();
        } else {
            assert_eq!(
                cached.segments[0].locator,
                ResourceLocator::Time {
                    start_ms: 0,
                    end_ms: 1250
                }
            );
        }
    }
}
#[test]
fn missing_audio_weights_and_corrupt_audio_do_not_block_text() {
    let dir = tempfile::tempdir().unwrap();
    let good = dir.path().join("text.txt");
    let audio = dir.path().join("audio.wav");
    let corrupt = dir.path().join("bad.mp3");
    std::fs::write(&good, "safe text").unwrap();
    std::fs::write(&audio, wav(20000, 16000)).unwrap();
    std::fs::write(&corrupt, b"not audio").unwrap();
    std::fs::write(
        dir.path().join("image.png"),
        b"fake model accepts image bytes",
    )
    .unwrap();
    #[cfg(feature = "pdf")]
    std::fs::write(
        dir.path().join("document.pdf"),
        pdf_fixture::pdf(&["Readable public text"], pdf_fixture::Kind::Plain),
    )
    .unwrap();
    let model = Model::new("missing");
    model.fail_after.store(0, Ordering::Relaxed);
    let engine = SearchEngine::builder().embedder(model).build().unwrap();
    let stats = engine.index().paths([dir.path()]).execute().unwrap().sync;
    assert_eq!(
        stats.resources_updated,
        2 + usize::from(cfg!(feature = "pdf"))
    );
    assert_eq!(stats.skipped.len(), 2);
    assert!(
        stats
            .skipped
            .iter()
            .any(|s| s.reason.contains("model files not found"))
    );
}

#[test]
fn truncated_wave_is_not_committed() {
    let id: ResourceId = "mem://audio/truncated".parse().unwrap();
    let mut bytes = wav(32000, 16000);
    bytes.truncate(bytes.len() - 1000);
    let error = audio::windows(bytes, &id, |_, _| Ok(())).unwrap_err();
    assert!(matches!(error, Error::UnsupportedAudio { .. }));
}

#[test]
fn decoded_duration_cap_is_independent_of_compression() {
    // 8 kHz mono input needs only half the encoded byte budget to exceed the
    // cap, while preserving one bounded output window at any instant.
    let id: ResourceId = "mem://audio/too-long".parse().unwrap();
    let mut windows = 0;
    let error = audio::windows(
        wav((audio::MAX_SAMPLES / 2 + 1) as usize, 8000),
        &id,
        |_, pcm| {
            assert!(pcm.len() <= audio::WINDOW);
            windows += 1;
            Ok(())
        },
    )
    .unwrap_err();
    assert_eq!(windows, 0);
    assert!(matches!(
        error,
        Error::UnsupportedAudio {
            reason: "decoded audio exceeds 1048.576 second duration cap",
            ..
        }
    ));
}
