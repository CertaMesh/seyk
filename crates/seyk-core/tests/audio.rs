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
    fn input_identity(&self, m: Modality) -> seyk_embed::Result<&str> {
        Ok(if m == Modality::Audio {
            self.policy
        } else {
            ""
        })
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
            )
        ]
    );
    let mut short = Vec::new();
    audio::windows(wav(1, 16000), &id, |locator, pcm| {
        short.push((locator, pcm.len()));
        Ok(())
    })
    .unwrap();
    assert_eq!(
        short,
        vec![(
            ResourceLocator::Time {
                start_ms: 0,
                end_ms: 1
            },
            320
        )]
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
    assert_eq!(result.hits.len(), 6);
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
    assert!(error.to_string().contains("configure an audio extractor"));
    // Different file content: two successful windows then a failure. Binding
    // must remain entirely the previous six windows, never a partial version.
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
    // The old binding survives, but a changed/unvalidated source is ineligible.
    assert!(after.hits.is_empty());
    assert_eq!(
        engine
            .with_cache(|c| c.resource(engine.profile_id(), &result.hits[0].resource))
            .unwrap()
            .unwrap()
            .segments
            .len(),
        6
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
    let source = audio::VerifiedReader::new(
        Box::new(std::io::Cursor::new(wav(8001, 8000))),
        &id,
        seyk_core::discovery::audio_byte_limit(1).unwrap(),
    )
    .unwrap();
    let error = audio::windows_verified(source, &id, 1, |_, pcm| {
        assert!(pcm.len() <= audio::WINDOW);
        windows += 1;
        Ok(())
    })
    .unwrap_err();
    assert_eq!(windows, 0);
    assert!(matches!(
        error,
        Error::UnsupportedAudio {
            reason: "decoded audio exceeds configured duration limit",
            ..
        }
    ));
}

#[test]
fn limits_filter_both_cache_fast_paths_and_cached_only_search() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clip.wav");
    let db = dir.path().join("cache.db");
    std::fs::write(&path, wav(32001, 16000)).unwrap();
    let model = Arc::new(Model::new("policy"));
    let make = |seconds| {
        SearchEngine::builder()
            .embedder(model.clone())
            .max_audio_duration_secs(seconds)
            .cache(SqliteCache::open(&db).unwrap())
            .build()
            .unwrap()
    };
    let high = make(3);
    let first = high.search("q").paths([&path]).execute().unwrap();
    assert_eq!(first.hits.len(), 1);
    let id = first.hits[0].resource.clone();
    let old = high
        .with_cache(|c| c.resource(high.profile_id(), &id))
        .unwrap()
        .unwrap()
        .segments;
    drop(high);
    for touch in [false, true] {
        if touch {
            std::fs::OpenOptions::new()
                .write(true)
                .open(&path)
                .unwrap()
                .set_times(std::fs::FileTimes::new().set_modified(
                    std::time::SystemTime::UNIX_EPOCH
                        + std::time::Duration::from_secs(1_700_000_000),
                ))
                .unwrap();
        }
        let low = make(2);
        assert!(
            low.search("q")
                .paths([&path])
                .sync(false)
                .execute()
                .unwrap()
                .hits
                .is_empty()
        );
        let result = low.search("q").paths([&path]).execute().unwrap();
        assert!(result.hits.is_empty());
        assert_eq!(result.stats.sync.skipped.len(), 1);
        assert!(low.extract_audio_samples(&first.hits[0]).is_err());
    }
    let high = make(4);
    let again = high.search("q").paths([&path]).execute().unwrap();
    assert_eq!(again.hits.len(), 1);
    assert_eq!(again.stats.sync.segments_embedded, 0);
    assert_eq!(
        high.with_cache(|c| c.resource(high.profile_id(), &id))
            .unwrap()
            .unwrap()
            .segments,
        old
    );
    assert_eq!(model.calls.lock().unwrap().len(), 1);
}

#[test]
fn changed_during_decode_leaves_no_binding_and_no_slice() {
    use std::io::{Seek, SeekFrom, Write};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clip.wav");
    std::fs::write(&path, wav(16000 * 100, 16000)).unwrap();
    let id = ResourceId::from_path(&path).unwrap();
    let source =
        audio::VerifiedReader::new(Box::new(std::fs::File::open(&path).unwrap()), &id, u64::MAX)
            .unwrap();
    let mut windows = 0;
    let error = audio::windows_verified(source, &id, 7200, |_, _| {
        windows += 1;
        if windows == 1 {
            let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
            f.seek(SeekFrom::Start(2 * 1024 * 1024)).unwrap();
            f.write_all(&[0xff; 32]).unwrap();
        }
        Ok(())
    })
    .unwrap_err();
    assert!(windows > 0);
    assert!(matches!(error, Error::ChangedWhileReading { .. }));
    let source =
        audio::VerifiedReader::new(Box::new(std::fs::File::open(&path).unwrap()), &id, u64::MAX)
            .unwrap();
    let mut f = std::fs::OpenOptions::new().write(true).open(&path).unwrap();
    f.seek(SeekFrom::Start(44)).unwrap();
    f.write_all(&[0xff; 32]).unwrap();
    assert!(matches!(
        audio::slice_verified(
            source,
            &id,
            &ResourceLocator::Time {
                start_ms: 0,
                end_ms: 1000
            },
            7200
        ),
        Err(Error::ChangedWhileReading { .. })
    ));
    // Mutation through the embedding callback occurs after a validated batch,
    // before the last chunk is consumed. No resource can be committed.
    struct MutatingModel {
        inner: Model,
        path: std::path::PathBuf,
        changed: std::sync::atomic::AtomicBool,
    }
    impl Embedder for MutatingModel {
        fn identity(&self) -> &ModelIdentity {
            self.inner.identity()
        }
        fn input_identity(&self, modality: Modality) -> seyk_embed::Result<&str> {
            self.inner.input_identity(modality)
        }
        fn supports(&self, m: Modality) -> bool {
            self.inner.supports(m)
        }
        fn embed_batch(&self, r: &[EmbeddingRequest<'_>]) -> seyk_embed::Result<Vec<Embedding>> {
            if !self.changed.swap(true, Ordering::Relaxed) {
                let mut f = std::fs::OpenOptions::new()
                    .write(true)
                    .open(&self.path)
                    .unwrap();
                f.seek(SeekFrom::Start(2 * 1024 * 1024)).unwrap();
                f.write_all(&[0xaa; 32]).unwrap();
            }
            self.inner.embed_batch(r)
        }
    }
    let engine = SearchEngine::builder()
        .embedder(MutatingModel {
            inner: Model::new(""),
            path: path.clone(),
            changed: std::sync::atomic::AtomicBool::new(false),
        })
        .batch_size(1)
        .build()
        .unwrap();
    let report = engine.index().paths([&path]).execute().unwrap();
    assert_eq!(report.sync.resources_updated, 0);
    assert_eq!(report.sync.skipped.len(), 1);
    assert!(
        report.sync.skipped[0]
            .reason
            .contains("changed while reading")
    );
    assert!(
        engine
            .with_cache(|c| c.resource(engine.profile_id(), &id))
            .unwrap()
            .is_none()
    );
}

#[test]
fn audio_uses_provider_open_without_whole_file_reads() {
    struct StreamingOnly;
    impl seyk_core::ResourceProvider for StreamingOnly {
        fn list(
            &self,
            s: &seyk_core::ResourceScope,
        ) -> seyk_core::Result<Vec<seyk_core::ProvidedResource>> {
            seyk_core::ResourceProvider::list(&seyk_core::FsProvider, s)
        }
        fn read(&self, _: &ResourceId) -> seyk_core::Result<Vec<u8>> {
            panic!("audio must not read the whole resource")
        }
        fn open(
            &self,
            id: &ResourceId,
        ) -> seyk_core::Result<Box<dyn seyk_core::provider::ReadSeek>> {
            seyk_core::ResourceProvider::open(&seyk_core::FsProvider, id)
        }
        fn version(&self, id: &ResourceId) -> seyk_core::Result<seyk_core::ResourceVersion> {
            seyk_core::ResourceProvider::version(&seyk_core::FsProvider, id)
        }
    }
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clip.wav");
    std::fs::write(&path, wav(16001, 16000)).unwrap();
    let engine = SearchEngine::builder()
        .provider(StreamingOnly)
        .embedder(Model::new("policy"))
        .build()
        .unwrap();
    let result = engine.search("q").paths([path]).execute().unwrap();
    assert_eq!(result.hits.len(), 1);
    assert_eq!(
        engine.extract_audio_samples(&result.hits[0]).unwrap().len(),
        16001
    );
}

struct SpeechSpy(Arc<AtomicUsize>);
impl audio::SpeechToText for SpeechSpy {
    fn implementation(&self) -> &'static str {
        "test-spy"
    }
    fn manifest_sha256(&self) -> String {
        "test-manifest".into()
    }
    fn transcribe(&self, samples: &[f32], _: Option<&str>) -> seyk_core::Result<audio::Transcript> {
        self.0.fetch_add(1, Ordering::SeqCst);
        assert!(samples.len() <= 480000);
        Ok(audio::Transcript {
            text: "private sentinel transcript".into(),
            language: Some("en".into()),
            no_speech: false,
        })
    }
}
#[test]
fn stt_is_explicit_verified_and_never_cached_or_previewed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("clip.wav");
    let db = dir.path().join("cache.db");
    std::fs::write(&path, wav(160001, 16000)).unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let engine = SearchEngine::builder()
        .embedder(Arc::new(Model::new("policy1")))
        .cache(SqliteCache::open(&db).unwrap())
        .audio_extractor(audio::AudioExtractor::new(SpeechSpy(calls.clone())))
        .build()
        .unwrap();
    engine.index().paths([&path]).execute().unwrap();
    let hits = engine
        .search("query")
        .paths([&path])
        .mode(SearchMode::Semantic)
        .limit(20)
        .execute()
        .unwrap()
        .hits;
    assert!(!hits.is_empty());
    assert!(hits.iter().all(|h| h.preview.is_none()));
    assert_eq!(calls.load(Ordering::SeqCst), 0);
    let mut hit = hits[0].clone();
    hit.locator = ResourceLocator::Time {
        start_ms: 8000,
        end_ms: 10001,
    };
    let extracted = engine.extract(&hit).unwrap();
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert!(!format!("{extracted:?}").contains("private sentinel"));
    let provenance = extracted.provenance.unwrap().speech.unwrap();
    assert_eq!(
        (
            provenance.sample_start,
            provenance.sample_end,
            provenance.sample_rate
        ),
        (128000, 160001, 16000)
    );
    assert_eq!(provenance.requested_locator, hit.locator);
    assert_eq!(provenance.language_detected.as_deref(), Some("en"));
    assert_eq!(provenance.fidelity, "machine_transcription");
    hit.locator = ResourceLocator::Time {
        start_ms: 0,
        end_ms: 30001,
    };
    assert!(matches!(
        engine.extract(&hit),
        Err(Error::AudioIntervalTooLong)
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    std::fs::write(&path, wav(160002, 16000)).unwrap();
    assert!(matches!(
        engine.extract(&hits[0]),
        Err(Error::StaleResource { .. })
    ));
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    drop(engine);
    let persisted = std::fs::read(db).unwrap();
    assert!(
        !persisted
            .windows(27)
            .any(|b| b == b"private sentinel transcript")
    );
}

#[test]
fn persisted_audio_identities_match_pre_lazy_hashing_baseline() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("signal.wav");
    let text = dir.path().join("text.txt");
    let db = dir.path().join("cache.db");
    std::fs::write(&path, wav(200, 16000)).unwrap();
    std::fs::write(&text, "unchanged text").unwrap();
    let policy = "60fe96207a5119529b8f6d86a53e2fac20fdd06d8b66e9a24dae0a61e8fd04b6";
    let model = Arc::new(Model::new(policy));
    let engine = SearchEngine::builder()
        .embedder(model)
        .cache(SqliteCache::open(&db).unwrap())
        .build()
        .unwrap();
    engine.index().paths([&path, &text]).execute().unwrap();
    let id = ResourceId::from_path(std::fs::canonicalize(&path).unwrap()).unwrap();
    let cached = engine
        .with_cache(|c| c.resource(engine.profile_id(), &id))
        .unwrap()
        .unwrap();
    // Captured by running this fixture against f9464e5 before lazy hashing.
    assert_eq!(engine.profile_id(), "59413168d46a5954340dcc36");
    assert_eq!(
        cached.segments[0].input_hash,
        "7fa4293275e52ff8cca42fc8c59cec67d78fb5a4528187953a987390782f852e"
    );
    assert_eq!(
        cached.descriptor_hash.as_deref(),
        Some("bfac8c3a6a86c657fc46d13ff2ed8a217eed48ec532996abc475d2581796f75f")
    );
    drop(engine);
    let model = Arc::new(Model::new(policy));
    let engine = SearchEngine::builder()
        .embedder(Box::new(model.clone()))
        .cache(SqliteCache::open(&db).unwrap())
        .build()
        .unwrap();
    for sync in [false, true] {
        let result = engine
            .search("query")
            .paths([&path, &text])
            .sync(sync)
            .execute()
            .unwrap();
        assert!(!result.hits.is_empty());
        assert!(result.stats.sync.skipped.is_empty());
        assert_eq!(result.stats.sync.resources_updated, 0);
    }
    assert!(model.calls.lock().unwrap().is_empty());
}

/// Instrument the engine's identity admission seam without model inference.
struct LazyPolicy {
    inner: Model,
    weights: std::path::PathBuf,
    digest: std::sync::OnceLock<String>,
    audio_identity_calls: AtomicUsize,
}
impl LazyPolicy {
    fn new(weights: &std::path::Path) -> Self {
        Self {
            inner: Model::new("unused"),
            weights: weights.into(),
            digest: std::sync::OnceLock::new(),
            audio_identity_calls: AtomicUsize::new(0),
        }
    }
}
impl Embedder for LazyPolicy {
    fn identity(&self) -> &ModelIdentity {
        self.inner.identity()
    }
    fn supports(&self, modality: Modality) -> bool {
        self.inner.supports(modality)
    }
    fn input_identity(&self, modality: Modality) -> seyk_embed::Result<&str> {
        if modality != Modality::Audio {
            return Ok("");
        }
        self.audio_identity_calls.fetch_add(1, Ordering::Relaxed);
        if let Some(digest) = self.digest.get() {
            return Ok(digest);
        }
        let bytes = std::fs::read(&self.weights)
            .map_err(|_| seyk_embed::Error::ModelNotFound("test missing audio asset".into()))?;
        Ok(self
            .digest
            .get_or_init(|| blake3::hash(&bytes).to_hex().to_string()))
    }
    fn embed_batch(&self, requests: &[EmbeddingRequest<'_>]) -> seyk_embed::Result<Vec<Embedding>> {
        self.inner.embed_batch(requests)
    }
}

#[test]
fn text_only_search_never_requests_audio_identity() {
    let dir = tempfile::tempdir().unwrap();
    let text = dir.path().join("text.txt");
    std::fs::write(&text, "a synthetic text document").unwrap();
    let model = Arc::new(LazyPolicy::new(&dir.path().join("missing-weights")));
    let engine = SearchEngine::builder()
        .embedder(Box::new(model.clone()))
        .build()
        .unwrap();
    for sync in [true, false] {
        assert!(
            !engine
                .search("query")
                .paths([&text])
                .sync(sync)
                .execute()
                .unwrap()
                .hits
                .is_empty()
        );
    }
    assert_eq!(model.audio_identity_calls.load(Ordering::Relaxed), 0);
    assert!(model.digest.get().is_none());
}

#[test]
fn same_size_encoder_tamper_is_not_admitted_by_either_cache_path() {
    let dir = tempfile::tempdir().unwrap();
    let audio = dir.path().join("signal.wav");
    let weights = dir.path().join("weights");
    let db = dir.path().join("cache.db");
    std::fs::write(&audio, wav(200, 16000)).unwrap();
    std::fs::write(&weights, b"old-weights").unwrap();
    let model = Arc::new(LazyPolicy::new(&weights));
    let engine = SearchEngine::builder()
        .embedder(model.clone())
        .cache(SqliteCache::open(&db).unwrap())
        .build()
        .unwrap();
    let profile = engine.profile_id().to_owned();
    let result = engine.search("query").paths([&audio]).execute().unwrap();
    assert!(!result.hits.is_empty());
    assert!(model.digest.get().is_some());
    let id = result.hits[0].resource.clone();
    let old = engine
        .with_cache(|c| c.resource(&profile, &id))
        .unwrap()
        .unwrap();
    drop(engine);
    let metadata = std::fs::metadata(&weights).unwrap();
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .open(&weights)
        .unwrap();
    use std::io::Write;
    file.write_all(b"new-weights").unwrap();
    file.set_times(std::fs::FileTimes::new().set_modified(metadata.modified().unwrap()))
        .unwrap();
    let after = file.metadata().unwrap();
    assert_eq!(metadata.len(), after.len());
    assert_eq!(metadata.modified().unwrap(), after.modified().unwrap());
    #[cfg(unix)]
    {
        use std::os::unix::fs::MetadataExt;
        assert_eq!(metadata.ino(), after.ino());
    }
    for sync in [false, true] {
        let changed = Arc::new(LazyPolicy::new(&weights));
        // Fail fresh inference so an incorrectly admitted old vector is visible.
        changed.inner.fail_after.store(0, Ordering::Relaxed);
        let engine = SearchEngine::builder()
            .embedder(changed.clone())
            .cache(SqliteCache::open(&db).unwrap())
            .build()
            .unwrap();
        assert_eq!(engine.profile_id(), profile);
        let result = engine
            .search("query")
            .paths([&audio])
            .sync(sync)
            .execute()
            .unwrap();
        assert!(result.hits.is_empty());
        assert!(changed.audio_identity_calls.load(Ordering::Relaxed) > 0);
        assert_ne!(changed.digest.get(), model.digest.get());
        assert_eq!(changed.inner.calls.lock().unwrap().len(), usize::from(sync));
        // The old rows are retained but ineligible, not rewritten or deleted.
        let stored = engine
            .with_cache(|c| c.resource(&profile, &id))
            .unwrap()
            .unwrap();
        assert_eq!(stored.descriptor_hash, old.descriptor_hash);
        assert_eq!(stored.segments[0].input_hash, old.segments[0].input_hash);
    }
    // A failed identity cannot admit cached audio. A subsequent explicit search
    // can retry, while text resources in the same search remain usable.
    let text = dir.path().join("text.txt");
    std::fs::write(&text, "synthetic text remains searchable").unwrap();
    let priming = SearchEngine::builder()
        .embedder(model.clone())
        .cache(SqliteCache::open(&db).unwrap())
        .build()
        .unwrap();
    let text_id = ResourceId::from_path(std::fs::canonicalize(&text).unwrap()).unwrap();
    assert!(
        priming
            .search("query")
            .paths([&text])
            .execute()
            .unwrap()
            .hits
            .iter()
            .any(|hit| hit.resource == text_id)
    );
    drop(priming);
    std::fs::remove_file(&weights).unwrap();
    let missing = Arc::new(LazyPolicy::new(&weights));
    let engine = SearchEngine::builder()
        .embedder(missing.clone())
        .cache(SqliteCache::open(&db).unwrap())
        .build()
        .unwrap();
    for sync in [false, true] {
        let result = engine
            .search("query")
            .paths([&audio, &text])
            .sync(sync)
            .execute()
            .unwrap();
        assert!(result.hits.iter().all(|hit| hit.resource != id));
        assert!(result.hits.iter().any(|hit| hit.resource == text_id));
        assert!(
            result
                .stats
                .sync
                .skipped
                .iter()
                .any(|skipped| skipped.resource == id)
        );
        assert_eq!(
            result
                .stats
                .sync
                .skipped_counts
                .iter()
                .map(|count| count.count)
                .sum::<usize>(),
            result.stats.sync.skipped.len()
        );
        assert!(missing.digest.get().is_none());
    }
    std::fs::write(&weights, b"old-weights").unwrap();
    assert!(
        !engine
            .search("query")
            .paths([&audio])
            .sync(false)
            .execute()
            .unwrap()
            .hits
            .is_empty()
    );
}
