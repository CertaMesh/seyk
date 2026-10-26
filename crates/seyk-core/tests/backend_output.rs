mod common;

use common::FakeEmbedder;
use seyk_core::{
    Dimension, Embedder, Embedding, EmbeddingRequest, EmbeddingTask, Modality, ModelIdentity,
    SearchEngine,
};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

type Reply = (usize, seyk_embed::Result<Vec<Embedding>>);

/// Each response is paired with the expected request count. This also detects
/// extra retries or dropped requests at the public backend boundary.
struct Scripted {
    inner: FakeEmbedder,
    replies: Mutex<VecDeque<Reply>>,
}

impl Scripted {
    fn new(replies: impl IntoIterator<Item = Reply>) -> Self {
        Self {
            inner: FakeEmbedder::new(Dimension::D128, "backend-output"),
            replies: Mutex::new(replies.into_iter().collect()),
        }
    }

    fn assert_finished(&self) {
        assert!(self.replies.lock().unwrap().is_empty());
    }
}

impl Embedder for Scripted {
    fn identity(&self) -> &ModelIdentity {
        self.inner.identity()
    }

    fn supports(&self, modality: Modality) -> bool {
        self.inner.supports(modality)
    }

    fn embed_batch(&self, requests: &[EmbeddingRequest<'_>]) -> seyk_embed::Result<Vec<Embedding>> {
        let (count, result) = self
            .replies
            .lock()
            .unwrap()
            .pop_front()
            .expect("unexpected backend call");
        assert_eq!(requests.len(), count);
        assert!(
            requests
                .iter()
                .all(|r| r.task == EmbeddingTask::SearchDocument)
        );
        result
    }
}

fn vector(width: usize) -> Embedding {
    Embedding::from_raw(vec![1.0; width], Dimension::try_from(width).unwrap()).unwrap()
}

fn batch_error() -> seyk_embed::Result<Vec<Embedding>> {
    Err(seyk_embed::Error::Embedding(
        "synthetic batch failure".into(),
    ))
}

#[test]
fn singleton_retries_require_exactly_one_vector() {
    for count in [0, 1, 2, 3] {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "alpha garden").unwrap();
        std::fs::write(dir.path().join("b.txt"), "beta harbor").unwrap();
        let backend = Arc::new(Scripted::new([
            (2, batch_error()),
            (1, Ok(vec![vector(128); count])),
            (1, Ok(vec![vector(128)])),
        ]));
        let engine = SearchEngine::builder()
            .shared_embedder(backend.clone())
            .build()
            .unwrap();
        let report = engine.index().paths([dir.path()]).execute().unwrap();
        let valid = usize::from(count == 1);
        assert_eq!(report.sync.resources_updated, 1 + valid, "count={count}");
        assert_eq!(report.sync.segments_embedded, 1 + valid, "count={count}");
        assert_eq!(report.sync.skipped.len(), 1 - valid, "count={count}");
        backend.assert_finished();
    }
}

#[test]
fn wrong_width_bulk_output_retries_without_caching_any_bulk_vectors() {
    for width in [127, 129] {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "alpha garden").unwrap();
        std::fs::write(dir.path().join("b.txt"), "beta harbor").unwrap();
        let backend = Arc::new(Scripted::new([
            (2, Ok(vec![vector(128), vector(width)])),
            (1, Ok(vec![vector(128)])),
            (1, Ok(vec![vector(width)])),
        ]));
        let engine = SearchEngine::builder()
            .shared_embedder(backend.clone())
            .build()
            .unwrap();
        let report = engine.index().paths([dir.path()]).execute().unwrap();
        assert_eq!(report.sync.resources_updated, 1, "width={width}");
        assert_eq!(report.sync.segments_embedded, 1, "width={width}");
        assert_eq!(report.sync.skipped.len(), 1, "width={width}");
        assert!(report.sync.skipped[0].reason.contains("dimension mismatch"));
        backend.assert_finished();
    }
}

#[test]
fn wrong_bulk_counts_retry_each_request() {
    for count in [0, 1, 3] {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("a.txt"), "alpha garden").unwrap();
        std::fs::write(dir.path().join("b.txt"), "beta harbor").unwrap();
        let backend = Arc::new(Scripted::new([
            (2, Ok(vec![vector(128); count])),
            (1, Ok(vec![vector(128)])),
            (1, Ok(vec![vector(128)])),
        ]));
        let engine = SearchEngine::builder()
            .shared_embedder(backend.clone())
            .build()
            .unwrap();
        let report = engine.index().paths([dir.path()]).execute().unwrap();
        assert_eq!(report.sync.resources_updated, 2, "count={count}");
        assert_eq!(report.sync.segments_embedded, 2, "count={count}");
        assert!(report.sync.skipped.is_empty());
        backend.assert_finished();
    }
}

fn engine_with_cache(backend: &Arc<Scripted>, sqlite: bool, dir: &std::path::Path) -> SearchEngine {
    let builder = SearchEngine::builder().shared_embedder(backend.clone());
    let builder = if sqlite {
        builder.cache(seyk_core::SqliteCache::open(dir.join("cache.db")).unwrap())
    } else {
        builder
    };
    builder.build().unwrap()
}

#[test]
fn successful_outputs_preserve_order_and_exact_bytes_in_both_caches() {
    // Preserved input deliberately bypasses normalization. Validation must
    // return these exact bits, including negative zero, in request order.
    let mut first = vec![0.0f32; 128];
    first[..3].copy_from_slice(&[3.0, 4.0, -0.0]);
    let mut second = vec![0.0f32; 128];
    second[..3].copy_from_slice(&[-0.5, 0.25, -0.0]);
    let embedding = |values: &Vec<f32>| Embedding::from_preserved(values.clone()).unwrap();
    for sqlite in [false, true] {
        for retry in [false, true] {
            let dir = tempfile::tempdir().unwrap();
            let a = dir.path().join("a.txt");
            let b = dir.path().join("b.txt");
            std::fs::write(&a, "alpha garden").unwrap();
            std::fs::write(&b, "beta harbor").unwrap();
            let replies = if retry {
                vec![
                    (2, batch_error()),
                    (1, Ok(vec![embedding(&first)])),
                    (1, Ok(vec![embedding(&second)])),
                ]
            } else {
                vec![(2, Ok(vec![embedding(&first), embedding(&second)]))]
            };
            let backend = Arc::new(Scripted::new(replies));
            let engine = engine_with_cache(&backend, sqlite, dir.path());
            let report = engine.index().paths([dir.path()]).execute().unwrap();
            assert_eq!(report.sync.resources_updated, 2);
            assert_eq!(report.sync.segments_embedded, 2);
            assert!(report.sync.skipped.is_empty());
            for (path, expected) in [(&a, &first), (&b, &second)] {
                let source =
                    seyk_core::ResourceId::from_path(path.canonicalize().unwrap()).unwrap();
                let stored = engine
                    .with_cache(|c| c.vectors_for(engine.profile_id(), &[source].into()))
                    .unwrap();
                assert_eq!(stored.len(), 1);
                let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
                assert_eq!(bits(&stored[0].vector), bits(expected));
            }
            let warm = engine.index().paths([dir.path()]).execute().unwrap();
            assert_eq!(warm.sync.resources_updated, 0);
            assert_eq!(warm.sync.segments_embedded, 0);
            backend.assert_finished();
        }
    }
}

#[test]
fn malformed_shared_inputs_leave_all_owners_retryable_in_both_caches() {
    for sqlite in [false, true] {
        for previously_indexed in [false, true] {
            for (count, width) in [(0, 128), (2, 128), (1, 127), (1, 129)] {
                let dir = tempfile::tempdir().unwrap();
                let mut owners = Vec::new();
                for name in ["a", "b"] {
                    let folder = dir.path().join(name);
                    std::fs::create_dir(&folder).unwrap();
                    owners.push(folder.join("note.txt"));
                }
                let good = dir.path().join("good.txt");
                let mut replies = Vec::new();
                if previously_indexed {
                    replies.push((2, Ok(vec![vector(128), vector(128)])));
                }
                replies.extend([
                    (2, batch_error()),
                    (1, Ok(vec![vector(width); count])),
                    (1, Ok(vec![vector(128)])),
                    // Only the shared failed input should be requested on repair.
                    (1, Ok(vec![vector(128)])),
                ]);
                let backend = Arc::new(Scripted::new(replies));
                let engine = engine_with_cache(&backend, sqlite, dir.path());
                if previously_indexed {
                    for path in &owners {
                        std::fs::write(path, "old shared text").unwrap();
                    }
                    std::fs::write(&good, "old independent text").unwrap();
                    let initial = engine.index().paths([dir.path()]).execute().unwrap();
                    assert_eq!(initial.sync.resources_updated, 3);
                    assert_eq!(initial.sync.segments_embedded, 2);
                }
                for path in &owners {
                    std::fs::write(path, "changed shared text requiring a new embedding").unwrap();
                }
                std::fs::write(&good, "changed independent text").unwrap();
                let ids: Vec<_> = owners
                    .iter()
                    .map(|p| seyk_core::ResourceId::from_path(p.canonicalize().unwrap()).unwrap())
                    .collect();
                let prior: Vec<_> = ids
                    .iter()
                    .map(|id| {
                        engine
                            .with_cache(|c| c.resource(engine.profile_id(), id))
                            .unwrap()
                    })
                    .collect();
                let failed = engine.index().paths([dir.path()]).execute().unwrap();
                assert_eq!(failed.sync.resources_updated, 1);
                assert_eq!(failed.sync.segments_embedded, 1);
                assert_eq!(failed.sync.skipped.len(), 2);
                for (id, before) in ids.iter().zip(&prior) {
                    assert!(failed.sync.skipped.iter().any(|s| &s.resource == id));
                    let after = engine
                        .with_cache(|c| c.resource(engine.profile_id(), id))
                        .unwrap();
                    assert_eq!(
                        &after, before,
                        "failed output must not mark an owner current"
                    );
                }
                let repaired = engine.index().paths([dir.path()]).execute().unwrap();
                assert_eq!(repaired.sync.resources_updated, 2);
                assert_eq!(
                    repaired.sync.segments_embedded, 1,
                    "shared input stays deduplicated"
                );
                assert!(repaired.sync.skipped.is_empty());
                let warm = engine.index().paths([dir.path()]).execute().unwrap();
                assert_eq!(warm.sync.resources_updated, 0);
                assert_eq!(warm.sync.segments_embedded, 0);
                backend.assert_finished();
            }
        }
    }
}

#[test]
fn one_request_batches_validate_once_and_never_cache_malformed_outputs() {
    for (count, width) in [(0, 128), (1, 128), (2, 128), (1, 127), (1, 129)] {
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("note.txt"), "alpha garden").unwrap();
        let backend = Arc::new(Scripted::new([(1, Ok(vec![vector(width); count]))]));
        let engine = SearchEngine::builder()
            .shared_embedder(backend.clone())
            .build()
            .unwrap();
        let report = engine.index().paths([dir.path()]).execute().unwrap();
        let valid = usize::from(count == 1 && width == 128);
        assert_eq!(report.sync.resources_updated, valid);
        assert_eq!(report.sync.segments_embedded, valid);
        assert_eq!(report.sync.skipped.len(), 1 - valid);
        let stats = engine.with_cache(|c| c.stats()).unwrap();
        assert_eq!(stats.profiles[0].embeddings, valid as u64);
        assert_eq!(stats.profiles[0].resources, valid as u64);
        backend.assert_finished();
    }
}
