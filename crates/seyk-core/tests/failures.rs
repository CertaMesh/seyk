mod common;

use std::fs;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use common::FakeEmbedder;
use seyk_core::{
    Dimension, Embedder, Embedding, MemoryCache, Modality, ModelIdentity, ResourceInput,
    SearchEngine, SqliteCache,
};
use tempfile::TempDir;

/// Fails any batch containing "POISON" while `poisoned` is set.
struct Flaky {
    inner: FakeEmbedder,
    poisoned: AtomicBool,
}

impl Embedder for Flaky {
    fn identity(&self) -> &ModelIdentity {
        self.inner.identity()
    }
    fn supports(&self, m: Modality) -> bool {
        self.inner.supports(m)
    }
    fn embed_query(&self, q: &str) -> seyk_embed::Result<Embedding> {
        self.inner.embed_query(q)
    }
    fn embed_resources(&self, inputs: &[ResourceInput<'_>]) -> seyk_embed::Result<Vec<Embedding>> {
        let poison = inputs
            .iter()
            .any(|i| matches!(i, ResourceInput::Text { text, .. } if text.contains("POISON")));
        if poison && self.poisoned.load(Ordering::SeqCst) {
            return Err(seyk_embed::Error::Embedding("poisoned input".into()));
        }
        self.inner.embed_resources(inputs)
    }

    fn embed_batch(
        &self,
        requests: &[seyk_embed::EmbeddingRequest<'_>],
    ) -> seyk_embed::Result<Vec<Embedding>> {
        if requests
            .iter()
            .all(|r| r.task == seyk_embed::EmbeddingTask::SearchDocument)
        {
            let inputs: Vec<_> = requests.iter().map(|r| r.input.clone()).collect();
            return self.embed_resources(&inputs);
        }
        requests
            .iter()
            .map(|r| match &r.input {
                ResourceInput::Text { text, .. }
                    if r.task == seyk_embed::EmbeddingTask::SearchQuery =>
                {
                    self.embed_query(text)
                }
                _ => self
                    .embed_resources(std::slice::from_ref(&r.input))
                    .map(|mut v| v.remove(0)),
            })
            .collect()
    }
}

/// The same note in two folders shares one model input. When that input fails,
/// neither copy may be recorded as indexed, or the second would never retry.
fn shared_failed_input_marks_every_owner_incomplete<C>(cache_for: impl Fn(&TempDir) -> C)
where
    C: seyk_core::EmbeddingCache + 'static,
{
    let temp = TempDir::new().unwrap();
    for dir in ["a", "b"] {
        fs::create_dir(temp.path().join(dir)).unwrap();
        fs::write(temp.path().join(dir).join("note.md"), "POISON the well").unwrap();
    }
    fs::write(temp.path().join("other.md"), "a perfectly fine note").unwrap();
    let flaky = Arc::new(Flaky {
        inner: FakeEmbedder::new(Dimension::D256, "r1"),
        poisoned: AtomicBool::new(true),
    });
    let engine = SearchEngine::builder()
        .shared_embedder(flaky.clone())
        .cache(cache_for(&temp))
        .build()
        .unwrap();

    let first = engine.index().paths([temp.path()]).execute().unwrap();
    let mut skipped: Vec<_> = first
        .sync
        .skipped
        .iter()
        .map(|s| {
            s.path
                .as_ref()
                .unwrap()
                .canonicalize()
                .unwrap()
                .strip_prefix(temp.path().canonicalize().unwrap())
                .unwrap()
                .to_owned()
        })
        .collect();
    skipped.sort();
    assert_eq!(
        skipped,
        ["a/note.md", "b/note.md"].map(std::path::PathBuf::from)
    );
    assert_eq!(first.sync.resources_updated, 1, "only other.md is complete");

    flaky.poisoned.store(false, Ordering::SeqCst);
    let second = engine.index().paths([temp.path()]).execute().unwrap();
    assert_eq!(second.sync.resources_updated, 2, "both copies retried");
    assert_eq!(
        second.sync.segments_embedded, 1,
        "shared input embedded once"
    );
    assert!(second.sync.skipped.is_empty());
}

#[test]
fn memory_cache() {
    shared_failed_input_marks_every_owner_incomplete(|_| MemoryCache::new());
}

#[test]
fn sqlite_cache() {
    shared_failed_input_marks_every_owner_incomplete(|t| {
        SqliteCache::open(t.path().join("cache.sqlite3")).unwrap()
    });
}
